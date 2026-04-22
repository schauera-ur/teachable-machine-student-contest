# Grad-CAM explainer for Teachable Machine models
# Andreas Schauer <andreas.schauer@ur.de> 2026
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

np.set_printoptions(suppress=True)


def find_model_dir(username):
    """Resolve the directory containing keras_model.h5 for a given username."""
    base = f"models/{username}"
    candidate = os.path.join(base, "converted_keras")
    if os.path.isfile(os.path.join(candidate, "keras_model.h5")):
        return candidate
    if os.path.isfile(os.path.join(base, "keras_model.h5")):
        return base
    return None


def load_class_names(labels_path):
    """Parse labels.txt lines like '0 Phone' into a list indexed by class id."""
    with open(labels_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    entries = sorted((int(l.split()[0]), l.split(None, 1)[1].strip()) for l in lines)
    return [name for _, name in entries]


def preprocess(img_array):
    """Resize to 224x224 and normalise to [-1, 1] (Teachable Machine convention)."""
    image = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32)
    return (arr / 127.5) - 1


def collect_images(folder="images"):
    """Return sorted list of image paths from the images/ folder."""
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print(f"Created folder '{folder}'. Place your images there and re-run.")
        sys.exit(0)
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(supported)
    )
    return paths


def find_last_spatial_layer(model):
    """Recursively scan the model tree and return the last layer with 4D spatial output."""
    last = [None]

    def scan(layers):
        for layer in layers:
            try:
                shape = layer.output_shape
                if isinstance(shape, (list, tuple)) and len(shape) == 4:
                    h, w = shape[1], shape[2]
                    if h is not None and w is not None and h > 1 and w > 1:
                        last[0] = layer
            except Exception:
                pass
            if hasattr(layer, "layers"):
                scan(layer.layers)

    scan(model.layers)
    return last[0]


def find_path_to_layer(root_model, target_layer):
    """
    Locate target_layer anywhere in the model tree.

    Returns (direct_parent, path) where:
      - direct_parent: the sub-model whose .layers list directly contains target_layer
      - path: list of (parent_model, child_index) from root_model down to direct_parent
        (empty list when target_layer is a direct child of root_model)
    Returns (None, None) if not found.
    """
    for i, layer in enumerate(root_model.layers):
        if layer is target_layer:
            return root_model, []
        if hasattr(layer, "layers"):
            direct_parent, sub_path = find_path_to_layer(layer, target_layer)
            if direct_parent is not None:
                return direct_parent, [(root_model, i)] + sub_path
    return None, None


def build_gradcam_forward(model):
    """
    Build a callable gradcam_forward(x) -> (conv_outputs, predictions).

    Works for arbitrarily nested sub-model architectures (e.g. Teachable Machine
    wrapping MobileNetV2 inside one or more Sequential wrappers).

    Strategy:
      1. Locate the target layer and its direct parent sub-model.
      2. Build a two-output feature_extractor on that parent:
         parent.input -> [target.output, parent.output]
      3. Build a recursive forward function that threads the call through
         every nesting level by applying the pre/post layers at each level.

    Returns (gradcam_forward, target_layer_name).
    """
    target_layer = find_last_spatial_layer(model)

    if target_layer is None:
        raise ValueError(
            "No convolutional layer with spatial output found in the model. "
            "Grad-CAM requires at least one Conv2D-like layer."
        )

    print(
        f"  Grad-CAM target layer: '{target_layer.name}'  "
        f"output shape: {target_layer.output_shape}"
    )

    direct_parent, path = find_path_to_layer(model, target_layer)
    if direct_parent is None:
        raise ValueError(f"Layer '{target_layer.name}' could not be located in the model tree.")

    # feature_extractor lives at the level that directly holds target_layer,
    # so target_layer.output IS in direct_parent's computation graph.
    feature_extractor = tf.keras.Model(
        inputs=direct_parent.input,
        outputs=[target_layer.output, direct_parent.output],
        name="gradcam_feature_extractor",
    )

    # Recursive routing: at each nesting level apply pre/post layers around the
    # next level's call, threading conv_out all the way back to the surface.
    def apply_routing(level_idx, x):
        if level_idx == len(path):
            return feature_extractor(x)
        parent, child_idx = path[level_idx]
        for layer in parent.layers[:child_idx]:
            if not isinstance(layer, tf.keras.layers.InputLayer):
                x = layer(x)
        conv_out, y = apply_routing(level_idx + 1, x)
        for layer in parent.layers[child_idx + 1:]:
            y = layer(y)
        return conv_out, y

    def gradcam_forward(x):
        return apply_routing(0, x)

    return gradcam_forward, target_layer.name


def compute_gradcam(gradcam_forward, img_normalized, class_idx):
    """
    Compute Grad-CAM heatmap for one image and one target class.

    Algorithm:
      1. Forward pass → record activations at the last conv layer.
      2. Backprop the class score to those activations.
      3. Global-average-pool the gradients → per-channel importance weights.
      4. Weighted sum of activation maps + ReLU → raw heatmap.
      5. Normalise to [0, 1].
    """
    img_tensor = tf.cast(np.expand_dims(img_normalized, 0), tf.float32)

    with tf.GradientTape() as tape:
        # Watching img_tensor ensures the tape tracks all downstream tensors,
        # including the intermediate conv_outputs, enabling gradient computation
        # w.r.t. that non-Variable tensor.
        tape.watch(img_tensor)
        conv_outputs, predictions = gradcam_forward(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError(
            f"Gradients are None for class {class_idx}. "
            "The model may have non-differentiable operations in this path."
        )

    # (H, W, C) importance weights via global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted linear combination of activation maps
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)

    # Keep only positive evidence for this class
    heatmap = tf.nn.relu(heatmap)

    # Normalise
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def make_overlay(img_array, heatmap, alpha=0.45, colormap="jet"):
    """
    Resize heatmap to match the image, apply a colormap, and blend with the
    original image. Returns (overlay_rgb_float, heatmap_resized_float).
    """
    h, w = img_array.shape[:2]

    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w, h), Image.Resampling.LANCZOS
        )
    ) / 255.0

    cmap = plt.get_cmap(colormap)
    heatmap_rgb = cmap(heatmap_resized)[:, :, :3]

    img_float = img_array.astype(np.float32) / 255.0
    overlay = (1 - alpha) * img_float + alpha * heatmap_rgb
    return np.clip(overlay, 0, 1), heatmap_resized


def explain_and_plot(img_path, model, class_names, gradcam_forward):
    """Run Grad-CAM on one image and display per-class heatmap overlays."""
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image)  # uint8 (224, 224, 3)

    # Prediction
    norm = preprocess(img_array)
    probs = model.predict(np.expand_dims(norm, 0), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]

    print("  Confidence scores:")
    for i, name in enumerate(class_names):
        marker = " <-- predicted" if i == pred_idx else ""
        print(f"    {name}: {probs[i]:.2%}{marker}")

    n_classes = len(class_names)
    # Extra column for colorbar reference
    fig, axes = plt.subplots(
        1, n_classes + 2, figsize=(4 * (n_classes + 1) + 1.5, 4.5)
    )
    fig.suptitle(
        f"Grad-CAM Explanation  —  {os.path.basename(img_path)}",
        fontsize=13,
        fontweight="bold",
    )

    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title(
        f"Original\nPredicted: {pred_class}\n({probs[pred_idx]:.2%})",
        fontsize=10,
    )
    axes[0].axis("off")

    # Per-class Grad-CAM overlays
    print("  Computing Grad-CAM heatmaps…")
    heatmap_img = None
    for i, name in enumerate(class_names):
        heatmap = compute_gradcam(gradcam_forward, norm, i)
        overlay, heatmap_resized = make_overlay(img_array, heatmap)

        axes[i + 1].imshow(overlay)
        colour = "green" if i == pred_idx else "black"
        axes[i + 1].set_title(
            f"Class: {name}\n({probs[i]:.2%})",
            fontsize=10,
            color=colour,
        )
        axes[i + 1].axis("off")
        heatmap_img = heatmap_resized  # keep last for colorbar reference

    # Colorbar axis
    ax_cb = axes[-1]
    ax_cb.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_cb, fraction=0.8, pad=0.0)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["low", "mid", "high"])
    cbar.set_label("Relevance", fontsize=9)
    ax_cb.set_visible(True)
    ax_cb.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- select model ---
    username = input("Enter the username (folder name inside models/): ").strip()
    model_dir = find_model_dir(username)

    if model_dir is None:
        print(
            f"Could not find keras_model.h5 for '{username}'.\n"
            f"Expected path: models/{username}/converted_keras/keras_model.h5"
        )
        sys.exit(1)

    print(f"Loading model from: {model_dir}")
    model = load_model(os.path.join(model_dir, "keras_model.h5"), compile=False)
    class_names = load_class_names(os.path.join(model_dir, "labels.txt"))
    print(f"Classes: {class_names}\n")

    # Build Grad-CAM callable once, reuse across all images
    print("Building Grad-CAM model…")
    gradcam_forward, layer_name = build_gradcam_forward(model)
    print(f"  Ready.\n")

    # --- collect images ---
    image_paths = collect_images("images")

    if not image_paths:
        print("No images found in the 'images/' folder. Add some images and re-run.")
        sys.exit(0)

    print(f"Found {len(image_paths)} image(s) in 'images/':\n")
    for idx, p in enumerate(image_paths):
        print(f"  [{idx}] {os.path.basename(p)}")

    choice = input(
        "\nEnter image index to explain, or press Enter to explain all: "
    ).strip()

    if choice == "":
        selected = image_paths
    elif choice.isdigit() and 0 <= int(choice) < len(image_paths):
        selected = [image_paths[int(choice)]]
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    for path in selected:
        print(f"\nExplaining: {path}")
        explain_and_plot(path, model, class_names, gradcam_forward)
