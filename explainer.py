# Nico Hambauer <nico.hambauer@ur.de> 2026
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries

np.set_printoptions(suppress=True)


def find_model_dir(username):
    """Resolve the directory containing keras_model.h5 for a given username."""
    base = f"models/{username}"
    # Some extractions land in a converted_keras subdirectory
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
    # Sort by numeric index to be safe
    entries = sorted((int(l.split()[0]), l.split(None, 1)[1].strip()) for l in lines)
    return [name for _, name in entries]


def preprocess(img_array):
    """Resize to 224x224 and normalise to [-1, 1] (Teachable Machine convention)."""
    image = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32)
    return (arr / 127.5) - 1


def make_batch_predict(model):
    """Return a predict function that LIME can call with batches of uint8 images."""
    def batch_predict(images):
        batch = np.array([preprocess(img) for img in images])
        return model.predict(batch, verbose=0)
    return batch_predict


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


def explain_and_plot(img_path, model, class_names, num_samples=10000):
    """Run LIME on one image and display the explanation."""
    # --- load & display-ready array ---
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image)  # uint8, shape (224, 224, 3)

    # --- model prediction ---
    norm = preprocess(img_array)
    probs = model.predict(np.expand_dims(norm, 0), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    print(f"  Confidence scores:")
    for i, name in enumerate(class_names):
        marker = " <-- predicted" if i == pred_idx else ""
        print(f"    {name}: {probs[i]:.2%}{marker}")

    # --- LIME explanation ---
    print(f"  Running LIME with {num_samples} samples (may take a moment)…")
    explainer = lime_image.LimeImageExplainer(verbose=False)
    explanation = explainer.explain_instance(
        img_array,
        make_batch_predict(model),
        top_labels=len(class_names),
        hide_color=0,
        num_samples=num_samples,
    )

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(4 * (n_classes + 1), 4.5))
    fig.suptitle(
        f"LIME Explanation  —  {os.path.basename(img_path)}",
        fontsize=13,
        fontweight="bold",
    )

    # original image
    axes[0].imshow(img_array)
    axes[0].set_title(
        f"Original\nPredicted: {pred_class}\n({probs[pred_idx]:.2%})",
        fontsize=10,
    )
    axes[0].axis("off")

    # one subplot per class
    for i, name in enumerate(class_names):
        temp, mask = explanation.get_image_and_mask(
            i,
            positive_only=False,
            num_features=10,
            hide_rest=False,
        )
        axes[i + 1].imshow(mark_boundaries(temp / 255.0, mask))

        # colour-code: highlight predicted class title
        colour = "green" if i == pred_idx else "black"
        axes[i + 1].set_title(
            f"Class: {name}\n({probs[i]:.2%})",
            fontsize=10,
            color=colour,
        )
        axes[i + 1].axis("off")

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

    # --- collect images ---
    image_paths = collect_images("images")

    if not image_paths:
        print("No images found in the 'images/' folder. Add some images and re-run.")
        sys.exit(0)

    print(f"Found {len(image_paths)} image(s) in 'images/':\n")
    for idx, p in enumerate(image_paths):
        print(f"  [{idx}] {os.path.basename(p)}")

    # let the user pick a specific image or run all
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
        explain_and_plot(path, model, class_names, num_samples=10000)
