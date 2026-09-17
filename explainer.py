# Nico Hambauer <nico.hambauer@ur.de> 2026
import os
import threading
import tkinter as tk

import cv2
import numpy as np
import ttkbootstrap as tb
from lime import lime_image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageOps, ImageTk
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import load_model
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox

np.set_printoptions(suppress=True)

MODELS_DIR = "models"
NUM_LIME_SAMPLES = 600
PREVIEW_SIZE = (440, 330)


def find_model_dir(username):
    """Resolve the directory containing keras_model.h5 for a given username."""
    base = f"{MODELS_DIR}/{username}"
    # Some extractions land in a converted_keras subdirectory
    candidate = os.path.join(base, "converted_keras")
    if os.path.isfile(os.path.join(candidate, "keras_model.h5")):
        return candidate
    if os.path.isfile(os.path.join(base, "keras_model.h5")):
        return base
    return None


def list_available_models():
    """Return sorted usernames under models/ that contain a usable keras model."""
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(u for u in os.listdir(MODELS_DIR) if find_model_dir(u) is not None)


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


def build_explanation_figure(image, model, class_names, num_samples=NUM_LIME_SAMPLES):
    """Run LIME on a PIL image and return (figure, predicted_class, probabilities)."""
    image = image.convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image)  # uint8, shape (224, 224, 3)

    norm = preprocess(img_array)
    probs = model.predict(np.expand_dims(norm, 0), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]

    explainer = lime_image.LimeImageExplainer(verbose=False)
    explanation = explainer.explain_instance(
        img_array,
        make_batch_predict(model),
        top_labels=len(class_names),
        hide_color=0,
        num_samples=num_samples,
    )

    n_classes = len(class_names)
    fig = Figure(figsize=(3.2 * (n_classes + 1), 3.6), dpi=100)
    fig.suptitle(
        f"Warum wurde das Bild als '{pred_class}' erkannt?",
        fontsize=13,
        fontweight="bold",
    )
    axes = fig.subplots(1, n_classes + 1)

    # original image
    axes[0].imshow(img_array)
    axes[0].set_title(f"Original\nVorhersage: {pred_class}\n({probs[pred_idx]:.2%})", fontsize=9)
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
        colour = "green" if i == pred_idx else "black"
        axes[i + 1].set_title(f"Klasse: {name}\n({probs[i]:.2%})", fontsize=9, color=colour)
        axes[i + 1].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig, pred_class, probs


class ExplainerApp(tb.Window):
    def __init__(self):
        super().__init__(
            title="Bild-Erklärer – Webcam",
            themename="pulse",
            size=(800, 700),
            minsize=(680, 600),
        )

        self.model = None
        self.class_names = None
        self.captured_image = None  # PIL Image once a photo is taken
        self.last_frame = None  # most recent live camera frame (RGB)
        self.camera = None
        self.live = False
        self.canvas_widget = None

        outer = tb.Frame(self, padding=20)
        outer.pack(fill=BOTH, expand=YES)

        tb.Label(
            outer, text="🔍 Warum sieht die KI das so?", font=("Helvetica", 20, "bold"), bootstyle=PRIMARY
        ).pack(anchor=W)
        tb.Label(
            outer,
            text="Wähle dein Modell, mach ein Webcam-Foto und lass dir die Entscheidung erklären.",
            font=("Helvetica", 11),
            bootstyle=SECONDARY,
        ).pack(anchor=W, pady=(4, 16))

        # Model selection
        model_row = tb.Frame(outer)
        model_row.pack(fill=X, pady=(0, 12))
        tb.Label(model_row, text="Modell:", font=("Helvetica", 12, "bold")).pack(side=LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = tb.Combobox(
            model_row, textvariable=self.model_var, state="readonly", font=("Helvetica", 12), bootstyle=PRIMARY
        )
        self.model_combo.pack(side=LEFT, fill=X, expand=YES, padx=(8, 8))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)
        tb.Button(model_row, text="🔄", width=3, bootstyle=(SECONDARY, OUTLINE), command=self.refresh_models).pack(
            side=LEFT
        )

        # Camera / result area
        self.view_frame = tb.Frame(outer)
        self.view_frame.pack(fill=BOTH, expand=YES, pady=(0, 12))

        self.camera_label = tk.Label(self.view_frame, bg="black")
        self.camera_label.pack(fill=BOTH, expand=YES)

        # Buttons
        button_row = tb.Frame(outer)
        button_row.pack(fill=X, pady=(0, 12))

        self.capture_button = tb.Button(
            button_row, text="📸 Foto aufnehmen", bootstyle=PRIMARY, command=self.capture_photo, state="disabled"
        )
        self.capture_button.pack(side=LEFT, padx=(0, 8), ipady=4)

        self.retake_button = tb.Button(
            button_row, text="🔁 Neues Foto", bootstyle=(SECONDARY, OUTLINE), command=self.retake_photo, state="disabled"
        )
        self.retake_button.pack(side=LEFT, padx=(0, 8), ipady=4)

        self.explain_button = tb.Button(
            button_row,
            text="🔍 Erklärung anzeigen",
            bootstyle=SUCCESS,
            command=self.start_explanation,
            state="disabled",
        )
        self.explain_button.pack(side=LEFT, fill=X, expand=YES, ipady=4)

        self.progress = tb.Progressbar(outer, mode=INDETERMINATE, bootstyle=(SUCCESS, STRIPED))

        self.status_var = tk.StringVar(value="Bitte zuerst ein Modell auswählen.")
        self.status_label = tb.Label(outer, textvariable=self.status_var, font=("Helvetica", 11), bootstyle=SECONDARY)
        self.status_label.pack(anchor=W)

        self.refresh_models()
        self.open_camera()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- model handling ---
    def refresh_models(self):
        models = list_available_models()
        self.model_combo["values"] = models
        if models and self.model_var.get() not in models:
            self.model_var.set("")
        if not models:
            self.status_var.set("Keine Modelle in 'models/' gefunden.")

    def on_model_selected(self, event=None):
        username = self.model_var.get()
        model_dir = find_model_dir(username)
        if model_dir is None:
            Messagebox.show_error(f"Kein Modell für '{username}' gefunden.", "Fehler")
            return
        try:
            self.status_var.set(f"Lade Modell '{username}'...")
            self.update_idletasks()
            self.model = load_model(os.path.join(model_dir, "keras_model.h5"), compile=False)
            self.class_names = load_class_names(os.path.join(model_dir, "labels.txt"))
            self.status_var.set(f"Modell '{username}' geladen. Klassen: {', '.join(self.class_names)}")
            self._update_capture_availability()
        except Exception as exc:
            Messagebox.show_error(str(exc), "Fehler beim Laden")

    # --- camera handling ---
    def open_camera(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.status_var.set("Keine Webcam gefunden. Bitte Zugriff/Anschluss prüfen.")
            self.camera = None
            return
        self.live = True
        self._update_capture_availability()
        self.update_camera_feed()

    def _update_capture_availability(self):
        if self.live and self.model is not None:
            self.capture_button.configure(state="normal")

    def update_camera_feed(self):
        if not self.live or self.camera is None:
            return
        ok, frame = self.camera.read()
        if ok:
            frame = cv2.flip(frame, 1)
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_image(self.last_frame)
        self.after(30, self.update_camera_feed)

    def show_image(self, rgb_array):
        image = Image.fromarray(rgb_array)
        image = ImageOps.contain(image, PREVIEW_SIZE)
        photo = ImageTk.PhotoImage(image)
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo

    def capture_photo(self):
        if self.last_frame is None:
            return
        self.live = False
        self.captured_image = Image.fromarray(self.last_frame)
        self.show_image(self.last_frame)
        self.capture_button.configure(state="disabled")
        self.retake_button.configure(state="normal")
        self.explain_button.configure(state="normal")
        self.status_var.set("Foto aufgenommen. Jetzt die Erklärung anzeigen lassen.")

    def retake_photo(self):
        self.clear_result()
        self.captured_image = None
        self.retake_button.configure(state="disabled")
        self.explain_button.configure(state="disabled")
        self.status_var.set("Live-Kamerabild.")
        self.live = True
        self._update_capture_availability()
        self.update_camera_feed()

    # --- explanation ---
    def start_explanation(self):
        if self.model is None or self.captured_image is None:
            return

        self.capture_button.configure(state="disabled")
        self.retake_button.configure(state="disabled")
        self.explain_button.configure(state="disabled")
        self.model_combo.configure(state="disabled")

        self.progress.pack(fill=X, pady=(0, 8), before=self.status_label)
        self.progress.start(12)
        self.status_var.set("Berechne Erklärung (kann einen Moment dauern)...")
        self.config(cursor="wait")

        thread = threading.Thread(target=self._run_explanation, daemon=True)
        thread.start()

    def _run_explanation(self):
        try:
            fig, pred_class, _ = build_explanation_figure(self.captured_image, self.model, self.class_names)
            self.after(0, self._on_explanation_done, fig, pred_class)
        except Exception as exc:
            self.after(0, self._on_explanation_error, str(exc))

    def _on_explanation_done(self, fig, pred_class):
        self.clear_result()

        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.view_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill=BOTH, expand=YES)
        self.camera_label.pack_forget()

        self.status_var.set(f"Vorhersage: {pred_class}")
        self.retake_button.configure(state="normal")
        self.model_combo.configure(state="readonly")
        self._finish_progress()

    def _on_explanation_error(self, message):
        Messagebox.show_error(message, "Fehler")
        self.retake_button.configure(state="normal")
        self.explain_button.configure(state="normal")
        self.model_combo.configure(state="readonly")
        self._finish_progress()

    def _finish_progress(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.config(cursor="")

    def clear_result(self):
        if self.canvas_widget is not None:
            self.canvas_widget.get_tk_widget().destroy()
            self.canvas_widget = None
        self.camera_label.pack(fill=BOTH, expand=YES)

    def on_close(self):
        self.live = False
        if self.camera is not None:
            self.camera.release()
        self.destroy()


if __name__ == "__main__":
    app = ExplainerApp()
    app.mainloop()
