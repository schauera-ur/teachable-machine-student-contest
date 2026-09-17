# Andreas Schauer <andreas.schauer@ur.de> 2026
import os
import shutil
import threading
import tkinter as tk
import zipfile
from tkinter import filedialog

import numpy as np
import pandas as pd
import ttkbootstrap as tb
from PIL import Image, ImageOps  # Install pillow instead of PIL
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model  # TensorFlow is required for Keras to work
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from ttkbootstrap.widgets.scrolled import ScrolledText

np.set_printoptions(suppress=True)

TESTSET_DIRECTORY = "final_testset/"
LEADERBOARD_CSV = "_data/leaderboard.csv"


def predict_single(model_dir, img_path):
    import matplotlib.pyplot as plt

    model_path = f"{model_dir}/keras_model.h5"
    labels_path = f"{model_dir}/labels.txt"

    model = load_model(model_path, compile=False)
    class_names = open(labels_path, "r").readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(img_path).convert("RGB")

    # resizing to 224x224 and crop from center
    image_size = (224, 224)
    image = ImageOps.fit(image, image_size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    plt.imshow(image_array)
    plt.show()

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)


def extract_model_zip(zip_path, student_dir):
    """Extract a Teachable Machine 'converted_keras.zip' into student_dir/converted_keras."""
    target_dir = os.path.join(student_dir, "converted_keras")
    os.makedirs(target_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # Remove __MACOSX folder if it exists
    macosx_dir = os.path.join(target_dir, "__MACOSX")
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir)

    # Fix nested converted_keras/converted_keras structure
    nested_dir = os.path.join(target_dir, "converted_keras")
    if os.path.isdir(nested_dir):
        for item in os.listdir(nested_dir):
            src = os.path.join(nested_dir, item)
            dst = os.path.join(target_dir, item)
            if os.path.exists(dst):
                os.remove(dst) if os.path.isfile(dst) else shutil.rmtree(dst)
            shutil.move(src, dst)
        os.rmdir(nested_dir)

    if not (os.path.exists(f"{target_dir}/keras_model.h5") and os.path.exists(f"{target_dir}/labels.txt")):
        raise RuntimeError(
            "Die ZIP-Datei enthält kein gültiges Teachable-Machine-Modell "
            "(keras_model.h5 / labels.txt fehlen)."
        )

    return target_dir


def evaluate_model_on_testset(model_dir, testset_directory, log=print):
    model_path = f"{model_dir}/keras_model.h5"
    labels_path = f"{model_dir}/labels.txt"

    log("Lade Modell...")
    model = load_model(model_path, compile=False)

    # Read the class names and create a mapping from class names to integer labels
    with open(labels_path, "r") as f:
        label_mappings = {line.split()[1]: int(line.split()[0]) for line in f.readlines()}

    image_paths = []
    true_labels = []

    # Collect all image paths and their corresponding true labels
    for category_folder in os.listdir(testset_directory):
        category_path = os.path.join(testset_directory, category_folder)
        if os.path.isdir(category_path):  # Check if it is a directory
            true_label = category_folder.split('-')[0]  # Extract the true label from the folder name
            true_label_int = label_mappings[true_label]  # Convert true label to int
            for image_name in os.listdir(category_path):
                if image_name.lower().endswith(".jpg"):  # Check if the file is an image
                    image_paths.append(os.path.join(category_path, image_name))
                    true_labels.append(true_label_int)

    log(f"Bereite {len(image_paths)} Testbilder vor...")
    data = np.ndarray(shape=(len(image_paths), 224, 224, 3), dtype=np.float32)

    # Process each image and load into the data array
    for i, image_path in enumerate(image_paths):
        # Preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[i] = normalized_image_array  # Load the image into the data array

    # Predict the classes for all images
    log("Werte Testset aus...")
    predictions = model.predict(data, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)

    # Print average confidence score per class
    class_names = {v: k for k, v in label_mappings.items()}
    log("Durchschnittliche Konfidenz pro Klasse:")
    for class_idx, class_name in sorted(class_names.items()):
        avg_conf = predictions[:, class_idx].mean()
        log(f"  {class_name}: {avg_conf:.2%}")
    overall_avg = predictions.max(axis=1).mean()
    log(f"  Gesamt (vorhergesagte Klasse): {overall_avg:.2%}")

    # Calculate the test accuracy
    return accuracy_score(true_labels, predicted_indices)


def record_students_score(pseudonym, score, csv_file_path):
    if not os.path.exists(csv_file_path):
        # Create a DataFrame with just the headers, and save it
        pd.DataFrame(columns=["pseudonym", "accuracy"]).to_csv(csv_file_path, index=False)

    df = pd.read_csv(csv_file_path)

    if pseudonym in df["pseudonym"].values:
        raise ValueError(f"Pseudonym '{pseudonym}' existiert bereits in der Bestenliste.")

    new_row_df = pd.DataFrame([{"pseudonym": pseudonym, "accuracy": f"{score:.4f}"}])
    df = pd.concat([df, new_row_df], ignore_index=True)

    df.to_csv(csv_file_path, index=False)


class ContestApp(tb.Window):
    def __init__(self):
        super().__init__(
            title="Teachable Machine Schüler-Contest",
            themename="pulse",
            size=(600, 560),
            minsize=(540, 500),
        )

        self.zip_path = None

        outer = tb.Frame(self, padding=24)
        outer.pack(fill=BOTH, expand=YES)

        tb.Label(
            outer, text="🏆 Teachable Machine Contest", font=("Helvetica", 22, "bold"), bootstyle=PRIMARY
        ).pack(anchor=W)
        tb.Label(
            outer,
            text="Lade deine trainierte Modell-ZIP hoch und starte die Auswertung.",
            font=("Helvetica", 11),
            bootstyle=SECONDARY,
        ).pack(anchor=W, pady=(4, 24))

        # Pseudonym
        tb.Label(outer, text="1. Dein Fantasiename für die Bestenliste", font=("Helvetica", 12, "bold")).pack(anchor=W)
        self.pseudonym_var = tk.StringVar()
        tb.Entry(outer, textvariable=self.pseudonym_var, font=("Helvetica", 12), bootstyle=PRIMARY).pack(
            fill=X, pady=(6, 20), ipady=4
        )

        # Zip upload
        tb.Label(outer, text="2. Deine Modell-Datei (converted_keras.zip)", font=("Helvetica", 12, "bold")).pack(anchor=W)
        upload_row = tb.Frame(outer)
        upload_row.pack(fill=X, pady=(6, 20))

        self.file_label_var = tk.StringVar(value="Keine Datei ausgewählt")
        tb.Label(upload_row, textvariable=self.file_label_var, bootstyle=SECONDARY).pack(
            side=LEFT, fill=X, expand=YES
        )
        tb.Button(
            upload_row, text="Datei auswählen…", bootstyle=(SECONDARY, OUTLINE), command=self.choose_zip
        ).pack(side=RIGHT)

        # Start button
        self.start_button = tb.Button(
            outer, text="▶  Auswertung starten", bootstyle=SUCCESS, command=self.start_evaluation
        )
        self.start_button.pack(fill=X, pady=(0, 16), ipady=6)

        self.progress = tb.Progressbar(outer, mode=INDETERMINATE, bootstyle=(SUCCESS, STRIPED))

        # Log output
        log_label = tb.Label(outer, text="Verlauf", font=("Helvetica", 12, "bold"))
        log_label.pack(anchor=W)

        self.log_text = ScrolledText(outer, height=8, autohide=True, font=("Menlo", 11))
        self.log_text.pack(fill=BOTH, expand=YES, pady=(6, 16))
        # ScrolledText doesn't delegate configure() to the inner Text widget, so we
        # keep a direct handle on it for toggling the disabled/normal state.
        self._log_inner = self.log_text._text
        self._log_inner.configure(state="disabled")

        self._log_label = log_label

        # Result
        self.result_var = tk.StringVar(value="")
        tb.Label(outer, textvariable=self.result_var, font=("Helvetica", 22, "bold"), bootstyle=SUCCESS).pack(
            anchor=W, pady=(4, 0)
        )

    def choose_zip(self):
        initial_dir = os.path.expanduser("~/Downloads")
        path = filedialog.askopenfilename(
            title="Wähle deine converted_keras.zip",
            initialdir=initial_dir if os.path.isdir(initial_dir) else os.getcwd(),
            filetypes=[("ZIP-Dateien", "*.zip")],
        )
        if path:
            self.zip_path = path
            self.file_label_var.set(os.path.basename(path))

    def log(self, message):
        self._log_inner.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self._log_inner.configure(state="disabled")

    def start_evaluation(self):
        pseudonym = self.pseudonym_var.get().strip()

        if not pseudonym:
            Messagebox.show_warning("Bitte gib zuerst einen Fantasienamen ein.", "Fehlende Angabe")
            return
        if not self.zip_path:
            Messagebox.show_warning("Bitte wähle zuerst deine converted_keras.zip aus.", "Fehlende Angabe")
            return

        student_dir = f"models/{pseudonym}"
        if os.path.exists(student_dir):
            Messagebox.show_error(
                f"Für '{pseudonym}' existiert bereits ein Eintrag. Bitte wähle einen anderen Fantasienamen.",
                "Name bereits vergeben",
            )
            return

        self._log_inner.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self._log_inner.configure(state="disabled")
        self.result_var.set("")

        self.start_button.configure(state="disabled")
        self.progress.pack(fill=X, pady=(0, 12), before=self._log_label)
        self.progress.start(12)
        self.config(cursor="wait")

        thread = threading.Thread(target=self._run_evaluation, args=(pseudonym, student_dir), daemon=True)
        thread.start()

    def _run_evaluation(self, pseudonym, student_dir):
        try:
            os.makedirs(student_dir)
            self.after(0, self.log, "Entpacke Modell-ZIP...")
            model_dir = extract_model_zip(self.zip_path, student_dir)

            accuracy = evaluate_model_on_testset(
                model_dir, TESTSET_DIRECTORY, log=lambda msg: self.after(0, self.log, msg)
            )

            record_students_score(pseudonym, accuracy, LEADERBOARD_CSV)
            self.after(0, self.log, "Ergebnis in der Bestenliste gespeichert.")
            self.after(0, self._on_success, accuracy)
        except Exception as exc:
            shutil.rmtree(student_dir, ignore_errors=True)
            self.after(0, self._on_error, str(exc))

    def _on_success(self, accuracy):
        self.result_var.set(f"🎉 Deine Genauigkeit: {accuracy:.2%}")
        self._finish()

    def _on_error(self, message):
        Messagebox.show_error(message, "Fehler")
        self._finish()

    def _finish(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.start_button.configure(state="normal")
        self.config(cursor="")


if __name__ == '__main__':
    app = ContestApp()
    app.mainloop()
