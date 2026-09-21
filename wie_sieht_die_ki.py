# Andreas Schauer <andreas.schauer@ur.de> 2026
"""
"Wie sieht die KI die Welt?" – Live-Bildschirm für den Tag der KI.

Zeigt das Webcam-Bild neben den Feature-Maps (Aktivierungen) eines
vortrainierten neuronalen Netzes (MobileNetV2, ImageNet-Gewichte).
Kein Training nötig – einfach starten und laufen lassen.

Voraussetzung: Beim allerersten Start lädt Keras die ImageNet-Gewichte
(~14 MB) herunter und legt sie in ~/.keras/models ab. Das braucht einmalig
Internet – danach läuft alles offline. Am besten also vor der Veranstaltung
einmal starten, damit der Download schon passiert ist.
"""
import threading

import cv2
import numpy as np
import tensorflow as tf
import ttkbootstrap as tb
from PIL import Image, ImageTk, ImageOps
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox

CAMERA_PREVIEW_SIZE = (360, 270)
GRID_SIZE = 4  # 4x4 = 16 Feature-Maps
TILE_SIZE = 100  # Pixelgröße je Kachel im Ergebnis-Grid
UPDATE_MS = 120  # Zeit zwischen zwei Frames (~8 fps)
MAX_CAMERA_INDEX = 4  # wie viele Kamera-Indizes beim Scannen probiert werden


def list_available_cameras(max_index=MAX_CAMERA_INDEX):
    """Probe camera indices and return the ones that actually deliver a frame."""
    available = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(index)
        cap.release()
    return available

# Kindgerechte Namen für ausgewählte MobileNetV2-Schichten, von früh (Kanten)
# bis tief (abstrakt). Alle vier werden in einem Forward-Pass gemeinsam
# berechnet, sodass ein Wechsel zwischen ihnen keine Verzögerung verursacht.
# Die Beschreibung erklärt Besucher:innen, wonach die Filter dieser Schicht ungefähr suchen.
LAYER_PRESETS = [
    (
        "Kanten",
        "Conv1_relu",
        "Jede Kachel ist ein eigener Filter. Er reagiert stark (helle Farbe), wo im Bild eine "
        "Kante oder ein Farbübergang in eine bestimmte Richtung verläuft.",
    ),
    (
        "Texturen",
        "block_2_expand_relu",
        "Die Filter kombinieren Kanten zu kleinen Texturen, z.B. Streifen, Ecken oder Farbflächen.",
    ),
    (
        "Muster",
        "block_6_expand_relu",
        "Die Filter erkennen bereits größere, sich wiederholende Muster und grobe Formen.",
    ),
    (
        "Abstrakt",
        "block_13_expand_relu",
        "Ganz tief im Netz: Die Filter reagieren auf abstrakte Kombinationen, die für uns Menschen "
        "kaum noch als konkretes Motiv erkennbar sind.",
    ),
]


def build_feature_extractor():
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    outputs = [base_model.get_layer(name).output for _, name, _ in LAYER_PRESETS]
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)


def activations_to_grid(activations, grid_size=GRID_SIZE, tile_size=TILE_SIZE):
    """Turn a (H, W, C) activation volume into one colourised, labelled grid image (RGB)."""
    height, width, channels = activations.shape
    num_tiles = grid_size * grid_size
    channel_indices = np.linspace(0, channels - 1, num_tiles, dtype=int)

    tiles = []
    for idx in channel_indices:
        channel = activations[:, :, idx]
        channel = channel - channel.min()
        max_val = channel.max()
        if max_val > 1e-6:
            channel = channel / max_val
        tile = (channel * 255).astype(np.uint8)
        tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_CUBIC)
        tile = cv2.applyColorMap(tile, cv2.COLORMAP_TURBO)

        # Label each tile with its filter/channel number, so it's clear that every
        # tile shows a *different* filter of the same layer, not the same thing four times.
        label = f"Filter {idx}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(tile, (0, 0), (text_w + 8, text_h + 8), (0, 0, 0), -1)
        cv2.putText(
            tile, label, (4, text_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )
        tiles.append(tile)

    rows = [
        np.hstack(tiles[row * grid_size:(row + 1) * grid_size])
        for row in range(grid_size)
    ]
    grid_bgr = np.vstack(rows)
    return cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)


class FeatureMapApp(tb.Window):
    def __init__(self):
        super().__init__(
            title="Wie sieht die KI die Welt?",
            themename="pulse",
            size=(1000, 680),
            minsize=(860, 600),
        )

        self.camera = None
        self.camera_index = 0
        self.running = False
        self.current_layer_idx = 0
        self.extractor = None

        outer = tb.Frame(self, padding=20)
        outer.pack(fill=BOTH, expand=YES)

        tb.Label(
            outer, text="🧠 Wie sieht die KI die Welt?", font=("Helvetica", 22, "bold"), bootstyle=PRIMARY
        ).pack(anchor=W)
        tb.Label(
            outer,
            text=(
                "Die bunten Muster zeigen, wonach die ersten Schichten eines neuronalen Netzes in "
                "deinem Kamerabild suchen – z.B. Kanten und Farbübergänge. Je tiefer die Schicht, "
                "desto abstrakter wird das, was die KI 'sieht'."
            ),
            font=("Helvetica", 11),
            bootstyle=SECONDARY,
            wraplength=920,
            justify=LEFT,
        ).pack(anchor=W, pady=(4, 16))

        # Controls
        controls_row = tb.Frame(outer)
        controls_row.pack(fill=X, pady=(0, 16))

        tb.Label(controls_row, text="Netzwerk-Tiefe:", font=("Helvetica", 12, "bold")).pack(side=LEFT)
        self.layer_var = tb.StringVar(value=LAYER_PRESETS[0][0])
        layer_combo = tb.Combobox(
            controls_row,
            textvariable=self.layer_var,
            values=[label for label, _, _ in LAYER_PRESETS],
            state="readonly",
            font=("Helvetica", 12),
            bootstyle=PRIMARY,
            width=14,
        )
        layer_combo.pack(side=LEFT, padx=(8, 16))
        layer_combo.bind("<<ComboboxSelected>>", self.on_layer_selected)

        tb.Label(controls_row, text="Kamera:", font=("Helvetica", 12, "bold")).pack(side=LEFT)
        self.camera_var = tb.StringVar(value="Kamera 0")
        self.camera_combo = tb.Combobox(
            controls_row,
            textvariable=self.camera_var,
            state="readonly",
            font=("Helvetica", 12),
            bootstyle=PRIMARY,
            width=10,
        )
        self.camera_combo.pack(side=LEFT, padx=(8, 4))
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        tb.Button(
            controls_row, text="🔄", width=3, bootstyle=(SECONDARY, OUTLINE), command=self.scan_cameras
        ).pack(side=LEFT, padx=(0, 16))

        self.fullscreen_button = tb.Button(
            controls_row, text="🖥️  Vollbild", bootstyle=(SECONDARY, OUTLINE), command=self.toggle_fullscreen
        )
        self.fullscreen_button.pack(side=RIGHT)

        # Main content: camera preview (left) + feature map grid (right)
        content_row = tb.Frame(outer)
        content_row.pack(fill=BOTH, expand=YES)

        camera_panel = tb.Frame(content_row)
        camera_panel.pack(side=LEFT, fill=Y, padx=(0, 16))
        tb.Label(camera_panel, text="Kamera", font=("Helvetica", 11, "bold"), bootstyle=SECONDARY).pack(anchor=W)
        self.camera_label = tb.Label(camera_panel, background="black")
        self.camera_label.pack(pady=(4, 0))

        grid_panel = tb.Frame(content_row)
        grid_panel.pack(side=LEFT, fill=BOTH, expand=YES)
        tb.Label(grid_panel, text="Feature-Maps", font=("Helvetica", 11, "bold"), bootstyle=SECONDARY).pack(anchor=W)
        self.grid_label = tb.Label(grid_panel, background="black")
        self.grid_label.pack(fill=BOTH, expand=YES, pady=(4, 0))

        self.layer_description_var = tb.StringVar()
        tb.Label(
            grid_panel,
            textvariable=self.layer_description_var,
            font=("Helvetica", 10),
            bootstyle=SECONDARY,
            wraplength=560,
            justify=LEFT,
        ).pack(anchor=W, pady=(6, 0))
        tb.Label(
            grid_panel,
            text=(
                "Jede der 16 Kacheln zeigt einen anderen Filter (Kanal) dieser Schicht: "
                "dunkel = schwache Reaktion an dieser Bildstelle, hell/rot = starke Reaktion."
            ),
            font=("Helvetica", 9),
            bootstyle=SECONDARY,
            wraplength=560,
            justify=LEFT,
        ).pack(anchor=W, pady=(2, 0))

        self.status_var = tb.StringVar(value="Lade neuronales Netz…")
        tb.Label(outer, textvariable=self.status_var, font=("Helvetica", 10), bootstyle=SECONDARY).pack(
            anchor=W, pady=(12, 0)
        )

        self.bind("<Escape>", lambda event: self.set_fullscreen(False))
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Load the model after the window is visible, so the UI isn't blank while it loads.
        self.after(50, self.initialize)

    def initialize(self):
        self.update_idletasks()
        self.extractor = build_feature_extractor()
        self.open_camera(self.camera_index)
        self.scan_cameras()
        self.update_status()

    def open_camera(self, index=0):
        self.camera = cv2.VideoCapture(index)
        if not self.camera.isOpened():
            self.status_var.set(f"Keine Webcam an Index {index} gefunden. Bitte Zugriff/Anschluss prüfen.")
            self.camera = None
            return
        self.camera_index = index
        self.camera_var.set(f"Kamera {index}")
        self.running = True
        self.update_loop()

    def scan_cameras(self):
        threading.Thread(target=self._scan_cameras_worker, daemon=True).start()

    def _scan_cameras_worker(self):
        indices = list_available_cameras()
        self.after(0, self._on_cameras_found, indices)

    def _on_cameras_found(self, indices):
        if not indices:
            indices = [self.camera_index]
        values = [f"Kamera {i}" for i in indices]
        self.camera_combo["values"] = values
        current_label = f"Kamera {self.camera_index}"
        if current_label not in values:
            self.camera_var.set(values[0])

    def on_camera_selected(self, event=None):
        label = self.camera_var.get()
        index = int(label.replace("Kamera ", ""))
        if index != self.camera_index:
            self.switch_camera(index)

    def switch_camera(self, index):
        new_camera = cv2.VideoCapture(index)
        if not new_camera.isOpened():
            new_camera.release()
            Messagebox.show_error(f"Kamera {index} konnte nicht geöffnet werden.", "Fehler")
            self.camera_var.set(f"Kamera {self.camera_index}")
            return
        if self.camera is not None:
            self.camera.release()
        self.camera = new_camera
        self.camera_index = index
        if not self.running:
            self.running = True
            self.update_loop()

    def on_layer_selected(self, event=None):
        label = self.layer_var.get()
        self.current_layer_idx = [name for name, _, _ in LAYER_PRESETS].index(label)
        self.update_status()

    def update_status(self):
        label, layer_name, description = LAYER_PRESETS[self.current_layer_idx]
        self.layer_description_var.set(description)
        if self.extractor is not None:
            shape = self.extractor.outputs[self.current_layer_idx].shape
            self.status_var.set(
                f"Schicht: {layer_name}  ·  Auflösung: {shape[1]}×{shape[2]}  ·  "
                f"{shape[3]} Kanäle (zeige {GRID_SIZE * GRID_SIZE})"
            )

    def toggle_fullscreen(self):
        is_fullscreen = bool(self.attributes("-fullscreen"))
        self.set_fullscreen(not is_fullscreen)

    def set_fullscreen(self, value):
        self.attributes("-fullscreen", value)

    def update_loop(self):
        if not self.running or self.camera is None:
            return
        ok, frame = self.camera.read()
        if ok:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_camera(rgb)
            self.process_frame(rgb)
        self.after(UPDATE_MS, self.update_loop)

    def process_frame(self, rgb):
        resized = cv2.resize(rgb, (224, 224))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(resized.astype(np.float32))
        x = np.expand_dims(x, 0)

        outputs = self.extractor(x, training=False)
        activations = outputs[self.current_layer_idx].numpy()[0]

        grid_rgb = activations_to_grid(activations)
        self.show_grid(grid_rgb)

    def show_camera(self, rgb_array):
        image = Image.fromarray(rgb_array)
        image = ImageOps.contain(image, CAMERA_PREVIEW_SIZE)
        photo = ImageTk.PhotoImage(image)
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo

    def show_grid(self, rgb_array):
        image = Image.fromarray(rgb_array)
        photo = ImageTk.PhotoImage(image)
        self.grid_label.configure(image=photo)
        self.grid_label.image = photo

    def on_close(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
        self.destroy()


if __name__ == "__main__":
    app = FeatureMapApp()
    app.mainloop()
