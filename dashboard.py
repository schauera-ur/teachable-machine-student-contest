# Andreas Schauer <andreas.schauer@ur.de> 2026
"""
Dashboard für den Tag der KI.

Startet main.py, explainer.py und wie_sieht_die_ki.py per Klick als eigene
Prozesse, sodass am Stand niemand ein Terminal bedienen muss.
"""
import os
import subprocess
import sys

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

APPS = [
    {
        "key": "main",
        "icon": "🏆",
        "title": "Teachable-Machine-Contest",
        "description": "Modell-ZIP hochladen, Genauigkeit berechnen und in der Bestenliste speichern.",
        "script": "main.py",
    },
    {
        "key": "explainer",
        "icon": "🔍",
        "title": "Bild-Erklärer",
        "description": "Webcam-Foto aufnehmen und erklären lassen, warum die KI so entschieden hat.",
        "script": "explainer.py",
    },
    {
        "key": "feature_maps",
        "icon": "🧠",
        "title": "Wie sieht die KI die Welt?",
        "description": "Live-Feature-Maps eines neuronalen Netzes auf dem großen Bildschirm.",
        "script": "wie_sieht_die_ki.py",
    },
]

POLL_MS = 1000


class DashboardApp(tb.Window):
    def __init__(self):
        super().__init__(
            title="Tag der KI – Dashboard",
            themename="pulse",
            size=(640, 620),
            minsize=(560, 520),
        )

        self.processes = {}  # key -> subprocess.Popen
        self.status_vars = {}  # key -> tb.StringVar
        self.buttons = {}  # key -> tb.Button
        self.user_stopped = set()  # keys stopped via the "Beenden" button, not a crash

        outer = tb.Frame(self, padding=24)
        outer.pack(fill=BOTH, expand=YES)

        tb.Label(
            outer, text="🚀 Tag der KI – Dashboard", font=("Helvetica", 22, "bold"), bootstyle=PRIMARY
        ).pack(anchor=W)
        tb.Label(
            outer,
            text="Wähle eine Anwendung zum Starten. Jede läuft in einem eigenen Fenster.",
            font=("Helvetica", 11),
            bootstyle=SECONDARY,
        ).pack(anchor=W, pady=(4, 20))

        for app in APPS:
            self.build_card(outer, app)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(POLL_MS, self.poll_processes)

    def build_card(self, parent, app):
        key = app["key"]

        card = tb.Frame(parent, padding=16, bootstyle=LIGHT)
        card.pack(fill=X, pady=(0, 16))

        tb.Label(
            card, text=f"{app['icon']}  {app['title']}", font=("Helvetica", 14, "bold")
        ).pack(anchor=W)
        tb.Label(
            card,
            text=app["description"],
            font=("Helvetica", 10),
            bootstyle=SECONDARY,
            wraplength=520,
            justify=LEFT,
        ).pack(anchor=W, pady=(2, 10))

        footer = tb.Frame(card)
        footer.pack(fill=X)

        status_var = tb.StringVar(value="Nicht gestartet")
        tb.Label(footer, textvariable=status_var, font=("Helvetica", 10), bootstyle=SECONDARY).pack(
            side=LEFT
        )
        self.status_vars[key] = status_var

        button = tb.Button(
            footer, text="▶  Starten", bootstyle=SUCCESS, command=lambda: self.toggle_app(key)
        )
        button.pack(side=RIGHT)
        self.buttons[key] = button

    def toggle_app(self, key):
        proc = self.processes.get(key)
        if proc is not None and proc.poll() is None:
            self.stop_app(key)
        else:
            self.start_app(key)

    def start_app(self, key):
        app = next(a for a in APPS if a["key"] == key)
        script_path = os.path.join(BASE_DIR, app["script"])
        try:
            proc = subprocess.Popen([sys.executable, script_path], cwd=BASE_DIR)
        except Exception as exc:
            Messagebox.show_error(str(exc), "Fehler beim Starten")
            return
        self.processes[key] = proc
        self.user_stopped.discard(key)
        self.update_card(key)

    def stop_app(self, key):
        proc = self.processes.get(key)
        if proc is not None and proc.poll() is None:
            self.user_stopped.add(key)
            proc.terminate()
        self.update_card(key)

    def update_card(self, key):
        proc = self.processes.get(key)
        status_var = self.status_vars[key]
        button = self.buttons[key]

        if proc is None:
            status_var.set("Nicht gestartet")
            button.configure(text="▶  Starten", bootstyle=SUCCESS)
            return

        return_code = proc.poll()
        if return_code is None:
            status_var.set(f"● Läuft (PID {proc.pid})")
            button.configure(text="⏹  Beenden", bootstyle=(DANGER, OUTLINE))
        elif return_code == 0 or key in self.user_stopped:
            status_var.set("Beendet.")
            button.configure(text="▶  Starten", bootstyle=SUCCESS)
            self.user_stopped.discard(key)
        else:
            status_var.set(f"Beendet mit Fehlercode {return_code}.")
            button.configure(text="▶  Starten", bootstyle=SUCCESS)

    def poll_processes(self):
        for key in self.processes:
            self.update_card(key)
        self.after(POLL_MS, self.poll_processes)

    def on_close(self):
        # Gestartete Anwendungen laufen als eigenständige Prozesse weiter,
        # damit ein Schließen des Dashboards nicht versehentlich eine laufende
        # Station am Stand beendet.
        self.destroy()


if __name__ == "__main__":
    app = DashboardApp()
    app.mainloop()
