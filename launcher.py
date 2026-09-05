"""
ISL Project Launcher

Modes:
  1. ISL Detection   — live debug recognition (1-9, A-Z)
  2. ISL Practice    — quiz / learn by signing a shown target
  3. Keyboard Mapper — type recognized signs into any text field
  4. Game Gestures   — rule-based gesture game controls
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox
from tkinter.font import Font

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.h5")
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")


MODES = [
    {
        "key": "detect",
        "title": "ISL Detection",
        "subtitle": "Debug / check recognition",
        "detail": "Live webcam view of detected signs\n(digits 1–9 and letters A–Z).",
        "script": os.path.join(SRC_DIR, "isl_detection.py"),
        "accent": "#1B7F5A",
        "needs_model": True,
    },
    {
        "key": "practice",
        "title": "ISL Practice",
        "subtitle": "Learn with a test",
        "detail": "A random letter or number is shown.\nSign it correctly to score points.",
        "script": os.path.join(SRC_DIR, "isl_practice.py"),
        "accent": "#C45C26",
        "needs_model": True,
    },
    {
        "key": "typer",
        "title": "Keyboard Mapper",
        "subtitle": "Sign → type characters",
        "detail": "Stable signs are typed into the\nfocused text field (both hands).",
        "script": os.path.join(SRC_DIR, "isl_typer.py"),
        "accent": "#2B6CB0",
        "needs_model": True,
    },
    {
        "key": "game",
        "title": "Game Gestures",
        "subtitle": "Control games with hands",
        "detail": "Thumb WASD, palm cursor, fist click,\nopen palm jump. No ISL model needed.",
        "script": os.path.join(SRC_DIR, "game_control.py"),
        "accent": "#5B4B8A",
        "needs_model": False,
    },
]


def detect_python():
    """Prefer project .venv, else current interpreter."""
    candidates = [
        os.path.join(VENV_DIR, "Scripts", "python.exe"),
        os.path.join(VENV_DIR, "bin", "python"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return sys.executable


def launch_script(python_exe, script_path):
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found:\n{script_path}")
    cmd = [python_exe, script_path]
    if os.name == "nt":
        return subprocess.Popen(cmd, cwd=PROJECT_ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE)
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT, start_new_session=True)


class LauncherGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GestureGuru")
        self.geometry("920x580")
        self.minsize(820, 520)
        self.configure(bg="#F3EEE6")

        self.python_exe = detect_python()
        self._build()

    def _build(self):
        title_font = Font(family="Segoe UI Semibold", size=22)
        body_font = Font(family="Segoe UI", size=11)
        small_font = Font(family="Segoe UI", size=9)

        header = tk.Frame(self, bg="#F3EEE6")
        header.pack(fill="x", padx=28, pady=(22, 8))

        tk.Label(
            header,
            text="GestureGuru",
            font=title_font,
            bg="#F3EEE6",
            fg="#1F1A17",
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Interactive Indian Sign Language Learning System — alphabets A–Z and numbers 1–9.",
            font=body_font,
            bg="#F3EEE6",
            fg="#5A534C",
        ).pack(anchor="w", pady=(4, 0))

        model_ok = os.path.exists(MODEL_PATH)
        status = (
            f"Model: ready  ·  Python: {self.python_exe}"
            if model_ok
            else f"Model missing at models/model.h5  ·  Python: {self.python_exe}"
        )
        tk.Label(
            header,
            text=status,
            font=small_font,
            bg="#F3EEE6",
            fg="#1B7F5A" if model_ok else "#B42318",
        ).pack(anchor="w", pady=(10, 0))

        grid = tk.Frame(self, bg="#F3EEE6")
        grid.pack(fill="both", expand=True, padx=24, pady=12)

        for i, mode in enumerate(MODES):
            row, col = divmod(i, 2)
            card = tk.Frame(
                grid,
                bg="#FFFCF8",
                highlightbackground="#E4DCD1",
                highlightthickness=1,
                padx=18,
                pady=16,
            )
            card.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)

            accent = tk.Frame(card, bg=mode["accent"], height=4)
            accent.pack(fill="x", pady=(0, 12))

            tk.Label(
                card,
                text=mode["title"],
                font=Font(family="Segoe UI Semibold", size=15),
                bg="#FFFCF8",
                fg="#1F1A17",
            ).pack(anchor="w")
            tk.Label(
                card,
                text=mode["subtitle"],
                font=Font(family="Segoe UI", size=10),
                bg="#FFFCF8",
                fg=mode["accent"],
            ).pack(anchor="w", pady=(2, 8))
            tk.Label(
                card,
                text=mode["detail"],
                font=body_font,
                bg="#FFFCF8",
                fg="#5A534C",
                justify="left",
            ).pack(anchor="w")

            btn = tk.Button(
                card,
                text="Launch",
                font=Font(family="Segoe UI Semibold", size=11),
                bg=mode["accent"],
                fg="white",
                activebackground=mode["accent"],
                activeforeground="white",
                relief="flat",
                padx=14,
                pady=6,
                cursor="hand2",
                command=lambda m=mode: self.start(m),
            )
            btn.pack(anchor="e", pady=(16, 0))

        for r in range(2):
            grid.grid_rowconfigure(r, weight=1)
        for c in range(2):
            grid.grid_columnconfigure(c, weight=1)

        footer = tk.Frame(self, bg="#F3EEE6")
        footer.pack(fill="x", padx=28, pady=(0, 18))
        tk.Button(
            footer,
            text="Open project folder",
            command=self.open_folder,
            relief="flat",
            bg="#E8E0D6",
            fg="#1F1A17",
            padx=10,
            pady=4,
        ).pack(side="left")
        tk.Button(
            footer,
            text="Quit",
            command=self.destroy,
            relief="flat",
            bg="#E8E0D6",
            fg="#1F1A17",
            padx=10,
            pady=4,
        ).pack(side="right")

        self.log = tk.Label(footer, text="", font=small_font, bg="#F3EEE6", fg="#5A534C")
        self.log.pack(side="left", padx=16)

    def open_folder(self):
        try:
            if sys.platform.startswith("win"):
                os.startfile(PROJECT_ROOT)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", PROJECT_ROOT])
            else:
                subprocess.Popen(["xdg-open", PROJECT_ROOT])
        except Exception as e:
            messagebox.showwarning("Open folder", str(e))

    def start(self, mode):
        if mode["needs_model"] and not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Model missing",
                "models/model.h5 was not found.\n\n"
                "Run setup.bat first, or place your trained model in the models folder.",
            )
            return
        try:
            proc = launch_script(self.python_exe, mode["script"])
            self.log.config(text=f"Started {mode['title']} (pid {proc.pid})")
        except Exception as e:
            messagebox.showerror("Launch error", str(e))


if __name__ == "__main__":
    app = LauncherGUI()
    app.mainloop()
