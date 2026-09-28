"""
Settings Management and Graphical Configuration Module for AirControl V2.

Provides:
- Persistent JSON configuration (settings.json)
- Dynamic runtime updates to config module
- Modern, lightweight offline Tkinter configuration dialog
- Fallback safe defaults if config file is corrupted or absent
"""

import json
import os
import sys
from typing import Dict, Any, Optional, Callable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "CAMERA_INDEX": config.CAMERA_INDEX,
    "CAMERA_WIDTH": config.CAMERA_WIDTH,
    "CAMERA_HEIGHT": config.CAMERA_HEIGHT,
    "FLIP_CAMERA": config.FLIP_CAMERA,
    "DEFAULT_MODE": config.DEFAULT_MODE,
    "SMOOTHING_FACTOR": config.SMOOTHING_FACTOR,
    "PINCH_THRESHOLD": config.PINCH_THRESHOLD,
    "SCROLL_AMOUNT": config.SCROLL_AMOUNT,
    "SCROLL_THRESHOLD": config.SCROLL_THRESHOLD,
    "SWIPE_THRESHOLD": config.SWIPE_THRESHOLD,
    "CLICK_COOLDOWN": config.CLICK_COOLDOWN,
    "ACTIVE_AREA_LEFT": config.ACTIVE_AREA_LEFT,
    "ACTIVE_AREA_RIGHT": config.ACTIVE_AREA_RIGHT,
    "ACTIVE_AREA_TOP": config.ACTIVE_AREA_TOP,
    "ACTIVE_AREA_BOTTOM": config.ACTIVE_AREA_BOTTOM,
}


def load_settings() -> Dict[str, Any]:
    """Loads user configuration from settings.json, merging with defaults."""
    settings = dict(DEFAULT_SETTINGS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                user_data = json.load(f)
                if isinstance(user_data, dict):
                    settings.update(user_data)
        except Exception as err:
            print(f"[AirControl Settings] Notice: Could not read {SETTINGS_FILE} ({err}). Using defaults.")

    apply_settings_to_config(settings)
    return settings


def save_settings(settings: Dict[str, Any]) -> bool:
    """Saves given configuration dictionary to settings.json."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        apply_settings_to_config(settings)
        return True
    except Exception as err:
        print(f"[AirControl Settings] Error saving {SETTINGS_FILE}: {err}")
        return False


def apply_settings_to_config(settings: Dict[str, Any]) -> None:
    """Applies dictionary parameters to the active config module in memory."""
    for key, value in settings.items():
        if hasattr(config, key):
            # Maintain expected type
            curr_type = type(getattr(config, key))
            try:
                setattr(config, key, curr_type(value))
            except Exception:
                setattr(config, key, value)


class SettingsGUI:
    """Modern, native desktop GUI for adjusting AirControl preferences."""

    def __init__(self, on_save_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.on_save_callback = on_save_callback
        self.settings = load_settings()
        self.root = None

    def run(self) -> None:
        """Launches the Tkinter settings window."""
        import tkinter as tk
        from tkinter import ttk, messagebox

        root = tk.Tk()
        self.root = root
        root.title("AirControl V2 — Settings & Preferences")
        root.geometry("520x640")
        root.resizable(False, False)

        # Dark theme color palette
        bg_dark = "#1e1e24"
        bg_card = "#2b2b36"
        fg_text = "#f0f0f5"
        accent_cyan = "#00d2ff"
        accent_blue = "#007acc"

        root.configure(bg=bg_dark)

        # Style configuration
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background=bg_dark, foreground=fg_text, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=bg_dark, foreground=accent_cyan, font=("Segoe UI", 13, "bold"))
        style.configure("Card.TFrame", background=bg_card)
        style.configure("TCheckbutton", background=bg_dark, foreground=fg_text, font=("Segoe UI", 10))

        # Header
        header_frame = tk.Frame(root, bg=bg_dark, pady=12)
        header_frame.pack(fill="x", padx=20)
        lbl_title = ttk.Label(header_frame, text="AirControl V2 Preferences", style="Header.TLabel")
        lbl_title.pack(anchor="w")
        lbl_sub = tk.Label(
            header_frame,
            text="Adjust tracking sensitivity, gesture thresholds, and screen mapping.",
            bg=bg_dark,
            fg="#9ca3af",
            font=("Segoe UI", 9),
        )
        lbl_sub.pack(anchor="w", pady=(2, 0))

        # Main Scrollable / Form Frame
        form_frame = tk.Frame(root, bg=bg_dark, padx=20)
        form_frame.pack(fill="both", expand=True)

        # Variable bindings
        var_smoothing = tk.DoubleVar(value=float(self.settings.get("SMOOTHING_FACTOR", 0.35)))
        var_scroll_amt = tk.IntVar(value=int(self.settings.get("SCROLL_AMOUNT", 30)))
        var_pinch_thresh = tk.DoubleVar(value=float(self.settings.get("PINCH_THRESHOLD", 0.05)))
        var_click_cool = tk.DoubleVar(value=float(self.settings.get("CLICK_COOLDOWN", 0.40)))
        var_flip_cam = tk.BooleanVar(value=bool(self.settings.get("FLIP_CAMERA", True)))
        var_cam_idx = tk.IntVar(value=int(self.settings.get("CAMERA_INDEX", 0)))
        var_mode = tk.StringVar(value=str(self.settings.get("DEFAULT_MODE", "DESKTOP")))
        var_active_margin = tk.DoubleVar(value=float(self.settings.get("ACTIVE_AREA_LEFT", 0.15)))

        row = 0

        def add_slider(label_text: str, var: tk.Variable, from_: float, to_: float, res: float, unit: str = ""):
            nonlocal row
            f = tk.Frame(form_frame, bg=bg_dark, pady=6)
            f.pack(fill="x")

            top_row = tk.Frame(f, bg=bg_dark)
            top_row.pack(fill="x")
            lbl = tk.Label(top_row, text=label_text, bg=bg_dark, fg=fg_text, font=("Segoe UI", 9, "bold"))
            lbl.pack(side="left")

            val_lbl = tk.Label(top_row, text=f"{var.get():.2f}{unit}" if isinstance(res, float) else f"{var.get()}{unit}", bg=bg_dark, fg=accent_cyan, font=("Segoe UI", 9))
            val_lbl.pack(side="right")

            def on_slide(val):
                formatted = f"{float(val):.2f}{unit}" if isinstance(res, float) else f"{int(float(val))}{unit}"
                val_lbl.config(text=formatted)

            s = tk.Scale(
                f,
                from_=from_,
                to=to_,
                resolution=res,
                orient="horizontal",
                variable=var,
                command=on_slide,
                bg=bg_card,
                fg=fg_text,
                troughcolor="#141418",
                highlightthickness=0,
                activebackground=accent_cyan,
                showvalue=False,
            )
            s.pack(fill="x", pady=(2, 0))
            row += 1

        # 1. Cursor Smoothing
        add_slider("Cursor Smoothing Factor (Lower = smoother, Higher = faster)", var_smoothing, 0.10, 0.80, 0.05)

        # 2. Scroll Sensitivity
        add_slider("Scroll Speed (Units per tick)", var_scroll_amt, 10, 80, 5)

        # 3. Pinch Detection Threshold
        add_slider("Pinch Distance Threshold (Click sensitivity)", var_pinch_thresh, 0.03, 0.10, 0.01)

        # 4. Click Cooldown
        add_slider("Click Cooldown Time (Debounce seconds)", var_click_cool, 0.20, 0.80, 0.05, "s")

        # 5. Active Region Margin
        add_slider("Camera Active Region Margin (Screen reach box)", var_active_margin, 0.05, 0.25, 0.02)

        # 6. Default Mode
        mode_frame = tk.Frame(form_frame, bg=bg_dark, pady=8)
        mode_frame.pack(fill="x")
        tk.Label(mode_frame, text="Default Startup Mode:", bg=bg_dark, fg=fg_text, font=("Segoe UI", 9, "bold")).pack(side="left")
        mode_menu = ttk.Combobox(mode_frame, textvariable=var_mode, values=["DESKTOP", "MEDIA", "PRESENTATION"], state="readonly", width=14)
        mode_menu.pack(side="right")

        # 7. Camera Options
        cam_frame = tk.Frame(form_frame, bg=bg_dark, pady=8)
        cam_frame.pack(fill="x")
        chk_flip = tk.Checkbutton(
            cam_frame,
            text="Flip Camera Preview (Selfie mirror mode)",
            variable=var_flip_cam,
            bg=bg_dark,
            fg=fg_text,
            selectcolor=bg_card,
            activebackground=bg_dark,
            activeforeground=fg_text,
            font=("Segoe UI", 9),
        )
        chk_flip.pack(side="left")

        cam_idx_frame = tk.Frame(form_frame, bg=bg_dark, pady=4)
        cam_idx_frame.pack(fill="x")
        tk.Label(cam_idx_frame, text="Webcam Device Index:", bg=bg_dark, fg=fg_text, font=("Segoe UI", 9)).pack(side="left")
        spin_cam = tk.Spinbox(cam_idx_frame, from_=0, to=4, textvariable=var_cam_idx, width=4, bg=bg_card, fg=fg_text)
        spin_cam.pack(side="right")

        # Bottom Buttons
        btn_frame = tk.Frame(root, bg=bg_dark, pady=15, padx=20)
        btn_frame.pack(fill="x", side="bottom")

        def on_reset():
            var_smoothing.set(DEFAULT_SETTINGS["SMOOTHING_FACTOR"])
            var_scroll_amt.set(DEFAULT_SETTINGS["SCROLL_AMOUNT"])
            var_pinch_thresh.set(DEFAULT_SETTINGS["PINCH_THRESHOLD"])
            var_click_cool.set(DEFAULT_SETTINGS["CLICK_COOLDOWN"])
            var_flip_cam.set(DEFAULT_SETTINGS["FLIP_CAMERA"])
            var_cam_idx.set(DEFAULT_SETTINGS["CAMERA_INDEX"])
            var_mode.set(DEFAULT_SETTINGS["DEFAULT_MODE"])
            var_active_margin.set(DEFAULT_SETTINGS["ACTIVE_AREA_LEFT"])

        def on_save():
            new_settings = {
                "SMOOTHING_FACTOR": round(var_smoothing.get(), 2),
                "SCROLL_AMOUNT": int(var_scroll_amt.get()),
                "PINCH_THRESHOLD": round(var_pinch_thresh.get(), 3),
                "CLICK_COOLDOWN": round(var_click_cool.get(), 2),
                "FLIP_CAMERA": bool(var_flip_cam.get()),
                "CAMERA_INDEX": int(var_cam_idx.get()),
                "DEFAULT_MODE": str(var_mode.get()),
                "ACTIVE_AREA_LEFT": round(var_active_margin.get(), 2),
                "ACTIVE_AREA_RIGHT": round(1.0 - var_active_margin.get(), 2),
                "ACTIVE_AREA_TOP": round(var_active_margin.get(), 2),
                "ACTIVE_AREA_BOTTOM": round(1.0 - var_active_margin.get(), 2),
            }
            if save_settings(new_settings):
                if self.on_save_callback:
                    self.on_save_callback(new_settings)
                messagebox.showinfo("AirControl Settings", "Preferences saved successfully!")
                root.destroy()
            else:
                messagebox.showerror("Error", "Could not save preferences to settings.json.")

        btn_reset = tk.Button(
            btn_frame,
            text="Reset Defaults",
            command=on_reset,
            bg="#374151",
            fg="#d1d5db",
            relief="flat",
            padx=12,
            pady=6,
            font=("Segoe UI", 9),
            cursor="hand2",
        )
        btn_reset.pack(side="left")

        btn_save = tk.Button(
            btn_frame,
            text="Save & Apply",
            command=on_save,
            bg=accent_blue,
            fg="#ffffff",
            relief="flat",
            padx=16,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        btn_save.pack(side="right")

        root.mainloop()


def launch_settings_gui(on_save_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
    """Helper to instantiate and launch the settings window."""
    gui = SettingsGUI(on_save_callback=on_save_callback)
    gui.run()


if __name__ == "__main__":
    launch_settings_gui()
