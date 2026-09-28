# AirControl V2 — Two-Hand Gesture Control for Windows

**Version:** V2.0.0 (Production Release)  
**Platform:** Windows 10 / 11 (64-bit)  
**Execution:** 100% Offline | Zero Cloud | Zero Telemetry

AirControl V2 is a high-performance, privacy-first computer vision desktop application that turns your laptop webcam into an intuitive touchless controller. Using MediaPipe Hands, AirControl V2 tracks **two hands simultaneously**, distinguishes physical Left and Right hands with mirror calibration, and maps natural gestures into smooth, responsive laptop controls (mouse, click, scroll, volume, zoom, slides, and media).

---

## 🌟 Key Highlights & Features

1. **Simultaneous Two-Hand Interaction**:
   - **Physical RIGHT Hand**: Dedicated to cursor navigation and left-clicking.
   - **Physical LEFT Hand**: Dedicated to continuous scrolling, system volume, presentation slides, and media play/pause.
   - **BOTH HANDS Together**: Synchronized two-hand screen zoom (Ctrl + / -).
2. **Speed-Adaptive Cursor Smoothing**:
   - Dynamic interpolation filter adapts to hand speed. When moving slowly or hovering, sub-pixel smoothing eliminates hand tremors. When moving quickly, smoothing relaxes for instant, zero-lag cursor response.
3. **Hardware-Accurate Windows Audio**:
   - Integrates with Windows Core Audio via `pycaw` (`EndpointVolume`) with exponential smoothing and graceful fallback.
4. **Live On-Screen HUD & Cheatsheet**:
   - Real-time action banner showing active controls.
   - Visual cheatsheet in the top-right corner.
   - Color-coded skeletons (Cyan for Right Hand, Amber for Left Hand).
   - Dynamic fingertip reticles (crosshairs for cursor, filled indicator on click, arrow on scroll).
5. **FailSafe Protection & Auto-Pause**:
   - PyAutoGUI failsafe is strictly enforced (`FAILSAFE = True`).
   - If the cursor reaches any screen corner, automation instantly auto-pauses, alerts the user on screen, and prevents runaway automation.
6. **Graphical Settings & Preferences GUI**:
   - Press **'S'** to open the native settings dialog.
   - Adjust smoothing factor, scroll speed, pinch threshold, click debounce, camera device, and screen margins.
   - Saves persistently to `settings.json` (with automatic fallback to defaults).
7. **100% Offline & Private**:
   - Video frames are processed in volatile RAM and **never saved, recorded, or transmitted**.

---

## ✋ Complete Gesture Reference Table

| Hand | Gesture | Action | Active Mode | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **RIGHT** | **Index Finger Up** | Move Mouse Cursor | All Modes | Speed-adaptive smoothing within active camera region. |
| **RIGHT** | **Thumb-Index Pinch** | Left Click | All Modes | Debounced single click with release re-arm protection. |
| **LEFT** | **Index + Middle Up (Peace Sign)** | Scroll Up / Down | All Modes | Natural hand elevation changes scroll rate and direction. |
| **LEFT** | **Three Fingers Up (Index, Middle, Ring)** | Adjust System Volume | All Modes | Thumb-to-index distance maps 0% to 100% volume. |
| **LEFT** | **Open Palm (Held 0.8s)** | Play / Pause Media | Media Mode | Sends Space key with debounce. |
| **LEFT** | **Horizontal Swipe (Left / Right)** | Prev / Next Slide | Presentation Mode | Sends Left / Right Arrow keys. |
| **BOTH** | **Both Hands Pinched Together** | Screen Zoom In / Out | All Modes | Move hands apart to zoom in; pinch closer to zoom out. |

---

## ⌨️ Live Keyboard Controls

When the camera preview window is focused:
- **`Space`** or **`P`**: Toggle **Pause / Resume** (Global safety pause).
- **`M`**: Cycle Modes (**`DESKTOP`** ➔ **`MEDIA`** ➔ **`PRESENTATION`**).
- **`S`**: Open **Settings & Sensitivity GUI**.
- **`R`**: Reset mouse cursor tracking baseline.
- **`Q`**: Safe and clean application exit.

---

## 🚀 Quick Start (1-Command or 1-Click)

### Option 1: 1-Click Launcher (Easiest)
Double-click [`run.bat`](file:///c:/ai.offline/run.bat) in the project folder.

### Option 2: PowerShell One-Liner (Rock Solid)
```powershell
Set-Location "c:\ai.offline"; & "c:\ai.offline\.venv\Scripts\python.exe" "c:\ai.offline\main.py" 2>$null
```
*(Redirecting `2>$null` silences low-level C++ MediaPipe initialization messages so PowerShell remains completely clean!)*

### Option 3: System Health Diagnostic
To verify camera index, permissions, and audio endpoints:
```powershell
Set-Location "c:\ai.offline"; & "c:\ai.offline\.venv\Scripts\python.exe" "c:\ai.offline\diagnose.py"
```

---

## 🧪 Automated Unit Test Suite

AirControl V2 includes 24 comprehensive unit tests that run in **< 0.2 seconds** with 100% mocked hardware automation:
```powershell
Set-Location "c:\ai.offline"; & "c:\ai.offline\.venv\Scripts\python.exe" -m unittest discover tests
```

Test coverage includes:
- `test_stage1_tracking.py`: MediaPipe hands, physical role calibration, position continuity, ambiguity detection.
- `test_stage2_gestures.py`: Gesture classification, pinch detection, peace sign scrolling, volume calculation, swipe detection, priority arbitration.
- `test_controller.py`: Active region screen mapping, speed-adaptive smoothing, jitter dead zone, click debounce, audio volume, zoom hotkeys, and failsafe isolation.
- `test_settings.py`: JSON config loading, runtime update propagation, default fallbacks.

---

## 🏗️ Architecture & Project Structure

```
c:\ai.offline\
├── config.py                 # Central configuration parameters and thresholds
├── main.py                   # Main loop orchestrating vision, gestures, and controller
├── diagnose.py               # Pre-flight diagnostic and hardware check
├── run.bat                   # 1-Click Windows launcher script
├── settings.json             # User preferences (auto-generated on save, git-ignored)
├── src/
│   ├── hand_tracker.py       # MediaPipe Hands wrapper with fast inference downscaling
│   ├── hand_roles.py         # Physical Left/Right calibration & occlusion tracking
│   ├── gestures.py           # Gesture recognition and multi-hand priority arbitration
│   ├── controller.py         # PyAutoGUI mouse/keyboard and pycaw Windows audio
│   ├── overlay.py            # HUD overlay, reticles, cheatsheet, and action banners
│   └── settings_ui.py        # Modern Tkinter settings dialog
└── tests/
    ├── test_stage1_tracking.py
    ├── test_stage2_gestures.py
    ├── test_controller.py
    └── test_settings.py
```

---

## 📄 License
MIT License. Free for personal, educational, and commercial use.
