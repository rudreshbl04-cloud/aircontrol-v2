# 🖐️ AirControl V2 — Touchless Two-Hand Gesture Control for Windows

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011%20(64--bit)-0078D6.svg?logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe%20Hands-orange.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Offline](https://img.shields.io/badge/Privacy-100%25%20Offline%20%7C%20Zero%20Cloud-success.svg)](#-privacy--offline-first-architecture)
[![Tests](https://img.shields.io/badge/Tests-38%20Passed%20%7C%20100%25-brightgreen.svg)](#-automated-testing--validation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AirControl V2** transforms standard Windows laptop webcams into high-precision, touchless spatial input controllers. Powered by MediaPipe Hands and PyAutoGUI, it tracks **two hands concurrently**, distinguishes physical Left and Right hands, and executes simultaneous desktop automation (**"both at a time"**) — cursor navigation, left/right clicks, drag-and-drop, smooth scrolling, volume control, zoom, and slide navigation with zero cloud dependency.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Complete Gesture & Action Matrix](#-complete-gesture--action-matrix)
- [Quick Start Guide](#-quick-start-guide)
- [Installation & Manual Setup](#-installation--manual-setup)
- [Runtime Modes & Controls](#-runtime-modes--controls)
- [Keyboard Hotkeys](#-keyboard-hotkeys)
- [Settings & Customization](#-settings--customization)
- [Hardware Health Diagnostic](#-hardware-health-diagnostic)
- [Automated Testing & Validation](#-automated-testing--validation)
- [Architecture & Pipeline](#-architecture--pipeline)
- [Privacy & Offline-First Architecture](#-privacy--offline-first-architecture)
- [Troubleshooting & FAQ](#-troubleshooting--faq)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Key Features

### 1. Simultaneous Two-Hand Control ("Both at a Time")
- **Physical Right Hand**: Dedicated cursor positioning, left-clicking, drag-and-drop, and right-clicking.
- **Physical Left Hand**: Dedicated continuous scrolling, Windows Core Audio volume adjustment, media play/pause, and presentation slide swiping.
- **Concurrent Execution**: Move the cursor or drag windows with your Right hand while simultaneously scrolling or adjusting volume with your Left hand without interference.
- **Two-Hand Zoom**: Pinch thumb and index on both hands simultaneously to zoom in/out with continuous distance tracking.

### 2. Universal Single-Hand Operation
- Work comfortably with just **one hand**. If only a single hand is in frame, AirControl V2 automatically enables full universal functionality:
  - Point index finger ➔ Moves mouse cursor
  - Pinch thumb & index ➔ Left clicks or drags
  - Pinch thumb & middle ➔ Right clicks
  - Show peace sign ➔ Scrolls up or down
  - Show three fingers ➔ Adjusts system volume
  - Hold open palm ➔ Plays/pauses media or toggles safety pause

### 3. Continuous Velocity-Adaptive Cursor Smoothing
- Employs a continuous sigmoid velocity filter ($S \in [0.22, 0.92]$) that adapts to hand speed in real-time.
- Eliminates hand tremors during delicate button targeting while allowing rapid, responsive cursor sweeps across multi-monitor setups.
- **6-Frame Tracking Grace Hold**: Momentary 1-frame landmark dropouts never teleport the mouse cursor.

### 4. Orientation & Tilt Invariance
- Joint-ratio algorithms utilize 3D landmark depth vectors ($x, y, z$). Gestures recognize accurately whether hands are held upright, tilted forward toward the webcam at a desk, or angled horizontally.
- Distance-invariant palm-scale normalization ($palm\_scale = \|wrist - middle\_mcp\|$) guarantees crisp pinches regardless of distance from the camera.

### 5. Instant Pinch-and-Drag & Right Click
- **Drag & Drop**: Pinch thumb and index finger, hold for 0.2s, and move to drag windows, files, or highlight text. Releasing the pinch releases the mouse button.
- **Right Click**: Fast thumb-to-middle fingertip contact triggers debounced context menus.

### 6. Hardware-Accurate Windows Audio
- Directly interfaces with Windows Core Audio via `pycaw` (`EndpointVolume`) with exponential smoothing and graceful fallback to system volume keystrokes.

---

## ✋ Complete Gesture & Action Matrix

| Physical Hand | Gesture Pose | Anatomical Finger State | Windows Desktop Action |
| :--- | :--- | :--- | :--- |
| **Right (or Single)** | **INDEX UP** | Index extended straight, other fingers curled | **Move Mouse Cursor** (velocity smoothed, zero lag) |
| **Right (or Single)** | **PINCH (Tap)** | Thumb tip touches Index fingertip ($< 0.2\text{s}$) | **Left Click** (crisp single trigger) |
| **Right (or Single)** | **PINCH (Hold & Move)** | Thumb touches Index, hold $> 0.2\text{s}$ & move hand | **Drag & Drop** (drag files, windows, select text) |
| **Right (or Single)** | **RIGHT PINCH** | Thumb tip touches Middle fingertip | **Right Click** (open context menus) |
| **Left (or Single)** | **PEACE SIGN** | Index + Middle fingers extended | **Scroll Up / Down** (hand elevation sets speed) |
| **Left (or Single)** | **THREE FINGERS** | Index + Middle + Ring extended | **System Volume** (thumb-index distance maps 0–100%) |
| **Left (or Single)** | **OPEN PALM** | All 5 fingers spread, hold $\sim 0.8\text{s}$ | **Play / Pause Media** (sends Spacebar key) |
| **Left (or Single)** | **SWIPE (L / R)** | Quick lateral hand flick | **Prev / Next Slide** (in Presentation Mode) |
| **Both Hands** | **DUAL PINCH** | Both hands pinch thumb-index simultaneously | **Zoom In / Out** (spread hands to zoom in/out) |
| **Any Hand** | **OPEN PALM (2s)** | Open palm held steady for 2.0 seconds | **Safety Pause / Resume** |

---

## 🚀 Quick Start Guide

### Option 1: 1-Click Windows Launcher (Easiest)
Double-click `run.bat` in the repository root directory. It automatically activates the virtual environment and launches AirControl V2.

### Option 2: PowerShell Launch
```powershell
Set-Location "c:\ai.offline"
& "c:\ai.offline\.venv\Scripts\python.exe" "c:\ai.offline\main.py"
```

---

## 📦 Installation & Manual Setup

### Prerequisites
- **OS**: Windows 10 or Windows 11 (64-bit)
- **Python**: Python 3.10, 3.11, or 3.12
- **Hardware**: Standard integrated or USB webcam (640x480 or higher)

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rudreshbl04-cloud/aircontrol-v2.git
   cd aircontrol-v2
   ```

2. **Create and Activate a Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Hardware & Permissions**:
   ```powershell
   python diagnose.py
   ```

5. **Start AirControl V2**:
   ```powershell
   python main.py
   ```

---

## 🎮 Runtime Modes & Controls

AirControl V2 features 3 specialized modes tailored to different workflows:

1. **DESKTOP Mode (Default)**:
   - Full mouse control, clicking, dragging, scrolling, and volume adjustments.
2. **MEDIA Mode**:
   - Optimized for video and audio playback (YouTube, Netflix, Spotify, VLC).
   - Left-hand open palm held for 0.8s sends play/pause toggles.
3. **PRESENTATION Mode**:
   - Optimized for PowerPoint, Google Slides, and PDF presentations.
   - Left-hand lateral swipes trigger slide transitions (Left arrow / Right arrow).

---

## ⌨️ Keyboard Hotkeys

When the camera preview window is focused:

| Key | Action | Description |
| :---: | :--- | :--- |
| **`Space`** or **`P`** | **Pause / Resume** | Instantly freezes or resumes all automation for safety |
| **`T`** | **Toggle Tracking Model** | Switches between **Full Model** (High Precision) and **Lite Model** (High Speed) |
| **`M`** | **Cycle Modes** | Switches between `DESKTOP` ➔ `MEDIA` ➔ `PRESENTATION` |
| **`S`** | **Open Settings GUI** | Launches interactive Tkinter sensitivity & preference dialog |
| **`R`** | **Reset Baseline** | Force-resets mouse tracking baseline and centers coordinates |
| **`Q`** | **Quit Application** | Safely releases webcam and terminates cleanly |

---

## ⚙️ Settings & Customization

Press **`S`** at any time to open the graphical Settings & Preferences UI:

- **Cursor Smoothing Factor** (`0.1` - `0.9`): Tailor the responsiveness vs. smoothing balance.
- **Scroll Sensitivity & Cooldown**: Fine-tune scroll units per gesture.
- **Pinch Thresholds**: Adjust pinch contact sensitivity for different hand sizes.
- **Active Camera Region**: Customize the boundary box mapped to your desktop display.
- **Model Complexity**: Select MediaPipe Full (1) or Lite (0) pipeline.
- **Feature Toggles**: Enable or disable Drag & Drop and Right Click.

*Preferences are saved to `settings.json` and persist across sessions.*

---

## 🩺 Hardware Health Diagnostic

AirControl V2 includes an automated diagnostic utility that inspects your local environment prior to runtime:

```powershell
python diagnose.py
```

It validates:
- [x] Python version compatibility ($\ge 3.10$)
- [x] Dependency availability (`cv2`, `mediapipe`, `pyautogui`, `pycaw`, `comtypes`)
- [x] Webcam device enumeration & lighting conditions
- [x] MediaPipe landmark inference pipeline
- [x] Windows Core Audio subsystem initialization

---

## 🧪 Automated Testing & Validation

The codebase includes an automated test suite with **38 test cases** covering every layer with 100% mocked OS hardware isolation:

```powershell
python -m unittest discover tests
```

### Test Suite Structure:
- [`tests/test_stage1_tracking.py`](tests/test_stage1_tracking.py): Two-hand tracking, model complexity toggle, role assignment, and hand occlusion continuity.
- [`tests/test_stage2_gestures.py`](tests/test_stage2_gestures.py): Gesture classification, single & dual pinch, scroll speed, volume mapping, and priority arbitration.
- [`tests/test_recognition_robustness.py`](tests/test_recognition_robustness.py): Natural curled-finger pinches, 3D tilt invariance, duplicate handedness auto-resolution, and distance scale invariance.
- [`tests/test_controller.py`](tests/test_controller.py): Active area coordinate mapping, continuous velocity smoothing, drag-and-drop, right-click, and FailSafe protection.
- [`tests/test_settings.py`](tests/test_settings.py): JSON configuration parsing, runtime propagation, and fallback resilience.

To run Stage 2 gesture recognition validation:
```powershell
python test_stage2.py
```

---

## 🏗️ Architecture & Pipeline

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Webcam Stream  │ ──> │ Downscaled Inference │ ──> │ MediaPipe Two-Hands  │
│ (640x480 BGR)   │     │    (480x360 RGB)     │     │ (21 3D Landmarks/Hand)│
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ Windows OS Exec │ <── │  Priority Arbiter &  │ <── │ Physical Hand Roles  │
│  (PyAutoGUI /   │     │  Gesture Classifier  │     │ (Mirror Calibration  │
│     pycaw)      │     │  (Dual & Universal)  │     │   & Continuity Hold) │
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
```

---

## 🔒 Privacy & Offline-First Architecture

AirControl V2 is built from the ground up to be strictly privacy-preserving:
- **Zero Cloud Communication**: Operates entirely offline without external API calls or telemetry.
- **Zero Local Disk Recordings**: Video frames are processed in volatile RAM and immediately discarded.
- **Hardware FailSafe**: Moving the mouse cursor into any screen corner triggers PyAutoGUI's FailSafe, instantly pausing all automation.

---

## ❓ Troubleshooting & FAQ

### Cursor doesn't reach the screen edges
Open Settings (**`S`**) and increase the margin of the active camera region (`ACTIVE_AREA_LEFT`, `ACTIVE_AREA_TOP`), or edit `config.py`.

### Automation suddenly paused ("FailSafe triggered")
PyAutoGUI's safety mechanism paused automation because the mouse touched a corner of the screen. Press **`Space`** to resume.

### Camera opens to a black screen
Ensure your physical webcam privacy shutter is open and no other software (Zoom, Teams, OBS) is holding an exclusive lock on the camera device.

### Cursor feels too fast or too slow
Press **`S`** to open the Settings GUI and adjust the **Cursor Smoothing** slider (`0.22` for extra smoothness, `0.70` for instant twitch reaction).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Run Unit Tests (`python -m unittest discover tests`)
5. Push to the Branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details. Free for personal, educational, and commercial use.
