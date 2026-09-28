# AirControl - Control Your Laptop With Hand Gestures

**Version:** V1  
**Status:** In Active Development (Stage 1: Webcam & MediaPipe Hand Tracking)

AirControl is a lightweight, local, offline Python desktop application that uses your laptop's webcam to track your hand in real time and convert intuitive hand gestures into laptop controls (mouse movement, clicking, scrolling, volume adjustment, and presentation/media navigation).

---

## 🔒 100% Offline & Privacy-First

AirControl is strictly designed to run locally:
- **No Internet Required:** Operates 100% offline at runtime.
- **Zero Cloud Services:** No external APIs, telemetry, or remote analytics.
- **Privacy First:** Frames captured by your webcam remain in volatile memory and are **never saved, recorded, or transmitted**.

---

## 📋 V1 Gesture Reference Table

| Gesture | Action | Mode | Notes |
| :--- | :--- | :--- | :--- |
| **Index Finger Up** | Move Mouse Cursor | Mouse | Uses configurable screen mapping & smoothing. |
| **Index + Thumb Pinch** | Left Click | Mouse | Debounced with cooldown to prevent rapid clicks. |
| **Index + Middle Fingers Up** | Scroll Up / Down | Mouse | Natural vertical hand movement scrolling. |
| **Three Fingers Up + Thumb Distance** | Volume Control | Both | Thumb-index distance maps to system volume. |
| **Swipe Left / Right** | Previous / Next Slide | Presentation | Sends Left/Right Arrow key with debouncing. |
| **Fist** | Play / Pause | Presentation | Sends Space key with cooldown. |
| **Open Palm (Held 2s)** | Toggle ON / PAUSED | Safety | Toggles laptop control safety state. |

---

## 🚀 Getting Started on Windows

### Prerequisites
- Windows 10 or 11
- Python 3.11 installed (Official 64-bit CPython recommended)
- A working laptop webcam or USB camera

### Quick Start (Recommended)
You can run AirControl directly using any of these simple methods in `c:\ai.offline`:

**Option A — Double-click launcher (Easiest):**
Double-click [`run.bat`](file:///c:/ai.offline/run.bat) in the project folder.

**Option B — Run in PowerShell / CMD:**
```powershell
.\.venv\Scripts\python.exe main.py
```
*(This bypasses all PowerShell script execution policy restrictions and PATH issues!)*

---

## ⌨️ Keyboard Controls (When Window Is Focused)
- **M**: Switch to **Mouse Mode**
- **P**: Switch to **Presentation Mode**
- **Q**: Safe exit and clean camera release

---

## 🛠️ Configuration
All runtime settings can be customized in [`config.py`](file:///c:/ai.offline/config.py):
- Camera index and resolution
- Active camera region boundaries
- Cursor smoothing factor
- Gesture detection and cooldown thresholds
- Low-light warning threshold

---

## 📄 License
This project is licensed under the MIT License - see the [`LICENSE`](file:///c:/ai.offline/LICENSE) file for details.
