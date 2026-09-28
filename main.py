"""
AirControl - Control Your Laptop With Hand Gestures
Version: V1
Stage 2: Gesture Detection Displayed on Screen Only

Main application loop with real-time hand landmark tracking and gesture recognition.
All recognized gestures, statuses, and modes are rendered on screen.
NO mouse movement or keyboard automation is executed in this stage.
"""

import os
import sys

# Silence low-level C++ logging from MediaPipe and OpenCV on Windows
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["ABSL_LOG_PREFIX"] = "0"

import time
from typing import Optional, Dict, Any

# Ensure project root is always in sys.path and is the active working directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import cv2
import numpy as np

try:
    import config
    from src.hand_tracker import HandTracker
    from src.gestures import GestureRecognizer
except ImportError as err:
    print(f"[AirControl Startup Error] Failed to import application modules: {err}")
    print("Please run with:")
    print('  Set-Location "c:\\ai.offline"; & "c:\\ai.offline\\.venv\\Scripts\\python.exe" "c:\\ai.offline\\main.py"')
    sys.exit(1)


def check_lighting(frame: np.ndarray, threshold: float = config.LOW_LIGHT_THRESHOLD) -> bool:
    """Checks if average lighting level across the frame is below the safety threshold."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    return mean_brightness < threshold


def open_webcam(camera_index: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
    """
    Attempts to open and configure the webcam device.
    Uses default native Windows backend to avoid backend incompatibility warnings.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        # Fallback to direct index try
        cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_active_region_boundary(frame: np.ndarray) -> None:
    """Draws a subtle boundary showing the active camera region configured in config.py."""
    h, w, _ = frame.shape
    x_min = int(config.ACTIVE_AREA_LEFT * w)
    x_max = int(config.ACTIVE_AREA_RIGHT * w)
    y_min = int(config.ACTIVE_AREA_TOP * h)
    y_max = int(config.ACTIVE_AREA_BOTTOM * h)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (80, 80, 80), 1)
    cv2.putText(
        frame,
        "Active Region",
        (x_min + 5, y_min + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )


def draw_ui_overlay(
    frame: np.ndarray,
    fps: float,
    gesture_data: Dict[str, Any],
    is_low_light: bool,
    mode: str = "MOUSE",
) -> None:
    """Renders status indicators, FPS, gesture names, progress bars, and mode on the preview frame."""
    h, w, _ = frame.shape
    gesture_name = gesture_data.get("gesture", "NONE")
    status = gesture_data.get("status", "ON")
    palm_progress = gesture_data.get("palm_hold_progress", 0.0)
    vol_pct = gesture_data.get("volume_pct")
    fingers = gesture_data.get("fingers")

    # HUD Card Dimensions
    hud_w = 340
    hud_h = 175
    cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (24, 24, 24), -1)
    cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (70, 70, 70), 1)

    # Title & Version
    cv2.putText(
        frame,
        "AirControl V1 (Stage 2: Detection Only)",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 200),
        1,
        cv2.LINE_AA,
    )

    # Status & Mode
    status_color = (0, 255, 0) if status == "ON" else (0, 165, 255)
    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        status_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Mode: {mode}",
        (180, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Gesture Display Banner with color-coded alerts
    gesture_colors = {
        "INDEX FINGER UP": (255, 200, 0),
        "PINCH": (0, 255, 255),
        "INDEX + MIDDLE": (255, 120, 0),
        "THREE FINGERS (VOLUME)": (200, 255, 50),
        "SWIPE LEFT": (255, 0, 255),
        "SWIPE RIGHT": (255, 0, 255),
        "FIST": (0, 140, 255),
        "OPEN PALM": (0, 255, 0),
        "NO HAND": (120, 120, 120),
        "UNKNOWN GESTURE": (160, 160, 160),
    }
    color = gesture_colors.get(gesture_name, (200, 200, 200))

    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (20, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    # Finger states badge representation [T] [I] [M] [R] [P]
    finger_labels = [("T", "thumb"), ("I", "index"), ("M", "middle"), ("R", "ring"), ("P", "pinky")]
    badge_x = 20
    badge_y = 108
    cv2.putText(
        frame,
        "Fingers:",
        (badge_x, badge_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    badge_x += 65
    for label, key in finger_labels:
        is_up = fingers.get(key, False) if fingers else False
        b_color = (0, 255, 0) if is_up else (80, 80, 80)
        cv2.rectangle(frame, (badge_x, badge_y - 12), (badge_x + 18, badge_y + 4), b_color, -1)
        cv2.putText(
            frame,
            label,
            (badge_x + 4, badge_y - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0) if is_up else (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        badge_x += 24

    # Detail progress bar: Open Palm Hold OR Volume Percentage
    if palm_progress > 0.0:
        bar_x = 20
        bar_y = 125
        bar_w = 300
        bar_h = 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * palm_progress)
        fill_color = (0, 255, 0) if palm_progress >= 1.0 else (0, 200, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
        hold_time = palm_progress * config.OPEN_PALM_HOLD_SECONDS
        cv2.putText(
            frame,
            f"Palm Hold: {hold_time:.1f}s / {config.OPEN_PALM_HOLD_SECONDS:.1f}s",
            (bar_x + 5, bar_y + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    elif vol_pct is not None:
        bar_x = 20
        bar_y = 125
        bar_w = 300
        bar_h = 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * (vol_pct / 100.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 230, 255), -1)
        cv2.putText(
            frame,
            f"Volume Level: {int(vol_pct)}%",
            (bar_x + 5, bar_y + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 137),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # Hotkey navigation guide
    cv2.putText(
        frame,
        "Keys: M = Mouse Mode | P = Presentation Mode | Q = Quit",
        (20, 162),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    # Low light warning banner at bottom
    if is_low_light:
        cv2.rectangle(frame, (10, h - 45), (320, h - 15), (0, 0, 160), -1)
        cv2.putText(
            frame,
            "WARNING: Low lighting detected",
            (18, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    """Main application loop for AirControl Stage 2."""
    print("=" * 64)
    print("  AirControl V1 — Control Your Laptop With Hand Gestures")
    print("  Running 100% Offline | Zero Data Sent | Privacy Protected")
    print("=" * 64)

    # Attempt to open webcam
    print(f"\n[1/3] Connecting to webcam (Device Index {config.CAMERA_INDEX})...")
    cap = open_webcam(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    if cap is None:
        print("\n[ERROR] Could not open webcam.")
        print("Troubleshooting steps:")
        print("  1. Check Windows Camera Permissions:")
        print("     Settings > Privacy & Security > Camera > Turn ON 'Let apps access your camera'.")
        print("  2. Ensure no other app (Zoom, Teams, Discord, Browser) is using the webcam.")
        print("  3. If your laptop has a webcam shutter/switch, verify it is open.")
        print(f"  4. In config.py, change CAMERA_INDEX = 1 if your system has multiple cameras.")
        sys.exit(1)

    print(f"[✓] Webcam connected successfully.")

    # Initialize Hand Tracker and Gesture Recognizer
    print("\n[2/3] Loading offline MediaPipe hand tracking and gesture engine...")
    try:
        tracker = HandTracker()
        recognizer = GestureRecognizer()
    except Exception as err:
        print(f"\n[ERROR] Initialization failed: {err}")
        cap.release()
        sys.exit(1)

    print("[✓] Hand tracking and gesture engine ready.")

    # Window creation
    print("\n[3/3] Opening camera preview window...")
    window_name = "AirControl V1 - Gesture Control Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.perf_counter()
    fps = 0.0
    current_mode = "MOUSE"

    print("\n" + "=" * 64)
    print("  AIRCONTROL IS ACTIVE AND RUNNING!")
    print("  Look at the preview window on your screen.")
    print("  • Hold your hand in front of the webcam to see gesture recognition.")
    print("  • Press 'M' for Mouse Mode, 'P' for Presentation Mode.")
    print("  • Press 'Q' while clicking on the preview window to exit.")
    print("=" * 64 + "\n")

    first_frame = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[AirControl Warning] Webcam stream interrupted.")
                break

            # Horizontally mirror the frame for natural interaction
            if config.FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            curr_time = time.perf_counter()

            # Check scene lighting
            is_low_light = check_lighting(frame)

            # Detect hand landmarks (deterministic 1 hand)
            landmarks = tracker.find_hand_landmarks(frame)

            # Classify gestures (without executing desktop control commands in Stage 2)
            gesture_data = recognizer.recognize(landmarks, current_time=curr_time)

            # Render landmarks on the frame
            if landmarks is not None:
                tracker.draw_landmarks_on_frame(frame)

            # Draw active region boundary
            draw_active_region_boundary(frame)

            # Calculate smoothed FPS
            delta = curr_time - prev_time
            prev_time = curr_time
            if delta > 0.0:
                instant_fps = 1.0 / delta
                fps = (0.9 * fps) + (0.1 * instant_fps) if fps > 0.0 else instant_fps

            # Render HUD overlay
            draw_ui_overlay(
                frame=frame,
                fps=fps,
                gesture_data=gesture_data,
                is_low_light=is_low_light,
                mode=current_mode,
            )

            # Display preview
            cv2.imshow(window_name, frame)

            # Pop window to front on initial launch so it doesn't hide behind terminal
            if first_frame:
                try:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
                except Exception:
                    pass
                first_frame = False

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("\n[AirControl] 'Q' pressed. Shutting down cleanly...")
                break
            elif key in (ord('m'), ord('M')):
                current_mode = "MOUSE"
                print("[AirControl] Switched to MOUSE mode")
            elif key in (ord('p'), ord('P')):
                current_mode = "PRESENTATION"
                print("[AirControl] Switched to PRESENTATION mode")

    except KeyboardInterrupt:
        print("\n[AirControl] KeyboardInterrupt received. Exiting...")
    except Exception as err:
        print(f"\n[AirControl Runtime Error] An unexpected error occurred: {err}")
    finally:
        print("[AirControl] Releasing camera and closing windows...")
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[AirControl] Shutdown complete. Goodbye!")


if __name__ == "__main__":
    main()
