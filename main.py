"""
AirControl - Control Your Laptop With Hand Gestures
Version: V2
Stage 1: Two-Hand Tracking & Physical Role Calibration

Main application entry point orchestrating two-hand detection,
handedness assignment, role stability, HUD overlays, and clean shutdown.
NO OS automation (mouse/keyboard control) is executed in this stage.
"""

import os
import sys

# Silence low-level C++ logging from MediaPipe and OpenCV on Windows
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["ABSL_LOG_PREFIX"] = "0"

import time
from typing import Optional

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
    from src.hand_roles import HandRoleManager
    from src.overlay import CameraOverlay, COLOR_RIGHT_HAND, COLOR_LEFT_HAND
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
    """Attempts to open and configure the webcam device with native Windows backend."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main() -> None:
    """Main application loop for AirControl V2 Stage 1 (Two-Hand Tracking)."""
    print("=" * 64)
    print("  AirControl V2 — Two-Hand Gesture Control (Stage 1: Tracking)")
    print("  Tracking 2 Hands | Physical Role Calibration | 100% Offline")
    print("=" * 64)

    # 1. Connect Webcam
    print(f"\n[1/3] Connecting to webcam (Device Index {config.CAMERA_INDEX})...")
    cap = open_webcam(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    if cap is None:
        print("\n[ERROR] Could not open webcam.")
        print("Troubleshooting steps:")
        print("  1. Check Windows Camera Privacy Settings:")
        print("     Settings > Privacy & Security > Camera > Enable 'Camera access'")
        print("  2. Ensure no other application (Zoom, Teams, Skype, Browser) is using the camera.")
        print("  3. If your laptop has a webcam privacy slider/switch, verify it is open.")
        print(f"  4. In config.py, change CAMERA_INDEX = 1 if your system has multiple cameras.")
        sys.exit(1)

    print("[✓] Webcam connected successfully.")

    # 2. Initialize Two-Hand Tracking & Role Engine
    print("\n[2/3] Initializing MediaPipe two-hand tracking and role manager...")
    try:
        tracker = HandTracker(max_num_hands=config.MAX_NUM_HANDS)
        role_manager = HandRoleManager()
        overlay = CameraOverlay()
    except Exception as err:
        print(f"\n[ERROR] Engine initialization failed: {err}")
        cap.release()
        sys.exit(1)

    print("[✓] Two-hand tracking engine and role stability manager ready.")

    # 3. Create Display Window
    print("\n[3/3] Opening camera preview window...")
    window_name = "AirControl V2 - Two-Hand Tracking Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.perf_counter()
    fps = 0.0
    current_mode = "DESKTOP"
    is_paused = False
    first_frame = True

    print("\n" + "=" * 64)
    print("  AIRCONTROL V2 STAGE 1 IS ACTIVE AND RUNNING!")
    print("  • Present ONE or BOTH hands in front of the webcam.")
    print("  • Cyan skeleton = Physical RIGHT Hand.")
    print("  • Amber skeleton = Physical LEFT Hand.")
    print("  • Press 'F8' or Space to toggle PAUSE / RESUME.")
    print("  • Press 'M' to cycle Modes (Desktop / Media / Presentation).")
    print("  • Press 'Q' while focused on the preview window to exit.")
    print("=" * 64 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[AirControl Warning] Camera stream interrupted.")
                break

            # Horizontally mirror the frame for natural mirror interaction
            if config.FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            curr_time = time.perf_counter()

            # Check scene lighting
            is_low_light = check_lighting(frame)

            # Process up to TWO hands
            raw_hands = tracker.process_frame(frame)

            # Assign and stabilize physical LEFT and RIGHT roles
            role_result = role_manager.update(raw_hands)

            # Draw distinct color skeletons for each assigned hand
            if role_result.right_hand is not None:
                tracker.draw_landmarks_for_hand(
                    frame=frame,
                    hand_idx=role_result.right_hand.raw_index,
                    color_bgr=COLOR_RIGHT_HAND,
                )
            if role_result.left_hand is not None:
                tracker.draw_landmarks_for_hand(
                    frame=frame,
                    hand_idx=role_result.left_hand.raw_index,
                    color_bgr=COLOR_LEFT_HAND,
                )

            # Calculate smoothed FPS
            delta = curr_time - prev_time
            prev_time = curr_time
            if delta > 0.0:
                instant_fps = 1.0 / delta
                fps = (0.9 * fps) + (0.1 * instant_fps) if fps > 0.0 else instant_fps

            # Render complete graphical HUD overlay
            overlay.render(
                frame=frame,
                roles=role_result,
                fps=fps,
                is_paused=is_paused,
                mode=current_mode,
                is_low_light=is_low_light,
            )

            # Display preview frame
            cv2.imshow(window_name, frame)

            # Pop window to front on initial launch
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
                print("\n[AirControl] 'Q' pressed. Exiting cleanly...")
                break
            elif key in (ord('m'), ord('M')):
                # Cycle modes: DESKTOP -> MEDIA -> PRESENTATION -> DESKTOP
                mode_cycle = {"DESKTOP": "MEDIA", "MEDIA": "PRESENTATION", "PRESENTATION": "DESKTOP"}
                current_mode = mode_cycle.get(current_mode, "DESKTOP")
                print(f"[AirControl] Switched mode to: {current_mode}")
            elif key in (ord(' '), ord('p'), ord('P')):
                # Safety pause/resume toggle via keyboard
                is_paused = not is_paused
                status_str = "PAUSED" if is_paused else "RESUMED"
                print(f"[AirControl Safety] Application {status_str}")

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
