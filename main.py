"""
AirControl - Control Your Laptop With Hand Gestures
Version: V2 (Complete Production Release)

Main application entry point orchestrating:
- Real-time two-hand tracking (MediaPipe Hands)
- Physical Left/Right hand calibration & occlusion continuity
- Gesture recognition & action priority arbitration
- Full desktop automation (Mouse Cursor, Click, Scroll, Volume, Zoom, Media, Presentation)
- Live HUD overlay with reticles, gesture cheatsheet, and status indicators
- In-memory settings loading and persistent configuration GUI (Settings 'S')
- FailSafe protection & safety auto-pause
- 100% Offline execution
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
import pyautogui

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
    from src.gestures import TwoHandGestureRecognizer
    from src.overlay import CameraOverlay, COLOR_RIGHT_HAND, COLOR_LEFT_HAND
    from src.controller import DesktopController
    from src.settings_ui import load_settings, launch_settings_gui
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
    """Attempts to open and configure the webcam device with auto-failover across available indices."""
    candidate_indices = [camera_index] + [i for i in range(4) if i != camera_index]

    for idx in candidate_indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Verify camera actually delivers valid image frames
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                if idx != camera_index:
                    print(f"[AirControl] Notice: Camera Index {camera_index} unavailable; auto-switched to active Index {idx}.")
                    config.CAMERA_INDEX = idx
                return cap
            cap.release()

    return None


def main() -> None:
    """Main application loop for AirControl V2 (Two-Hand Gesture Control)."""
    # 0. Load any persistent user preferences
    load_settings()

    print("=" * 66)
    print("  AirControl V2 — Two-Hand Gesture Control (Full Working Model)")
    print("  Complete Laptop Automation | Live Action HUD | 100% Offline")
    print("=" * 66)

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

    # 2. Initialize Tracking, Roles, Gestures, and Desktop Controller
    print("\n[2/3] Initializing two-hand tracking, gestures, and automation controller...")
    try:
        tracker = HandTracker(max_num_hands=config.MAX_NUM_HANDS)
        role_manager = HandRoleManager()
        recognizer = TwoHandGestureRecognizer()
        overlay = CameraOverlay()
        controller = DesktopController()
    except Exception as err:
        print(f"\n[ERROR] Engine initialization failed: {err}")
        cap.release()
        sys.exit(1)

    print("[✓] Two-hand tracking, gesture engine, and OS controller ready.")

    # 3. Create Display Window
    print("\n[3/3] Opening camera preview window with live action indicators...")
    window_name = "AirControl V2 - Gesture Control"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.perf_counter()
    fps = 0.0
    current_mode = config.DEFAULT_MODE
    is_paused = False
    failsafe_triggered = False
    first_frame = True

    print("\n" + "=" * 66)
    print("  AIRCONTROL V2 IS ACTIVE AND READY TO CONTROL YOUR LAPTOP!")
    print("  • Look at the GESTURE CHEATSHEET on the top-right of the screen.")
    print("  • RIGHT HAND: Index Up (Move Mouse) | Pinch (Click / Drag) | Middle Pinch (Right Click).")
    print("  • LEFT HAND: Peace Sign (Scroll Up/Down) | 3 Fingers (System Volume).")
    print("  • BOTH AT A TIME: Simultaneous cursor navigation & scrolling.")
    print("  • BOTH HANDS: Pinch Both Hands (Screen Zoom In/Out).")
    print("  • Press 'T' to toggle Tracking Model (Full High-Precision vs Lite Fast).")
    print("  • Press 'Space' to toggle PAUSE / RESUME.")
    print("  • Press 'M' to cycle Modes (Desktop / Media / Presentation).")
    print("  • Press 'S' to open Settings & Sensitivity GUI.")
    print("  • Press 'R' to reset tracking baseline.")
    print("  • Press 'Q' to exit.")
    print("=" * 66 + "\n")

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

            # Classify gestures and arbitrate action priorities
            gesture_result = recognizer.process(
                roles=role_result,
                is_paused=is_paused,
                mode=current_mode,
                current_time=curr_time,
            )

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

            # ------------------------------------------------------------------
            # Execute Desktop Automation (Mouse, Click, Scroll, Volume, Zoom)
            # ------------------------------------------------------------------
            if not is_paused:
                try:
                    controller.execute(
                        gestures=gesture_result,
                        roles=role_result,
                        current_time=curr_time,
                    )
                except pyautogui.FailSafeException:
                    is_paused = True
                    failsafe_triggered = True
                    controller.reset_cursor_baseline()
                    print("\n[AirControl Safety] FailSafe triggered (cursor reached screen corner).")
                    print("Automation auto-paused for safety. Press Space to resume.")
            else:
                controller.reset_cursor_baseline()

            # Calculate smoothed FPS
            delta = curr_time - prev_time
            prev_time = curr_time
            if delta > 0.0:
                instant_fps = 1.0 / delta
                fps = (0.9 * fps) + (0.1 * instant_fps) if fps > 0.0 else instant_fps

            # Render complete graphical HUD overlay with live action indicators
            overlay.render(
                frame=frame,
                roles=role_result,
                gestures=gesture_result,
                fps=fps,
                is_paused=is_paused,
                mode=current_mode,
                is_low_light=is_low_light,
                failsafe_triggered=failsafe_triggered,
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
                mode_cycle = {"DESKTOP": "MEDIA", "MEDIA": "PRESENTATION", "PRESENTATION": "DESKTOP"}
                current_mode = mode_cycle.get(current_mode, "DESKTOP")
                print(f"[AirControl] Switched mode to: {current_mode}")
            elif key in (ord(' '), ord('p'), ord('P')):
                is_paused = not is_paused
                failsafe_triggered = False
                status_str = "PAUSED" if is_paused else "RESUMED"
                print(f"[AirControl Safety] Application {status_str}")
            elif key in (ord('s'), ord('S')):
                print("\n[AirControl] Opening Settings & Preferences dialog...")
                prev_pause = is_paused
                is_paused = True
                try:
                    launch_settings_gui()
                    # Re-apply updated parameters
                    current_mode = config.DEFAULT_MODE
                    print("[AirControl] Settings updated successfully.")
                except Exception as err:
                    print(f"[AirControl] Notice launching settings: {err}")
                is_paused = prev_pause
            elif key in (ord('t'), ord('T')):
                new_c = tracker.toggle_model_complexity()
                model_name = "FULL (High Precision)" if new_c == 1 else "LITE (High Speed)"
                controller.reset_cursor_baseline(force=True)
                print(f"[AirControl Model] Tracking model switched to: {model_name}")
            elif key in (ord('r'), ord('R')):
                controller.reset_cursor_baseline(force=True)
                print("[AirControl] Cursor baseline reset.")

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
