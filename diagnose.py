"""
AirControl System Diagnostic and Health-Check Tool.

Run this script to verify your Windows environment, camera permissions,
device index, MediaPipe compatibility, and gesture detection pipeline.
"""

import os
import sys

# Silence C-level warnings during diagnostic checks
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np


def run_diagnostics():
    print("=" * 64)
    print("       AirControl V1 — System Diagnostic Health Check")
    print("=" * 64)

    all_passed = True

    # 1. Python & Environment
    print("\n[Check 1/5] Checking Python Environment...")
    py_ver = sys.version.split()[0]
    print(f"  • Python Version: {py_ver}")
    print(f"  • Python Executable: {sys.executable}")
    if "3.11" in py_ver or "3.10" in py_ver or "3.12" in py_ver:
        print("  [PASS] Compatible Python version detected.")
    else:
        print("  [WARN] Python version might have MediaPipe compatibility differences.")

    # 2. Dependency Imports
    print("\n[Check 2/5] Checking Required Packages...")
    packages = ["cv2", "mediapipe", "pyautogui", "pycaw", "comtypes"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "Available")
            print(f"  [PASS] {pkg} loaded successfully (Version: {version})")
        except ImportError as err:
            print(f"  [FAIL] Could not load {pkg}: {err}")
            all_passed = False

    # 3. Camera Detection & Windows Permissions
    print("\n[Check 3/5] Testing Windows Camera Access...")
    import cv2
    import config

    working_cameras = []
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                mean_b = float(np.mean(frame))
                h, w, _ = frame.shape
                working_cameras.append((idx, w, h, mean_b))
                status_str = f"Resolution: {w}x{h}, Average Brightness: {mean_b:.1f}"
                if mean_b < 5.0:
                    status_str += " (Warning: Frame is pitch black - check webcam shutter!)"
                print(f"  [PASS] Camera Index {idx}: Active ({status_str})")
            else:
                print(f"  [WARN] Camera Index {idx}: Opened but could not read frame.")
            cap.release()
        else:
            print(f"  • Camera Index {idx}: Not present.")

    if working_cameras:
        configured_idx = config.CAMERA_INDEX
        found_configured = any(c[0] == configured_idx for c in working_cameras)
        if found_configured:
            print(f"  [PASS] Configured CAMERA_INDEX = {configured_idx} is ready to use.")
        else:
            print(f"  [RECOMMENDATION] Camera index {configured_idx} is not working.")
            print(f"     Change CAMERA_INDEX = {working_cameras[0][0]} in config.py.")
    else:
        print("  [FAIL] No working cameras detected!")
        print("  Troubleshooting Camera Permissions on Windows:")
        print("    1. Open Windows Settings > Privacy & Security > Camera")
        print("    2. Enable 'Camera access' toggle")
        print("    3. Enable 'Let desktop apps access your camera' toggle")
        print("    4. If your laptop has a physical webcam slider/switch, slide it OPEN.")
        all_passed = False

    # 4. MediaPipe Hand Tracker
    print("\n[Check 4/5] Testing MediaPipe Hand Tracker Initialization...")
    try:
        from src.hand_tracker import HandTracker
        tracker = HandTracker()
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.find_hand_landmarks(dummy)
        tracker.close()
        print("  [PASS] MediaPipe Hands initialized and executed inference locally.")
    except Exception as err:
        print(f"  [FAIL] HandTracker failed: {err}")
        all_passed = False

    # 5. Gesture Recognition Engine
    print("\n[Check 5/5] Testing Gesture Recognizer Pipeline...")
    try:
        from src.gestures import GestureRecognizer
        recognizer = GestureRecognizer()
        res = recognizer.recognize(None)
        if res.get("gesture") == "NO HAND":
            print("  [PASS] GestureRecognizer initialized and validated.")
        else:
            print(f"  [WARN] Unexpected response: {res}")
    except Exception as err:
        print(f"  [FAIL] GestureRecognizer error: {err}")
        all_passed = False

    print("\n" + "=" * 64)
    if all_passed and working_cameras:
        print("  HEALTH CHECK RESULT: ALL SYSTEMS OPERATIONAL! [READY]")
        print("  You can now start AirControl using:")
        print('  Set-Location "c:\\ai.offline"; & "c:\\ai.offline\\.venv\\Scripts\\python.exe" "c:\\ai.offline\\main.py"')
    else:
        print("  HEALTH CHECK RESULT: ATTENTION NEEDED (Review items marked [FAIL] above)")
    print("=" * 64)


if __name__ == "__main__":
    run_diagnostics()
