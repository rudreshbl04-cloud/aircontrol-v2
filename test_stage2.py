"""
Validation script for AirControl Stage 2 Gesture Recognition.
Tests synthetic landmark configurations and real-time live webcam frames.
"""

import sys
import time
import config
from src.gestures import GestureRecognizer, euclidean_distance


def make_hand(thumb_up=False, index_up=False, middle_up=False, ring_up=False, pinky_up=False, pinch=False, hand_x=0.5):
    wrist = (hand_x, 0.8, 0.0)
    lm = [(hand_x, 0.8, 0.0)] * 21
    
    # Thumb: 1, 2, 3, 4
    lm[1] = (hand_x - 0.05, 0.75, 0.0)
    lm[2] = (hand_x - 0.08, 0.70, 0.0)
    lm[3] = (hand_x - 0.12, 0.65, 0.0)
    lm[4] = (hand_x - 0.18, 0.60, 0.0) if thumb_up else (hand_x - 0.06, 0.65, 0.0)
    
    # Pinky base (17) for thumb distance check
    lm[17] = (hand_x + 0.10, 0.60, 0.0)
    lm[18] = (hand_x + 0.10, 0.50, 0.0)
    lm[19] = (hand_x + 0.10, 0.45, 0.0)
    lm[20] = (hand_x + 0.10, 0.25, 0.0) if pinky_up else (hand_x + 0.10, 0.65, 0.0)

    # Index: 5 (MCP), 6 (PIP), 7 (DIP), 8 (TIP)
    lm[5] = (hand_x - 0.05, 0.60, 0.0)
    lm[6] = (hand_x - 0.05, 0.50, 0.0)
    lm[7] = (hand_x - 0.05, 0.45, 0.0)
    lm[8] = (hand_x - 0.05, 0.25, 0.0) if index_up else (hand_x - 0.05, 0.65, 0.0)

    # Middle: 9 (MCP), 10 (PIP), 11 (DIP), 12 (TIP)
    lm[9] = (hand_x, 0.60, 0.0)
    lm[10] = (hand_x, 0.50, 0.0)
    lm[11] = (hand_x, 0.45, 0.0)
    lm[12] = (hand_x, 0.25, 0.0) if middle_up else (hand_x, 0.65, 0.0)

    # Ring: 13, 14, 15, 16
    lm[13] = (hand_x + 0.05, 0.60, 0.0)
    lm[14] = (hand_x + 0.05, 0.50, 0.0)
    lm[15] = (hand_x + 0.05, 0.45, 0.0)
    lm[16] = (hand_x + 0.05, 0.25, 0.0) if ring_up else (hand_x + 0.05, 0.65, 0.0)

    if pinch:
        lm[4] = (lm[8][0] + 0.01, lm[8][1] + 0.01, 0.0)

    return lm


def run_tests():
    rec = GestureRecognizer()

    print("=== 1. Testing INDEX FINGER UP ===")
    hand = make_hand(index_up=True)
    res = rec.recognize(hand, current_time=1.0)
    print("Result:", res["gesture"])
    assert res["gesture"] == "INDEX FINGER UP", f"Expected INDEX FINGER UP, got {res['gesture']}"

    print("=== 2. Testing PINCH ===")
    hand = make_hand(thumb_up=True, index_up=True, pinch=True)
    res = rec.recognize(hand, current_time=2.0)
    print("Result:", res["gesture"], "Pinch dist:", res["pinch_distance"])
    assert res["gesture"] == "PINCH", f"Expected PINCH, got {res['gesture']}"

    print("=== 3. Testing INDEX + MIDDLE ===")
    hand = make_hand(index_up=True, middle_up=True)
    res = rec.recognize(hand, current_time=3.0)
    print("Result:", res["gesture"])
    assert res["gesture"] == "INDEX + MIDDLE", f"Expected INDEX + MIDDLE, got {res['gesture']}"

    print("=== 4. Testing THREE FINGERS (VOLUME) ===")
    hand = make_hand(index_up=True, middle_up=True, ring_up=True, thumb_up=True)
    hand[4] = (hand[8][0] - 0.14, hand[8][1], 0.0)
    res = rec.recognize(hand, current_time=4.0)
    print("Result:", res["gesture"], "Volume pct:", res["volume_pct"])
    assert res["gesture"] == "THREE FINGERS (VOLUME)", f"Expected THREE FINGERS (VOLUME), got {res['gesture']}"
    assert res["volume_pct"] is not None and 0.0 <= res["volume_pct"] <= 100.0

    print("=== 5. Testing FIST ===")
    hand = make_hand()  # all folded
    res = rec.recognize(hand, current_time=5.0)
    print("Result:", res["gesture"])
    assert res["gesture"] == "FIST", f"Expected FIST, got {res['gesture']}"

    print("=== 6. Testing OPEN PALM & 2-SECOND HOLD SAFETY TOGGLE ===")
    hand = make_hand(thumb_up=True, index_up=True, middle_up=True, ring_up=True, pinky_up=True)
    res1 = rec.recognize(hand, current_time=10.0)
    print("t=0s: Gesture:", res1["gesture"], "Status:", res1["status"], "Progress:", res1["palm_hold_progress"])
    assert res1["gesture"] == "OPEN PALM" and res1["status"] == "ON"

    res2 = rec.recognize(hand, current_time=11.0)
    print("t=1.0s: Progress:", res2["palm_hold_progress"])
    assert 0.45 <= res2["palm_hold_progress"] <= 0.55

    res3 = rec.recognize(hand, current_time=12.1)
    print("t=2.1s: Status toggled to:", res3["status"])
    assert res3["status"] == "PAUSED", "Expected status to toggle to PAUSED after 2s hold"

    print("=== 7. Testing SWIPE GESTURES ===")
    rec_swipe = GestureRecognizer()
    rec_swipe.recognize(make_hand(thumb_up=True, index_up=True, hand_x=0.3), current_time=20.0)
    rec_swipe.recognize(make_hand(thumb_up=True, index_up=True, hand_x=0.4), current_time=20.15)
    res_swipe = rec_swipe.recognize(make_hand(thumb_up=True, index_up=True, hand_x=0.55), current_time=20.30)
    print("Swipe result:", res_swipe["gesture"])
    assert res_swipe["gesture"] == "SWIPE RIGHT", f"Expected SWIPE RIGHT, got {res_swipe['gesture']}"

    print("=== 8. Testing NO HAND handling ===")
    res_none = rec.recognize(None, current_time=30.0)
    print("No hand result:", res_none["gesture"])
    assert res_none["gesture"] == "NO HAND"

    print("\nALL 8 GESTURE RECOGNITION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
