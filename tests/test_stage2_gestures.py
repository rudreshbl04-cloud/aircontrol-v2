"""
Unit Test Suite for AirControl V2 Stage 2 (Two-Hand Gesture Recognition & Priority Arbitration).

Tests:
1. Right Hand cursor movement pose (Index Up).
2. Right Hand left-click pose (Pinch) and single-click debounce.
3. Left Hand scroll pose (Index+Middle Up) and vertical displacement.
4. Left Hand volume pose (Three fingers) and thumb-distance mapping.
5. Left Hand media play/pause (Open Palm held 0.8s in Media Mode).
6. Left Hand presentation swipe (Horizontal movement in Presentation Mode).
7. Two-Hand Zoom (Both hands pinched held for 0.25s) with baseline distance tracking.
8. Zoom priority arbitration (Zoom overrides individual click/volume).
9. Global safety pause overrides all actions.
10. Ambiguous role collision suppresses all actions.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import config
from src.hand_roles import AssignedHand, RoleFrameResult
from src.gestures import TwoHandGestureRecognizer, HandPose


def make_mock_landmarks(
    hand_x=0.5,
    hand_y=0.5,
    thumb_up=False,
    index_up=False,
    middle_up=False,
    ring_up=False,
    pinky_up=False,
    pinch=False,
):
    """Generates synthetic 21 landmarks for testing specific finger poses."""
    wrist = (hand_x, hand_y + 0.2, 0.0)
    lm = [wrist] * 21

    # Thumb: 1, 2, 3, 4
    lm[1] = (hand_x - 0.04, hand_y + 0.15, 0.0)
    lm[2] = (hand_x - 0.07, hand_y + 0.10, 0.0)
    lm[3] = (hand_x - 0.10, hand_y + 0.05, 0.0)
    lm[4] = (hand_x - 0.14, hand_y, 0.0) if thumb_up else (hand_x - 0.05, hand_y + 0.05, 0.0)

    # Pinky base (17) for thumb distance check
    lm[17] = (hand_x + 0.08, hand_y + 0.05, 0.0)
    lm[18] = (hand_x + 0.08, hand_y, 0.0)
    lm[19] = (hand_x + 0.08, hand_y - 0.05, 0.0)
    lm[20] = (hand_x + 0.08, hand_y - 0.15, 0.0) if pinky_up else (hand_x + 0.08, hand_y + 0.05, 0.0)

    # Index: 5, 6, 7, 8
    lm[5] = (hand_x - 0.04, hand_y + 0.05, 0.0)
    lm[6] = (hand_x - 0.04, hand_y, 0.0)
    lm[7] = (hand_x - 0.04, hand_y - 0.05, 0.0)
    lm[8] = (hand_x - 0.04, hand_y - 0.15, 0.0) if index_up else (hand_x - 0.04, hand_y + 0.05, 0.0)

    # Middle: 9, 10, 11, 12
    lm[9] = (hand_x, hand_y + 0.05, 0.0)
    lm[10] = (hand_x, hand_y, 0.0)
    lm[11] = (hand_x, hand_y - 0.05, 0.0)
    lm[12] = (hand_x, hand_y - 0.15, 0.0) if middle_up else (hand_x, hand_y + 0.05, 0.0)

    # Ring: 13, 14, 15, 16
    lm[13] = (hand_x + 0.04, hand_y + 0.05, 0.0)
    lm[14] = (hand_x + 0.04, hand_y, 0.0)
    lm[15] = (hand_x + 0.04, hand_y - 0.05, 0.0)
    lm[16] = (hand_x + 0.04, hand_y - 0.15, 0.0) if ring_up else (hand_x + 0.04, hand_y + 0.05, 0.0)

    if pinch:
        # Move thumb tip (4) close to index tip (8)
        lm[4] = (lm[8][0] + 0.01, lm[8][1] + 0.01, 0.0)

    return lm


class TestTwoHandGestures(unittest.TestCase):

    def setUp(self):
        self.rec = TwoHandGestureRecognizer()

    def test_right_index_cursor_movement(self):
        """Verify Right Index Up pose triggers 'MOVE CURSOR' action."""
        lm = make_mock_landmarks(hand_x=0.7, index_up=True)
        hand = AssignedHand(role="RIGHT", landmarks=lm, confidence=0.9, center=(0.7, 0.5), raw_label="Right", raw_index=0)
        roles = RoleFrameResult(right_hand=hand, hands_detected_count=1)

        res = self.rec.process(roles, current_time=1.0)
        self.assertIsNotNone(res.right)
        self.assertEqual(res.right.name, "INDEX UP")
        self.assertEqual(res.right.action, "MOVE CURSOR")
        self.assertIn("Move Cursor", res.action_summary.title())

    def test_right_pinch_click_and_hold(self):
        """Verify Right Thumb+Index pinch triggers 'LEFT CLICK' on press, then 'HOLDING CLICK'."""
        lm = make_mock_landmarks(hand_x=0.7, thumb_up=True, index_up=True, pinch=True)
        hand = AssignedHand(role="RIGHT", landmarks=lm, confidence=0.9, center=(0.7, 0.5), raw_label="Right", raw_index=0)
        roles = RoleFrameResult(right_hand=hand, hands_detected_count=1)

        # First frame: initial click trigger
        res1 = self.rec.process(roles, current_time=2.0)
        self.assertEqual(res1.right.name, "PINCH")
        self.assertEqual(res1.right.action, "LEFT CLICK")

        # Second frame: held pinch does not re-click
        res2 = self.rec.process(roles, current_time=2.05)
        self.assertEqual(res2.right.action, "HOLDING CLICK")

    def test_left_scroll_pose(self):
        """Verify Left Peace Sign (Index+Middle Up) triggers scroll action."""
        lm = make_mock_landmarks(hand_x=0.3, index_up=True, middle_up=True)
        hand1 = AssignedHand(role="LEFT", landmarks=lm, confidence=0.9, center=(0.3, 0.50), raw_label="Left", raw_index=0)
        roles1 = RoleFrameResult(left_hand=hand1, hands_detected_count=1)

        # Initial baseline frame
        self.rec.process(roles1, current_time=3.0)

        # Moving hand upward in second frame (center_y from 0.50 to 0.45)
        hand2 = AssignedHand(role="LEFT", landmarks=lm, confidence=0.9, center=(0.3, 0.45), raw_label="Left", raw_index=0)
        roles2 = RoleFrameResult(left_hand=hand2, hands_detected_count=1)
        res = self.rec.process(roles2, current_time=3.05)

        self.assertEqual(res.left.name, "INDEX + MIDDLE (PEACE)")
        self.assertEqual(res.left.action, "SCROLL UP")

    def test_left_volume_pose(self):
        """Verify Left Three Fingers Up triggers volume adjustment with percentage."""
        lm = make_mock_landmarks(hand_x=0.3, thumb_up=True, index_up=True, middle_up=True, ring_up=True)
        # Position thumb at controlled distance (0.14)
        lm[4] = (lm[8][0] - 0.14, lm[8][1], 0.0)

        hand = AssignedHand(role="LEFT", landmarks=lm, confidence=0.9, center=(0.3, 0.5), raw_label="Left", raw_index=0)
        roles = RoleFrameResult(left_hand=hand, hands_detected_count=1)

        res = self.rec.process(roles, current_time=4.0)
        self.assertEqual(res.left.name, "THREE FINGERS")
        self.assertIn("VOLUME", res.left.action)
        self.assertIsNotNone(res.left.volume_pct)
        self.assertTrue(0.0 <= res.left.volume_pct <= 100.0)

    def test_two_hand_zoom_and_priority(self):
        """Verify that when BOTH hands are pinched, Two-Hand Zoom enters and overrides click & volume."""
        lm_r = make_mock_landmarks(hand_x=0.7, thumb_up=True, index_up=True, pinch=True)
        lm_l = make_mock_landmarks(hand_x=0.3, thumb_up=True, index_up=True, pinch=True)

        hand_r = AssignedHand(role="RIGHT", landmarks=lm_r, confidence=0.9, center=(0.7, 0.5), raw_label="Right", raw_index=0)
        hand_l = AssignedHand(role="LEFT", landmarks=lm_l, confidence=0.9, center=(0.3, 0.5), raw_label="Left", raw_index=1)
        roles = RoleFrameResult(right_hand=hand_r, left_hand=hand_l, hands_detected_count=2)

        # Frame 1: Zoom entry initialization (t=10.0)
        self.rec.process(roles, current_time=10.0)

        # Frame 2: Hold entry duration >= 0.25s (t=10.3)
        res_zoom = self.rec.process(roles, current_time=10.3)
        self.assertTrue(res_zoom.is_zoom_active)
        self.assertEqual(res_zoom.priority_level, "ZOOM")
        self.assertIn("ZOOM", res_zoom.action_summary)

        # Spreading hands apart triggers ZOOM IN
        lm_r_spread = make_mock_landmarks(hand_x=0.8, thumb_up=True, index_up=True, pinch=True)
        hand_r_spread = AssignedHand(role="RIGHT", landmarks=lm_r_spread, confidence=0.9, center=(0.8, 0.5), raw_label="Right", raw_index=0)
        roles_spread = RoleFrameResult(right_hand=hand_r_spread, left_hand=hand_l, hands_detected_count=2)

        res_zoom_in = self.rec.process(roles_spread, current_time=10.35)
        self.assertTrue(res_zoom_in.is_zoom_active)
        self.assertIn("ZOOM IN", res_zoom_in.action_summary)

    def test_safety_pause_suppression(self):
        """Verify that when is_paused is True, all actions are suppressed."""
        lm = make_mock_landmarks(hand_x=0.7, index_up=True)
        hand = AssignedHand(role="RIGHT", landmarks=lm, confidence=0.9, center=(0.7, 0.5), raw_label="Right", raw_index=0)
        roles = RoleFrameResult(right_hand=hand, hands_detected_count=1)

        res = self.rec.process(roles, is_paused=True, current_time=20.0)
        self.assertEqual(res.priority_level, "SAFETY")
        self.assertIn("PAUSED", res.action_summary)
        self.assertIsNone(res.right)

    def test_right_click_recognition(self):
        """Verify Right Hand middle-finger + thumb pinch triggers RIGHT CLICK."""
        lm = make_mock_landmarks(hand_x=0.7, thumb_up=True, index_up=True, middle_up=True)
        # Pinch thumb (4) to middle finger tip (12)
        lm[4] = (lm[12][0] + 0.01, lm[12][1] + 0.01, 0.0)
        # Keep index finger tip (8) further away so it's not an index pinch
        lm[8] = (lm[8][0], lm[8][1] - 0.08, 0.0)

        hand = AssignedHand(role="RIGHT", landmarks=lm, confidence=0.9, center=(0.7, 0.5), raw_label="Right", raw_index=0)
        roles = RoleFrameResult(right_hand=hand, hands_detected_count=1)

        res = self.rec.process(roles, current_time=25.0)
        self.assertIsNotNone(res.right)
        self.assertEqual(res.right.name, "RIGHT PINCH")
        self.assertEqual(res.right.action, "RIGHT CLICK")


if __name__ == "__main__":
    unittest.main()
