"""
Unit Test Suite for AirControl V2 Stage 1 (Two-Hand Tracking & Role Assignment).

Tests:
1. Two hands detected with distinct handedness (Right + Left).
2. Single physical right hand detected.
3. Single physical left hand detected.
4. Hand occlusion tolerance (missing for 2 frames does not immediately drop identity).
5. Hand disappearance (missing for >5 frames triggers just_lost event and resets state).
6. Hand crossing / overlap ambiguity detection.
7. Conflicting handedness resolution.
8. Physical calibration under mirrored vs non-mirrored preview.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import config
from src.hand_tracker import RawHandData
from src.hand_roles import HandRoleManager, AssignedHand


def create_mock_hand(raw_label="Right", center_x=0.7, center_y=0.5, score=0.95):
    """Creates a synthetic RawHandData object for testing."""
    # 21 mock landmarks
    landmarks = [(center_x + (i * 0.001), center_y + (i * 0.001), 0.0) for i in range(21)]
    return RawHandData(
        landmarks=landmarks,
        raw_label=raw_label,
        score=score,
        center=(center_x, center_y),
    )


class TestTwoHandRoles(unittest.TestCase):

    def setUp(self):
        self.mgr = HandRoleManager(
            confidence_threshold=0.60,
            occlusion_max_frames=5,
            ambiguity_threshold=0.08,
        )

    def test_two_distinct_hands(self):
        """Verify that when MediaPipe detects two distinct hands, both are correctly assigned."""
        # In mirrored mode, raw 'Right' = physical RIGHT, raw 'Left' = physical LEFT
        right_raw = create_mock_hand(raw_label="Right", center_x=0.75, center_y=0.5)
        left_raw = create_mock_hand(raw_label="Left", center_x=0.25, center_y=0.5)

        res = self.mgr.update([right_raw, left_raw])
        self.assertFalse(res.is_ambiguous)
        self.assertEqual(res.hands_detected_count, 2)
        self.assertIsNotNone(res.right_hand)
        self.assertIsNotNone(res.left_hand)
        self.assertEqual(res.right_hand.role, "RIGHT")
        self.assertEqual(res.left_hand.role, "LEFT")
        self.assertAlmostEqual(res.right_hand.center[0], 0.75)
        self.assertAlmostEqual(res.left_hand.center[0], 0.25)

    def test_single_right_hand(self):
        """Verify that when only the physical Right hand is present, Left is None."""
        right_raw = create_mock_hand(raw_label="Right", center_x=0.75, center_y=0.5)
        res = self.mgr.update([right_raw])

        self.assertFalse(res.is_ambiguous)
        self.assertEqual(res.hands_detected_count, 1)
        self.assertIsNotNone(res.right_hand)
        self.assertIsNone(res.left_hand)
        self.assertEqual(res.right_hand.role, "RIGHT")

    def test_single_left_hand(self):
        """Verify that when only the physical Left hand is present, Right is None."""
        left_raw = create_mock_hand(raw_label="Left", center_x=0.25, center_y=0.5)
        res = self.mgr.update([left_raw])

        self.assertFalse(res.is_ambiguous)
        self.assertEqual(res.hands_detected_count, 1)
        self.assertIsNone(res.right_hand)
        self.assertIsNotNone(res.left_hand)
        self.assertEqual(res.left_hand.role, "LEFT")

    def test_brief_occlusion_tolerance(self):
        """Verify that a brief 2-frame occlusion does not immediately trigger hand loss."""
        right_raw = create_mock_hand(raw_label="Right", center_x=0.75, center_y=0.5)
        self.mgr.update([right_raw])  # Frame 1: Right hand present

        # Frame 2 & 3: Hand temporarily lost
        res2 = self.mgr.update([])
        self.assertFalse(res2.right_just_lost)
        res3 = self.mgr.update([])
        self.assertFalse(res3.right_just_lost)

    def test_hand_disappearance_triggers_lost_event(self):
        """Verify that when hand is missing for > occlusion_max_frames, right_just_lost is flagged."""
        right_raw = create_mock_hand(raw_label="Right", center_x=0.75, center_y=0.5)
        self.mgr.update([right_raw])

        # Missing for 6 consecutive frames (> 5 max frames)
        for _ in range(5):
            self.mgr.update([])
        res_final = self.mgr.update([])

        self.assertTrue(res_final.right_just_lost)
        self.assertIsNone(res_final.right_hand)

    def test_overlapping_hands_trigger_ambiguity(self):
        """Verify that hands crossing within ambiguity threshold are flagged as ambiguous."""
        # Hands very close together (dx = 0.03, threshold = 0.08)
        hand1 = create_mock_hand(raw_label="Right", center_x=0.50, center_y=0.5)
        hand2 = create_mock_hand(raw_label="Left", center_x=0.53, center_y=0.5)

        res = self.mgr.update([hand1, hand2])
        self.assertTrue(res.is_ambiguous)
        self.assertIn("overlapping", res.ambiguity_reason.lower())
        self.assertIsNone(res.right_hand)
        self.assertIsNone(res.left_hand)

    def test_calibration_unmirrored_camera(self):
        """Verify that if FLIP_CAMERA is set to False, mapping inverts correctly."""
        orig_flip = config.FLIP_CAMERA
        try:
            config.FLIP_CAMERA = False
            mgr_unmirrored = HandRoleManager()
            # Under unmirrored camera, MediaPipe 'Right' corresponds to physical LEFT
            raw_h = create_mock_hand(raw_label="Right", center_x=0.3, center_y=0.5)
            res = mgr_unmirrored.update([raw_h])
            self.assertIsNotNone(res.left_hand)
            self.assertEqual(res.left_hand.role, "LEFT")
        finally:
            config.FLIP_CAMERA = orig_flip

    def test_model_complexity_toggle(self):
        """Verify HandTracker can switch between Full (1) and Lite (0) complexity."""
        from unittest.mock import MagicMock, patch
        with patch("mediapipe.solutions.hands.Hands") as mock_hands_cls:
            from src.hand_tracker import HandTracker
            tracker = HandTracker(model_complexity=1)
            self.assertEqual(tracker.model_complexity, 1)

            # Toggle to Lite (0)
            new_c = tracker.toggle_model_complexity()
            self.assertEqual(new_c, 0)
            self.assertEqual(tracker.model_complexity, 0)

            # Toggle back to Full (1)
            new_c2 = tracker.toggle_model_complexity()
            self.assertEqual(new_c2, 1)
            self.assertEqual(tracker.model_complexity, 1)


if __name__ == "__main__":
    unittest.main()
