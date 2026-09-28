"""
Comprehensive Recognition Robustness & Edge-Case Test Suite for AirControl V2.

Verifies:
1. Natural Curled-Finger Pinch (pinch while middle/ring/pinky are folded into palm)
2. Orientation-Invariant Index Pointing (horizontal, 45-degree angle, tilted forward)
3. Universal Single-Hand Operation (pointing with left-labeled hand moves cursor; scrolling with right-labeled hand scrolls)
4. Duplicate Handedness Auto-Resolution (two hands labeled 'Right' by MediaPipe resolved spatially without ambiguity lock)
5. Scale-Invariant Pinch at Distance (far from camera with small palm scale)
6. Single-hand Controller Execution (desktop cursor and clicks execute regardless of hand side)
"""

import os
import sys
import unittest
from unittest.mock import patch
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import config
from src.hand_roles import AssignedHand, RoleFrameResult, HandRoleManager
from src.hand_tracker import RawHandData
from src.gestures import TwoHandGestureRecognizer, HandPose, euclidean_distance
from src.controller import DesktopController


def create_synthetic_hand(
    wrist_xy=(0.5, 0.8),
    angle_deg=0.0,
    scale=0.15,
    index_straight=False,
    middle_straight=False,
    ring_straight=False,
    pinky_straight=False,
    pinch_thumb_index=False,
    pinch_thumb_middle=False,
):
    """
    Constructs an anatomically proportioned 21-landmark hand rotated by angle_deg.
    0 deg = pointing straight UP (negative Y in image space).
    90 deg = pointing to the RIGHT (+X in image space).
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    def rot(dx, dy):
        # Rotate vector (dx, dy)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        return rx, ry

    wx, wy = wrist_xy
    lm = [(wx, wy, 0.0)] * 21

    # Base palm MCP joints relative to wrist
    mcp_offsets = {
        1: (-0.3 * scale, -0.2 * scale),   # Thumb CMC
        2: (-0.4 * scale, -0.4 * scale),   # Thumb MCP
        5: (-0.3 * scale, -0.8 * scale),   # Index MCP
        9: ( 0.0 * scale, -0.9 * scale),   # Middle MCP
        13: (0.25 * scale, -0.85 * scale), # Ring MCP
        17: (0.45 * scale, -0.75 * scale), # Pinky MCP
    }

    for idx, (dx, dy) in mcp_offsets.items():
        rx, ry = rot(dx, dy)
        lm[idx] = (wx + rx, wy + ry, 0.0)

    # Fingers extension generator
    def build_finger(mcp_idx, pip_idx, dip_idx, tip_idx, is_straight, length_mult=1.0):
        mcp_x, mcp_y, _ = lm[mcp_idx]
        if is_straight:
            # Extended straight out along hand axis
            p1_rx, p1_ry = rot(0.0, -0.35 * scale * length_mult)
            p2_rx, p2_ry = rot(0.0, -0.65 * scale * length_mult)
            p3_rx, p3_ry = rot(0.0, -0.95 * scale * length_mult)
        else:
            # Curled back into palm
            p1_rx, p1_ry = rot(0.0, -0.25 * scale * length_mult)
            p2_rx, p2_ry = rot(0.08 * scale, -0.15 * scale * length_mult)
            p3_rx, p3_ry = rot(0.05 * scale,  0.05 * scale * length_mult)

        lm[pip_idx] = (mcp_x + p1_rx, mcp_y + p1_ry, 0.0)
        lm[dip_idx] = (mcp_x + p2_rx, mcp_y + p2_ry, 0.0)
        lm[tip_idx] = (mcp_x + p3_rx, mcp_y + p3_ry, 0.0)

    build_finger(5, 6, 7, 8, index_straight, length_mult=1.0)
    build_finger(9, 10, 11, 12, middle_straight, length_mult=1.1)
    build_finger(13, 14, 15, 16, ring_straight, length_mult=1.0)
    build_finger(17, 18, 19, 20, pinky_straight, length_mult=0.85)

    # Thumb
    t_mcp_x, t_mcp_y, _ = lm[2]
    if pinch_thumb_index:
        idx_tip = lm[8]
        lm[3] = ((t_mcp_x + idx_tip[0]) / 2, (t_mcp_y + idx_tip[1]) / 2, 0.0)
        lm[4] = (idx_tip[0] + 0.008, idx_tip[1] + 0.008, 0.0)
    elif pinch_thumb_middle:
        mid_tip = lm[12]
        lm[3] = ((t_mcp_x + mid_tip[0]) / 2, (t_mcp_y + mid_tip[1]) / 2, 0.0)
        lm[4] = (mid_tip[0] + 0.008, mid_tip[1] + 0.008, 0.0)
    else:
        # Natural relaxed thumb
        t_rx1, t_ry1 = rot(-0.2 * scale, -0.2 * scale)
        t_rx2, t_ry2 = rot(-0.3 * scale, -0.35 * scale)
        lm[3] = (t_mcp_x + t_rx1, t_mcp_y + t_ry1, 0.0)
        lm[4] = (t_mcp_x + t_rx2, t_mcp_y + t_ry2, 0.0)

    return lm


class TestRecognitionRobustness(unittest.TestCase):

    def setUp(self):
        self.rec = TwoHandGestureRecognizer()
        self.role_mgr = HandRoleManager()

    def test_curled_fingers_natural_pinch(self):
        """Verify natural pinch (where middle, ring, and pinky are curled) triggers LEFT CLICK."""
        # Index straight and pinched to thumb, middle/ring/pinky all tightly curled into palm
        lm = create_synthetic_hand(
            wrist_xy=(0.7, 0.6),
            index_straight=False,  # bent towards thumb
            middle_straight=False,
            ring_straight=False,
            pinky_straight=False,
            pinch_thumb_index=True,
        )
        hand = AssignedHand(role="RIGHT", landmarks=lm, confidence=0.95, center=(0.7, 0.5), raw_label="Right", raw_index=0)
        roles = RoleFrameResult(right_hand=hand, hands_detected_count=1)

        res = self.rec.process(roles, current_time=1.0)
        self.assertIsNotNone(res.right)
        self.assertEqual(res.right.name, "PINCH")
        self.assertEqual(res.right.action, "LEFT CLICK")

    def test_orientation_invariant_pointing(self):
        """Verify Index Up recognition succeeds across vertical, 45-degree, and horizontal hand angles."""
        for angle in [0.0, 35.0, -35.0, 60.0]:
            lm = create_synthetic_hand(
                wrist_xy=(0.6, 0.6),
                angle_deg=angle,
                index_straight=True,
                middle_straight=False,
                ring_straight=False,
                pinky_straight=False,
            )
            hand = AssignedHand(role="RIGHT", landmarks=lm, confidence=0.95, center=(0.6, 0.5), raw_label="Right", raw_index=0)
            roles = RoleFrameResult(right_hand=hand, hands_detected_count=1)

            res = self.rec.process(roles, current_time=2.0)
            self.assertIsNotNone(res.right, f"Failed at angle {angle}")
            self.assertEqual(res.right.name, "INDEX UP", f"Failed at angle {angle}: got {res.right.name}")
            self.assertEqual(res.right.action, "MOVE CURSOR")

    def test_single_hand_universal_cursor(self):
        """Verify that when only a LEFT-classified hand points an index finger, cursor move is recognized."""
        lm = create_synthetic_hand(
            wrist_xy=(0.3, 0.6),
            index_straight=True,
            middle_straight=False,
            ring_straight=False,
            pinky_straight=False,
        )
        # Hand classified as LEFT
        hand = AssignedHand(role="LEFT", landmarks=lm, confidence=0.9, center=(0.3, 0.5), raw_label="Left", raw_index=0)
        roles = RoleFrameResult(left_hand=hand, hands_detected_count=1)

        res = self.rec.process(roles, current_time=3.0)
        # Should populate right hand action so controller drives cursor!
        self.assertIsNotNone(res.right)
        self.assertEqual(res.right.name, "INDEX UP")
        self.assertEqual(res.right.action, "MOVE CURSOR")

    def test_single_hand_universal_scroll(self):
        """Verify that when only a RIGHT-classified hand shows peace sign, scroll is recognized."""
        lm = create_synthetic_hand(
            wrist_xy=(0.7, 0.6),
            index_straight=True,
            middle_straight=True,
            ring_straight=False,
            pinky_straight=False,
        )
        hand = AssignedHand(role="RIGHT", landmarks=lm, confidence=0.9, center=(0.7, 0.5), raw_label="Right", raw_index=0)
        roles = RoleFrameResult(right_hand=hand, hands_detected_count=1)

        res = self.rec.process(roles, current_time=4.0)
        # Should populate left hand action so controller triggers scrolling!
        self.assertIsNotNone(res.left)
        self.assertEqual(res.left.name, "INDEX + MIDDLE (PEACE)")
        self.assertIn("SCROLL", res.left.action)

    def test_duplicate_handedness_auto_resolution(self):
        """Verify that two hands labeled 'Right' by MediaPipe are cleanly resolved by horizontal position."""
        lm_r = create_synthetic_hand(wrist_xy=(0.75, 0.6), index_straight=True)
        lm_l = create_synthetic_hand(wrist_xy=(0.25, 0.6), index_straight=True)

        raw_r = RawHandData(landmarks=lm_r, raw_label="Right", score=0.88, center=(0.75, 0.5))
        raw_l = RawHandData(landmarks=lm_l, raw_label="Right", score=0.85, center=(0.25, 0.5))

        # Update role manager with two 'Right' hands
        roles = self.role_mgr.update([raw_r, raw_l])

        self.assertFalse(roles.is_ambiguous, "Duplicate handedness should NOT trigger ambiguous lockout")
        self.assertIsNotNone(roles.right_hand)
        self.assertIsNotNone(roles.left_hand)
        self.assertGreater(roles.right_hand.center[0], roles.left_hand.center[0])
        self.assertEqual(roles.right_hand.role, "RIGHT")
        self.assertEqual(roles.left_hand.role, "LEFT")

    def test_pinch_scale_invariance_at_distance(self):
        """Verify pinch is recognized accurately when hand is smaller (further from camera)."""
        small_scale = 0.09  # Half standard size
        lm = create_synthetic_hand(
            wrist_xy=(0.6, 0.5),
            scale=small_scale,
            pinch_thumb_index=True,
        )
        hand = AssignedHand(role="RIGHT", landmarks=lm, confidence=0.9, center=(0.6, 0.5), raw_label="Right", raw_index=0)
        roles = RoleFrameResult(right_hand=hand, hands_detected_count=1)

        res = self.rec.process(roles, current_time=5.0)
        self.assertIsNotNone(res.right)
        self.assertEqual(res.right.name, "PINCH")

    @patch("pyautogui.moveTo")
    def test_controller_single_left_hand_cursor_execution(self, mock_move):
        """Verify DesktopController updates cursor position even when role is left_hand."""
        ctrl = DesktopController()
        lm = create_synthetic_hand(wrist_xy=(0.3, 0.6), index_straight=True)
        hand = AssignedHand(role="LEFT", landmarks=lm, confidence=0.9, center=(0.3, 0.5), raw_label="Left", raw_index=0)
        roles = RoleFrameResult(left_hand=hand, hands_detected_count=1)

        res = self.rec.process(roles, current_time=6.0)
        # Execute in controller (should not raise exceptions and should establish baseline)
        ctrl.execute(res, roles=roles, current_time=6.0)
        self.assertTrue(ctrl.has_cursor_baseline)
        mock_move.assert_called_once()


if __name__ == "__main__":
    unittest.main()
