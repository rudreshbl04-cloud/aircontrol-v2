"""
Unit test suite for AirControl V2 Desktop Controller (src/controller.py).

All OS automation calls (mouse movement, clicks, keystrokes) are mocked
to prevent physical mouse movements during automated test execution.
"""

import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import config
from src.hand_roles import AssignedHand, RoleFrameResult
from src.gestures import HandPose, TwoHandGestureResult
from src.controller import DesktopController, AudioController


def make_test_hand(role="RIGHT", index_tip=(0.5, 0.5, 0.0)):
    lm = [(0.5, 0.5, 0.0)] * 21
    lm[8] = index_tip
    return AssignedHand(
        role=role,
        landmarks=lm,
        confidence=0.95,
        center=(index_tip[0], index_tip[1]),
        raw_label="Right" if role == "RIGHT" else "Left",
        raw_index=0,
    )


class TestDesktopController(unittest.TestCase):

    def setUp(self):
        with patch("pyautogui.size", return_value=(1920, 1080)):
            self.ctrl = DesktopController()
            self.ctrl.audio.is_available = False  # Avoid real audio calls in unit tests

    @patch("pyautogui.moveTo")
    def test_cursor_mapping_and_active_area(self, mock_move):
        """Verify normalized coordinates inside active area map across full screen."""
        # Active area: LEFT=0.15, RIGHT=0.85, TOP=0.15, BOTTOM=0.85
        # Fingertip at center (0.50, 0.50) -> norm is (0.5-0.15)/0.7 = 0.5 -> screen center (960, 540)
        coords = self.ctrl.update_cursor(0.50, 0.50)
        self.assertIsNotNone(coords)
        mock_move.assert_called_once()
        call_x, call_y = mock_move.call_args[0]
        self.assertEqual(call_x, 960)
        self.assertEqual(call_y, 540)

    @patch("pyautogui.moveTo")
    def test_cursor_clamping_outside_active_area(self, mock_move):
        """Verify fingertip beyond active margins clamps safely at screen edges."""
        # Top-left corner (0.05, 0.05) < active margins (0.15, 0.15)
        self.ctrl.update_cursor(0.05, 0.05)
        call_x, call_y = mock_move.call_args[0]
        # Should clamp near top-left safe coordinate (not 0,0 which triggers failsafe)
        self.assertTrue(2 <= call_x <= 10)
        self.assertTrue(2 <= call_y <= 10)

    @patch("pyautogui.moveTo")
    def test_cursor_jitter_dead_zone(self, mock_move):
        """Verify micro-movements below jitter dead zone are ignored."""
        # Initial baseline position
        self.ctrl.update_cursor(0.50, 0.50)
        mock_move.reset_mock()

        # Micro motion (e.g. 0.0005 normalized change, ~1 pixel)
        self.ctrl.update_cursor(0.5005, 0.50)
        # Should not call moveTo because distance is within jitter dead zone
        mock_move.assert_not_called()

    @patch("pyautogui.click")
    def test_click_and_debounce(self, mock_click):
        """Verify click triggers once, debounces during cooldown, and re-arms on release."""
        t0 = 10.0
        # 1. First pinch -> click executes
        fired = self.ctrl.trigger_left_click(current_time=t0)
        self.assertTrue(fired)
        mock_click.assert_called_once()

        # 2. Continuous pinch within cooldown -> rejected
        fired_again = self.ctrl.trigger_left_click(current_time=t0 + 0.1)
        self.assertFalse(fired_again)
        self.assertEqual(mock_click.call_count, 1)

        # 3. Time elapsed but hand still pinched (click_ready is False) -> rejected
        fired_without_release = self.ctrl.trigger_left_click(current_time=t0 + 1.0)
        self.assertFalse(fired_without_release)
        self.assertEqual(mock_click.call_count, 1)

        # 4. Release pinch
        self.ctrl.release_click_state()

        # 5. Subsequent pinch after release -> fires second click
        fired_second = self.ctrl.trigger_left_click(current_time=t0 + 1.5)
        self.assertTrue(fired_second)
        self.assertEqual(mock_click.call_count, 2)

    @patch("pyautogui.scroll")
    def test_scroll_trigger_and_direction(self, mock_scroll):
        """Verify scroll dy translates to upward and downward mouse wheel actions."""
        # Upward scroll (dy > 0)
        fired_up = self.ctrl.trigger_scroll(scroll_dy=0.05, current_time=5.0)
        self.assertTrue(fired_up)
        mock_scroll.assert_called_once()
        self.assertGreater(mock_scroll.call_args[0][0], 0)

        # Downward scroll after cooldown
        mock_scroll.reset_mock()
        fired_down = self.ctrl.trigger_scroll(scroll_dy=-0.05, current_time=5.2)
        self.assertTrue(fired_down)
        mock_scroll.assert_called_once()
        self.assertLess(mock_scroll.call_args[0][0], 0)

    @patch("pyautogui.hotkey")
    def test_two_hand_zoom(self, mock_hotkey):
        """Verify two-hand zoom sends Ctrl+ and Ctrl- with rate limiting."""
        # Zoom in
        fired_in = self.ctrl.trigger_zoom(zoom_delta=0.08, current_time=1.0)
        self.assertTrue(fired_in)
        mock_hotkey.assert_called_once_with("ctrl", "+")

        # Rate limited zoom attempt (delta too fast < 0.2s)
        fired_rapid = self.ctrl.trigger_zoom(zoom_delta=0.08, current_time=1.05)
        self.assertFalse(fired_rapid)

        # Zoom out after interval
        mock_hotkey.reset_mock()
        fired_out = self.ctrl.trigger_zoom(zoom_delta=-0.08, current_time=1.3)
        self.assertTrue(fired_out)
        mock_hotkey.assert_called_once_with("ctrl", "-")

    @patch("pyautogui.press")
    def test_slide_navigation(self, mock_press):
        """Verify presentation mode swipe triggers arrow keys."""
        # Next slide (swipe right)
        fired = self.ctrl.trigger_slide("NEXT SLIDE (RIGHT)", current_time=2.0)
        self.assertTrue(fired)
        mock_press.assert_called_once_with("right")

        mock_press.reset_mock()
        # Prev slide (swipe left after cooldown)
        fired_prev = self.ctrl.trigger_slide("PREV SLIDE (LEFT)", current_time=2.8)
        self.assertTrue(fired_prev)
        mock_press.assert_called_once_with("left")

    @patch("pyautogui.moveTo")
    @patch("pyautogui.click")
    def test_safety_and_ambiguity_override_automation(self, mock_click, mock_move):
        """Verify when priority_level is SAFETY or AMBIGUOUS, no OS automation occurs."""
        r_pose = HandPose(name="PINCH", action="LEFT CLICK")
        gestures = TwoHandGestureResult(
            right=r_pose,
            priority_level="SAFETY",
            action_summary="APPLICATION PAUSED",
        )
        roles = RoleFrameResult(right_hand=make_test_hand())

        self.ctrl.execute(gestures=gestures, roles=roles, current_time=3.0)
        mock_click.assert_not_called()
        mock_move.assert_not_called()

    @patch("pyautogui.rightClick")
    def test_right_click_and_debounce(self, mock_right_click):
        """Verify right click triggers on middle-pinch and debounces correctly."""
        t0 = 5.0
        fired = self.ctrl.trigger_right_click(current_time=t0)
        self.assertTrue(fired)
        mock_right_click.assert_called_once()

        # Reject during debounce cooldown
        fired_again = self.ctrl.trigger_right_click(current_time=t0 + 0.1)
        self.assertFalse(fired_again)
        self.assertEqual(mock_right_click.call_count, 1)

        # Allow after cooldown
        fired_later = self.ctrl.trigger_right_click(current_time=t0 + 0.5)
        self.assertTrue(fired_later)
        self.assertEqual(mock_right_click.call_count, 2)

    @patch("pyautogui.mouseDown")
    @patch("pyautogui.mouseUp")
    @patch("pyautogui.moveTo")
    @patch("pyautogui.click")
    def test_drag_and_drop_motion(self, mock_click, mock_move, mock_up, mock_down):
        """Verify holding pinch beyond threshold initiates mouse drag and moves cursor."""
        r_pose = HandPose(name="PINCH", action="LEFT CLICK")
        gestures = TwoHandGestureResult(right=r_pose, priority_level="CONTINUOUS")
        roles = RoleFrameResult(right_hand=make_test_hand(index_tip=(0.50, 0.50)))

        # Frame 1: initial pinch click
        self.ctrl.execute(gestures=gestures, roles=roles, current_time=1.0)
        mock_click.assert_called_once()
        self.assertFalse(self.ctrl.is_dragging)

        # Frame 2: held pinch > 0.20s initiates drag
        roles2 = RoleFrameResult(right_hand=make_test_hand(index_tip=(0.55, 0.55)))
        self.ctrl.execute(gestures=gestures, roles=roles2, current_time=1.25)
        mock_down.assert_called_once()
        self.assertTrue(self.ctrl.is_dragging)
        mock_move.assert_called()

        # Frame 3: release pinch releases drag via mouseUp
        r_idle = HandPose(name="IDLE", action="NONE")
        gestures_idle = TwoHandGestureResult(right=r_idle, priority_level="IDLE")
        self.ctrl.execute(gestures=gestures_idle, roles=roles2, current_time=1.35)
        mock_up.assert_called_once()
        self.assertFalse(self.ctrl.is_dragging)

    @patch("pyautogui.moveTo")
    def test_continuous_velocity_smoothing(self, mock_move):
        """Verify continuous smoothing scales smoothly without teleporting."""
        # Establish baseline
        self.ctrl.update_cursor(0.50, 0.50)
        mock_move.reset_mock()

        # Moderate movement
        coords1 = self.ctrl.update_cursor(0.53, 0.53)
        self.assertIsNotNone(coords1)
        mock_move.assert_called_once()
        c1_x, c1_y = coords1

        mock_move.reset_mock()
        # Fast movement - smoothly tracks forward without abrupt speed cliffs
        coords2 = self.ctrl.update_cursor(0.60, 0.60)
        self.assertIsNotNone(coords2)
        mock_move.assert_called_once()
        c2_x, c2_y = coords2

        self.assertGreater(c2_x, c1_x)
        self.assertGreater(c2_y, c1_y)

    @patch("pyautogui.moveTo")
    def test_cursor_baseline_grace_hold(self, mock_move):
        """Verify brief 2-frame tracking loss does not destroy cursor baseline."""
        # Establish baseline
        self.ctrl.update_cursor(0.50, 0.50)
        self.assertTrue(self.ctrl.has_cursor_baseline)

        # Brief non-force reset for 2 frames
        self.ctrl.reset_cursor_baseline(force=False)
        self.ctrl.reset_cursor_baseline(force=False)
        # Baseline must still be retained due to grace period!
        self.assertTrue(self.ctrl.has_cursor_baseline)

        # Force reset immediately drops baseline
        self.ctrl.reset_cursor_baseline(force=True)
        self.assertFalse(self.ctrl.has_cursor_baseline)

    @patch("pyautogui.moveTo")
    @patch("pyautogui.scroll")
    def test_simultaneous_two_hand_execution(self, mock_scroll, mock_move):
        """Verify Right Hand cursor movement and Left Hand scrolling execute together in same frame."""
        r_pose = HandPose(name="INDEX UP", action="MOVE CURSOR")
        l_pose = HandPose(name="INDEX + MIDDLE (PEACE)", action="SCROLL UP", scroll_dy=0.04)

        gestures = TwoHandGestureResult(
            right=r_pose,
            left=l_pose,
            priority_level="CONTINUOUS",
            action_summary="Right: MOVE CURSOR | Left: SCROLL UP",
        )
        roles = RoleFrameResult(
            right_hand=make_test_hand(role="RIGHT", index_tip=(0.52, 0.52)),
            left_hand=make_test_hand(role="LEFT", index_tip=(0.25, 0.45)),
        )

        self.ctrl.execute(gestures=gestures, roles=roles, current_time=5.0)
        mock_move.assert_called_once()
        mock_scroll.assert_called_once()


if __name__ == "__main__":
    unittest.main()
