"""
Two-Hand Gesture Recognition Engine for AirControl V2.

Responsible for:
- Normalized finger state extraction (extended vs folded)
- Right Hand: Cursor pose (Index Up), Click pose (Thumb-Index Pinch)
- Left Hand: Scroll pose (Index+Middle Up), Volume pose (Three Fingers),
  Media Play/Pause (Open Palm in Media Mode), Presentation Swipe (in Presentation Mode)
- Two-Hand Zoom: Both hands pinched simultaneously with baseline distance tracking
- Action priority arbitration and debouncing

Simulated / Logged actions are produced for Stage 2 (no real OS automation yet).
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import config
from src.hand_roles import AssignedHand, RoleFrameResult


def euclidean_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Calculates 2D normalized Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


@dataclass
class HandPose:
    """Classified pose and parameters for a single hand."""
    name: str                        # e.g. "INDEX UP", "PINCH", "SCROLL POSE", "THREE FINGERS", "OPEN PALM", "FIST", "IDLE"
    action: str                      # e.g. "MOVE CURSOR", "LEFT CLICK", "SCROLL", "VOLUME", "PLAY/PAUSE", "SWIPE", "NONE"
    pinch_distance: float = 1.0      # Thumb tip to index tip distance
    volume_pct: Optional[float] = None
    scroll_dy: float = 0.0           # Vertical displacement for scrolling
    swipe_direction: Optional[str] = None
    fingers: Dict[str, bool] = field(default_factory=dict)


@dataclass
class TwoHandGestureResult:
    """Arbitrated gesture recognition result for the current frame."""
    right: Optional[HandPose] = None
    left: Optional[HandPose] = None
    is_zoom_active: bool = False
    zoom_distance: float = 0.0
    zoom_delta: float = 0.0          # Positive = zoom in, Negative = zoom out
    action_summary: str = "IDLE"     # Plain English summary of active laptop control
    priority_level: str = "NORMAL"   # "SAFETY", "AMBIGUOUS", "ZOOM", "DISCRETE", "CONTINUOUS", "IDLE"


class TwoHandGestureRecognizer:
    """
    Evaluates physical Left and Right hand landmarks to recognize gestures,
    arbitrate priorities, and generate debounced actions.
    """

    def __init__(self) -> None:
        # History queue for Left hand horizontal swipe detection
        self.left_swipe_history: deque = deque(maxlen=20)
        self.last_left_swipe_time: float = 0.0
        self.active_swipe_gesture: Optional[str] = None
        self.swipe_display_until: float = 0.0

        # Left hand scroll baseline tracking
        self.last_scroll_y: Optional[float] = None

        # Left hand open palm hold timer for Play/Pause in Media Mode
        self.left_palm_start_time: Optional[float] = None
        self.last_play_pause_time: float = 0.0

        # Right hand click debounce
        self.right_pinch_was_active: bool = False
        self.last_click_time: float = 0.0

        # Two-Hand Zoom baseline tracking
        self.zoom_entry_start_time: Optional[float] = None
        self.is_zooming: bool = False
        self.zoom_baseline_distance: Optional[float] = None
        self.last_zoom_step_time: float = 0.0

    def get_finger_states(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, bool]:
        """
        Calculates extended vs folded states for all 5 fingers.
        Normalized ratios based on wrist and MCP joints.
        """
        wrist = landmarks[0]

        # Fingers: Index (8), Middle (12), Ring (16), Pinky (20)
        # Extended if TIP is further from wrist than PIP joint and TIP y is higher
        index_extended = (landmarks[8][1] < landmarks[6][1]) and (
            euclidean_distance(wrist, landmarks[8]) > euclidean_distance(wrist, landmarks[6])
        )
        middle_extended = (landmarks[12][1] < landmarks[10][1]) and (
            euclidean_distance(wrist, landmarks[12]) > euclidean_distance(wrist, landmarks[10])
        )
        ring_extended = (landmarks[16][1] < landmarks[14][1]) and (
            euclidean_distance(wrist, landmarks[16]) > euclidean_distance(wrist, landmarks[14])
        )
        pinky_extended = (landmarks[20][1] < landmarks[18][1]) and (
            euclidean_distance(wrist, landmarks[20]) > euclidean_distance(wrist, landmarks[18])
        )

        # Thumb: extended if tip is far from pinky base
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        pinky_mcp = landmarks[17]
        thumb_extended = (
            euclidean_distance(thumb_tip, pinky_mcp) > euclidean_distance(thumb_ip, pinky_mcp) * 1.12
        ) and (
            euclidean_distance(wrist, thumb_tip) > euclidean_distance(wrist, landmarks[2])
        )

        return {
            "thumb": thumb_extended,
            "index": index_extended,
            "middle": middle_extended,
            "ring": ring_extended,
            "pinky": pinky_extended,
        }

    def _classify_right_hand(self, hand: AssignedHand, current_time: float) -> HandPose:
        """Classifies gestures dedicated to the physical RIGHT hand (Cursor & Click)."""
        landmarks = hand.landmarks
        fingers = self.get_finger_states(landmarks)

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_dist = euclidean_distance(thumb_tip, index_tip)

        num_up = sum([fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]])

        # 1. Right Click (Pinch between Thumb and Index)
        # Pinch active when distance < threshold and fingers are not curled into a fist
        if pinch_dist < config.PINCH_THRESHOLD and num_up > 0:
            is_new_pinch = not self.right_pinch_was_active
            self.right_pinch_was_active = True
            
            action = "LEFT CLICK" if is_new_pinch else "HOLDING CLICK"
            return HandPose(
                name="PINCH",
                action=action,
                pinch_distance=pinch_dist,
                fingers=fingers,
            )

        self.right_pinch_was_active = False

        # 2. Right Cursor Movement (Index Finger Up, other fingers folded)
        if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            return HandPose(
                name="INDEX UP",
                action="MOVE CURSOR",
                pinch_distance=pinch_dist,
                fingers=fingers,
            )

        # 3. Idle / Rest pose
        return HandPose(
            name="IDLE",
            action="NONE",
            pinch_distance=pinch_dist,
            fingers=fingers,
        )

    def _classify_left_hand(
        self,
        hand: AssignedHand,
        current_time: float,
        mode: str = "DESKTOP",
    ) -> HandPose:
        """Classifies gestures dedicated to the physical LEFT hand (Scroll, Volume, Media, Swipe)."""
        landmarks = hand.landmarks
        fingers = self.get_finger_states(landmarks)

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_dist = euclidean_distance(thumb_tip, index_tip)
        hand_center_y = hand.center[1]
        hand_center_x = hand.center[0]

        num_up = sum([fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]])
        is_open_palm = (num_up == 4 and fingers["thumb"])

        # ----------------------------------------------------------------------
        # A. PRESENTATION MODE: Horizontal Swipe Detection (Slides)
        # ----------------------------------------------------------------------
        if mode == "PRESENTATION":
            self.left_swipe_history.append((current_time, hand_center_x))
            if current_time - self.last_left_swipe_time >= config.SWIPE_COOLDOWN:
                # Look back ~0.25 to ~0.40s
                old_x = None
                for t, x in self.left_swipe_history:
                    if 0.20 <= (current_time - t) <= 0.45:
                        old_x = x
                        break
                if old_x is not None:
                    dx = hand_center_x - old_x
                    if abs(dx) >= config.SWIPE_THRESHOLD:
                        self.last_left_swipe_time = current_time
                        self.left_swipe_history.clear()
                        direction = "NEXT SLIDE (RIGHT)" if dx > 0 else "PREV SLIDE (LEFT)"
                        self.active_swipe_gesture = direction
                        self.swipe_display_until = current_time + 0.60
                        return HandPose(
                            name="SWIPE",
                            action=direction,
                            swipe_direction=direction,
                            pinch_distance=pinch_dist,
                            fingers=fingers,
                        )

            if current_time < self.swipe_display_until and self.active_swipe_gesture:
                return HandPose(
                    name="SWIPE",
                    action=self.active_swipe_gesture,
                    swipe_direction=self.active_swipe_gesture,
                    pinch_distance=pinch_dist,
                    fingers=fingers,
                )

        # ----------------------------------------------------------------------
        # B. MEDIA MODE: Open Palm Held Briefly for Play/Pause
        # ----------------------------------------------------------------------
        if mode == "MEDIA" and is_open_palm:
            if self.left_palm_start_time is None:
                self.left_palm_start_time = current_time
            
            elapsed = current_time - self.left_palm_start_time
            if elapsed >= 0.80 and (current_time - self.last_play_pause_time >= config.PLAY_PAUSE_COOLDOWN):
                self.last_play_pause_time = current_time
                self.left_palm_start_time = None
                return HandPose(
                    name="OPEN PALM",
                    action="PLAY/PAUSE",
                    pinch_distance=pinch_dist,
                    fingers=fingers,
                )
            return HandPose(
                name="OPEN PALM",
                action=f"HOLDING PLAY/PAUSE ({elapsed:.1f}s)",
                pinch_distance=pinch_dist,
                fingers=fingers,
            )
        else:
            self.left_palm_start_time = None

        # ----------------------------------------------------------------------
        # C. LEFT SCROLL: Index + Middle Fingers Extended (Peace Sign)
        # ----------------------------------------------------------------------
        if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            scroll_dy = 0.0
            if self.last_scroll_y is not None:
                scroll_dy = self.last_scroll_y - hand_center_y  # Moving up is positive dy

            self.last_scroll_y = hand_center_y
            action = "SCROLL UP" if scroll_dy > 0.01 else ("SCROLL DOWN" if scroll_dy < -0.01 else "SCROLL IDLE")
            return HandPose(
                name="INDEX + MIDDLE (PEACE)",
                action=action,
                scroll_dy=scroll_dy,
                pinch_distance=pinch_dist,
                fingers=fingers,
            )

        self.last_scroll_y = None

        # ----------------------------------------------------------------------
        # D. LEFT VOLUME: Three Fingers Extended (Index, Middle, Ring)
        # ----------------------------------------------------------------------
        if fingers["index"] and fingers["middle"] and fingers["ring"] and not fingers["pinky"]:
            clamped_dist = max(config.VOLUME_MIN_DISTANCE, min(pinch_dist, config.VOLUME_MAX_DISTANCE))
            vol_range = config.VOLUME_MAX_DISTANCE - config.VOLUME_MIN_DISTANCE
            vol_pct = ((clamped_dist - config.VOLUME_MIN_DISTANCE) / vol_range) * 100.0

            return HandPose(
                name="THREE FINGERS",
                action=f"VOLUME ({int(vol_pct)}%)",
                volume_pct=vol_pct,
                pinch_distance=pinch_dist,
                fingers=fingers,
            )

        # Idle
        return HandPose(
            name="IDLE",
            action="NONE",
            pinch_distance=pinch_dist,
            fingers=fingers,
        )

    def process(
        self,
        roles: RoleFrameResult,
        is_paused: bool = False,
        mode: str = "DESKTOP",
        current_time: Optional[float] = None,
    ) -> TwoHandGestureResult:
        """
        Processes tracked physical hands, classifies poses, and arbitrates action priorities.
        """
        if current_time is None:
            current_time = time.perf_counter()

        result = TwoHandGestureResult()

        # ----------------------------------------------------------------------
        # Priority 1: Global Safety Pause
        # ----------------------------------------------------------------------
        if is_paused:
            result.action_summary = "PAUSED (ALL AUTOMATION FROZEN)"
            result.priority_level = "SAFETY"
            self.reset_baselines()
            return result

        # ----------------------------------------------------------------------
        # Priority 2: Ambiguous Roles or Overlapping Hands
        # ----------------------------------------------------------------------
        if roles.is_ambiguous:
            result.action_summary = f"SUPPRESSED: {roles.ambiguity_reason}"
            result.priority_level = "AMBIGUOUS"
            self.reset_baselines()
            return result

        # Hand disappearance baseline resets
        if roles.right_just_lost:
            self.right_pinch_was_active = False
        if roles.left_just_lost:
            self.last_scroll_y = None
            self.left_palm_start_time = None
            self.left_swipe_history.clear()

        # ----------------------------------------------------------------------
        # Priority 3: Two-Hand Screen Zoom (Takes priority over click & volume)
        # ----------------------------------------------------------------------
        # Zoom occurs when BOTH hands are visible and BOTH are in a pinch pose
        if roles.right_hand is not None and roles.left_hand is not None and mode != "PRESENTATION":
            r_dist = euclidean_distance(roles.right_hand.landmarks[4], roles.right_hand.landmarks[8])
            l_dist = euclidean_distance(roles.left_hand.landmarks[4], roles.left_hand.landmarks[8])

            if r_dist < config.PINCH_THRESHOLD * 1.3 and l_dist < config.PINCH_THRESHOLD * 1.3:
                # Both hands are in zoom-ready pinch pose
                pinch_center_r = roles.right_hand.landmarks[8]
                pinch_center_l = roles.left_hand.landmarks[8]
                current_zoom_dist = euclidean_distance(pinch_center_r, pinch_center_l)

                if not self.is_zooming:
                    if self.zoom_entry_start_time is None:
                        self.zoom_entry_start_time = current_time
                    elif (current_time - self.zoom_entry_start_time) >= 0.25:
                        # Enter zoom mode and capture baseline distance
                        self.is_zooming = True
                        self.zoom_baseline_distance = current_zoom_dist

                if self.is_zooming:
                    # Actively zooming
                    zoom_delta = current_zoom_dist - (self.zoom_baseline_distance or current_zoom_dist)
                    zoom_action = "ZOOM IN" if zoom_delta > 0.03 else ("ZOOM OUT" if zoom_delta < -0.03 else "ZOOM HOLD")

                    result.is_zoom_active = True
                    result.zoom_distance = current_zoom_dist
                    result.zoom_delta = zoom_delta
                    result.action_summary = f"TWO-HAND {zoom_action}"
                    result.priority_level = "ZOOM"

                    # Populate display poses
                    result.right = HandPose(name="PINCH (ZOOM)", action="ZOOM IN/OUT", pinch_distance=r_dist)
                    result.left = HandPose(name="PINCH (ZOOM)", action="ZOOM IN/OUT", pinch_distance=l_dist)
                    return result
            else:
                self.zoom_entry_start_time = None
                self.is_zooming = False
                self.zoom_baseline_distance = None
        else:
            self.zoom_entry_start_time = None
            self.is_zooming = False
            self.zoom_baseline_distance = None

        # ----------------------------------------------------------------------
        # Priority 4 & 5: Individual Hand Gestures (Right: Cursor/Click, Left: Scroll/Vol/Media/Swipe)
        # ----------------------------------------------------------------------
        active_actions: List[str] = []

        if roles.right_hand is not None:
            result.right = self._classify_right_hand(roles.right_hand, current_time)
            if result.right.action != "NONE":
                active_actions.append(f"Right: {result.right.action}")

        if roles.left_hand is not None:
            result.left = self._classify_left_hand(roles.left_hand, current_time, mode=mode)
            if result.left.action != "NONE":
                active_actions.append(f"Left: {result.left.action}")

        if active_actions:
            result.action_summary = " | ".join(active_actions)
            result.priority_level = "CONTINUOUS"
        else:
            result.action_summary = "IDLE (WAITING FOR GESTURE)"
            result.priority_level = "IDLE"

        return result

    def reset_baselines(self) -> None:
        """Resets all active tracking baselines and timers."""
        self.last_scroll_y = None
        self.left_palm_start_time = None
        self.left_swipe_history.clear()
        self.right_pinch_was_active = False
        self.zoom_entry_start_time = None
        self.is_zooming = False
        self.zoom_baseline_distance = None
