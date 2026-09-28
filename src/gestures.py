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


def euclidean_distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Calculates 3D Euclidean distance between two landmark points."""
    z1 = p1[2] if len(p1) > 2 else 0.0
    z2 = p2[2] if len(p2) > 2 else 0.0
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (z1 - z2)**2)


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
        Robust across all hand orientations (upright, forward tilt, or sideways).
        """
        wrist = landmarks[0]

        # Fingers: Index (8), Middle (12), Ring (16), Pinky (20)
        # Extended if TIP is further from wrist and MCP than PIP joint
        def is_finger_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            d_wrist_tip = euclidean_distance(wrist, tip)
            d_wrist_pip = euclidean_distance(wrist, pip)
            d_mcp_tip = euclidean_distance(mcp, tip)
            d_mcp_pip = euclidean_distance(mcp, pip)
            d_mcp_tip_3d = euclidean_distance_3d(mcp, tip)
            d_mcp_pip_3d = euclidean_distance_3d(mcp, pip)

            is_dist_extended = (d_wrist_tip > d_wrist_pip * 1.04) and (d_mcp_tip > d_mcp_pip * 1.04)
            is_3d_extended = (d_mcp_tip_3d > d_mcp_pip_3d * 1.08)
            is_y_extended = (tip[1] < pip[1]) and (d_wrist_tip > d_wrist_pip)
            return is_dist_extended or is_3d_extended or is_y_extended

        index_extended = is_finger_extended(8, 6, 5)
        middle_extended = is_finger_extended(12, 10, 9)
        ring_extended = is_finger_extended(16, 14, 13)
        pinky_extended = is_finger_extended(20, 18, 17)

        # Thumb: extended if tip is far from pinky base and further from wrist than IP joint
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        pinky_mcp = landmarks[17]
        thumb_extended = (
            euclidean_distance(thumb_tip, pinky_mcp) > euclidean_distance(thumb_ip, pinky_mcp) * 1.05
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
        middle_tip = landmarks[12] if len(landmarks) > 12 else landmarks[8]
        ring_tip = landmarks[16] if len(landmarks) > 16 else landmarks[8]
        pinky_tip = landmarks[20] if len(landmarks) > 20 else landmarks[8]

        pinch_dist = euclidean_distance(thumb_tip, index_tip)
        middle_pinch_dist = euclidean_distance(thumb_tip, middle_tip)

        # Palm scale for distance-invariant pinch detection
        palm_scale = max(0.08, euclidean_distance(landmarks[0], landmarks[9]))
        is_index_pinch = (pinch_dist < config.PINCH_THRESHOLD) or (pinch_dist / palm_scale < 0.30)
        is_middle_pinch = (middle_pinch_dist < 0.038) or (middle_pinch_dist / palm_scale < 0.22)

        num_up = sum([fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]])

        # 1. Left Click & Drag (Pinch between Thumb and Index)
        if is_index_pinch and not (num_up == 0 and not fingers["thumb"] and pinch_dist > 0.04):
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

        # 2. Right Click (Pinch between Thumb and Middle Finger)
        if (
            getattr(config, "RIGHT_CLICK_ENABLED", True)
            and is_middle_pinch
            and pinch_dist >= middle_pinch_dist * 1.3
        ):
            return HandPose(
                name="RIGHT PINCH",
                action="RIGHT CLICK",
                pinch_distance=middle_pinch_dist,
                fingers=fingers,
            )

        # 3. Right Cursor Movement (Index Finger Up, other fingers folded)
        # Orientation-invariant: compares index extension to other fingers regardless of hand tilt
        d_wrist_index = euclidean_distance(landmarks[0], index_tip)
        d_wrist_middle = euclidean_distance(landmarks[0], middle_tip)
        d_wrist_ring = euclidean_distance(landmarks[0], ring_tip)
        d_wrist_pinky = euclidean_distance(landmarks[0], pinky_tip)

        is_index_longest = (
            d_wrist_index > d_wrist_middle * 1.08
            and d_wrist_index > d_wrist_ring * 1.08
            and d_wrist_index > d_wrist_pinky * 1.08
        )
        is_other_folded = (not fingers["middle"] and not fingers["ring"] and not fingers["pinky"])
        is_upright_pointing = (landmarks[8][1] < landmarks[12][1] and landmarks[8][1] < landmarks[16][1])

        if fingers["index"] and (is_index_longest or is_other_folded or is_upright_pointing):
            return HandPose(
                name="INDEX UP",
                action="MOVE CURSOR",
                pinch_distance=pinch_dist,
                fingers=fingers,
            )

        # 4. Idle / Rest pose
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
        d_wrist_middle = euclidean_distance(landmarks[0], landmarks[12])
        d_wrist_ring = euclidean_distance(landmarks[0], landmarks[16])
        is_two_extended = fingers["index"] and fingers["middle"]
        is_ring_pinky_folded = (
            (not fingers["ring"] and not fingers["pinky"])
            or (d_wrist_middle > d_wrist_ring * 1.06)
            or (landmarks[8][1] < landmarks[16][1] and landmarks[12][1] < landmarks[16][1])
        )

        if is_two_extended and is_ring_pinky_folded:
            scroll_dy = 0.0
            if self.last_scroll_y is not None:
                scroll_dy = self.last_scroll_y - hand_center_y  # Moving up is positive dy

            self.last_scroll_y = hand_center_y
            action = "SCROLL UP" if scroll_dy > 0.003 else ("SCROLL DOWN" if scroll_dy < -0.003 else "SCROLL IDLE")
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
        is_three_extended = fingers["index"] and fingers["middle"] and fingers["ring"]
        is_pinky_folded = (
            (not fingers["pinky"])
            or (euclidean_distance(landmarks[0], landmarks[16]) > euclidean_distance(landmarks[0], landmarks[20]) * 1.05)
            or (landmarks[16][1] < landmarks[20][1] + 0.03)
        )

        if is_three_extended and is_pinky_folded:
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
        Seamlessly supports simultaneous two-hand control and universal single-hand operation.
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
        # Priority 4 & 5: Individual Hand Gestures (Dual-Hand & Universal Single-Hand)
        # ----------------------------------------------------------------------
        active_actions: List[str] = []

        # Dual Hand Mode: Both hands present
        if roles.right_hand is not None and roles.left_hand is not None:
            result.right = self._classify_right_hand(roles.right_hand, current_time)
            if result.right.action != "NONE":
                active_actions.append(f"Right: {result.right.action}")

            result.left = self._classify_left_hand(roles.left_hand, current_time, mode=mode)
            if result.left.action != "NONE":
                active_actions.append(f"Left: {result.left.action}")

        # Single Hand Mode: Exactly one hand present (Full Universal Control)
        elif roles.right_hand is not None:
            r_pose = self._classify_right_hand(roles.right_hand, current_time)
            if r_pose.name in ("INDEX UP", "PINCH", "RIGHT PINCH"):
                result.right = r_pose
                active_actions.append(f"Right: {r_pose.action}")
            else:
                l_pose = self._classify_left_hand(roles.right_hand, current_time, mode=mode)
                if l_pose.name in ("INDEX + MIDDLE (PEACE)", "THREE FINGERS", "OPEN PALM", "SWIPE"):
                    result.left = l_pose
                    active_actions.append(f"Left: {l_pose.action}")
                else:
                    result.right = r_pose

        elif roles.left_hand is not None:
            # Single hand assigned to Left: check cursor/click gestures first so navigation works
            r_pose = self._classify_right_hand(roles.left_hand, current_time)
            if r_pose.name in ("INDEX UP", "PINCH", "RIGHT PINCH"):
                result.right = r_pose
                active_actions.append(f"Right: {r_pose.action}")
            else:
                l_pose = self._classify_left_hand(roles.left_hand, current_time, mode=mode)
                if l_pose.action != "NONE":
                    active_actions.append(f"Left: {l_pose.action}")
                result.left = l_pose

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


class GestureRecognizer:
    """
    Single-hand gesture recognizer maintaining backward compatibility with V1 tools/tests.
    """

    def __init__(self) -> None:
        self.status: str = "ON"
        self.palm_hold_start: Optional[float] = None
        self.swipe_history: List[Tuple[float, float]] = []
        self._recognizer = TwoHandGestureRecognizer()

    def recognize(
        self,
        hand_landmarks: Optional[List[Tuple[float, float, float]]],
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Recognizes gestures on a single hand landmark list."""
        if current_time is None:
            current_time = time.perf_counter()

        if hand_landmarks is None or len(hand_landmarks) < 21:
            self.palm_hold_start = None
            return {
                "gesture": "NO HAND",
                "status": self.status,
                "palm_hold_progress": 0.0,
                "pinch_distance": 1.0,
                "volume_pct": None,
            }

        fingers = self._recognizer.get_finger_states(hand_landmarks)
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        pinch_dist = euclidean_distance(thumb_tip, index_tip)

        num_up = sum([fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]])
        is_open_palm = (num_up == 4 and fingers["thumb"])

        # Check swipe
        hand_x = hand_landmarks[0][0]
        self.swipe_history.append((current_time, hand_x))
        self.swipe_history = [(t, x) for (t, x) in self.swipe_history if current_time - t <= 0.60]
        swipe_gesture = None
        for t, x in self.swipe_history:
            if 0.15 <= (current_time - t) <= 0.45:
                dx = hand_x - x
                if abs(dx) >= config.SWIPE_THRESHOLD:
                    swipe_gesture = "SWIPE RIGHT" if dx > 0 else "SWIPE LEFT"
                    break

        if swipe_gesture:
            return {
                "gesture": swipe_gesture,
                "status": self.status,
                "palm_hold_progress": 0.0,
                "pinch_distance": pinch_dist,
                "volume_pct": None,
            }

        # Open palm 2-second hold
        progress = 0.0
        if is_open_palm:
            if self.palm_hold_start is None:
                self.palm_hold_start = current_time
            elapsed = current_time - self.palm_hold_start
            progress = min(1.0, elapsed / config.OPEN_PALM_HOLD_SECONDS)
            if elapsed >= config.OPEN_PALM_HOLD_SECONDS:
                self.status = "PAUSED" if self.status == "ON" else "ON"
                self.palm_hold_start = None
                progress = 1.0
            return {
                "gesture": "OPEN PALM",
                "status": self.status,
                "palm_hold_progress": progress,
                "pinch_distance": pinch_dist,
                "volume_pct": None,
            }
        else:
            self.palm_hold_start = None

        if num_up == 0 and not fingers["thumb"]:
            return {
                "gesture": "FIST",
                "status": self.status,
                "palm_hold_progress": 0.0,
                "pinch_distance": pinch_dist,
                "volume_pct": None,
            }

        palm_scale = max(0.08, euclidean_distance(hand_landmarks[0], hand_landmarks[9]))
        is_pinch = (pinch_dist < config.PINCH_THRESHOLD) or (pinch_dist / palm_scale < 0.35)
        if is_pinch and not (num_up == 0 and not fingers["thumb"] and pinch_dist > 0.04):
            return {
                "gesture": "PINCH",
                "status": self.status,
                "palm_hold_progress": 0.0,
                "pinch_distance": pinch_dist,
                "volume_pct": None,
            }

        d_wrist_index = euclidean_distance(hand_landmarks[0], index_tip)
        d_wrist_middle = euclidean_distance(hand_landmarks[0], hand_landmarks[12])
        d_wrist_ring = euclidean_distance(hand_landmarks[0], hand_landmarks[16])
        d_wrist_pinky = euclidean_distance(hand_landmarks[0], hand_landmarks[20])

        is_index_longest = (
            d_wrist_index > d_wrist_middle * 1.08
            and d_wrist_index > d_wrist_ring * 1.08
            and d_wrist_index > d_wrist_pinky * 1.08
        )
        if fingers["index"] and (is_index_longest or (not fingers["middle"] and not fingers["ring"] and not fingers["pinky"])):
            return {
                "gesture": "INDEX FINGER UP",
                "status": self.status,
                "palm_hold_progress": 0.0,
                "pinch_distance": pinch_dist,
                "volume_pct": None,
            }

        if fingers["index"] and fingers["middle"] and not (fingers["ring"] and fingers["pinky"]):
            if (not fingers["ring"] and not fingers["pinky"]) or (d_wrist_middle > d_wrist_ring * 1.06):
                return {
                    "gesture": "INDEX + MIDDLE",
                    "status": self.status,
                    "palm_hold_progress": 0.0,
                    "pinch_distance": pinch_dist,
                    "volume_pct": None,
                }

        if fingers["index"] and fingers["middle"] and fingers["ring"] and not fingers["pinky"]:
            clamped_dist = max(config.VOLUME_MIN_DISTANCE, min(pinch_dist, config.VOLUME_MAX_DISTANCE))
            vol_range = config.VOLUME_MAX_DISTANCE - config.VOLUME_MIN_DISTANCE
            vol_pct = ((clamped_dist - config.VOLUME_MIN_DISTANCE) / vol_range) * 100.0
            return {
                "gesture": "THREE FINGERS (VOLUME)",
                "status": self.status,
                "palm_hold_progress": 0.0,
                "pinch_distance": pinch_dist,
                "volume_pct": vol_pct,
            }

        return {
            "gesture": "IDLE",
            "status": self.status,
            "palm_hold_progress": 0.0,
            "pinch_distance": pinch_dist,
            "volume_pct": None,
        }
