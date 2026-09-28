"""
AirControl Gesture Recognition Module.

Responsible for:
- Finger state analysis (extended vs. folded)
- Pinch detection
- Fist detection
- Open-palm detection and 2-second safety hold timer
- Index and Index+Middle gesture classification
- Three fingers + thumb-index distance measurement for volume
- Horizontal swipe detection (Left/Right) with cooldowns and history tracking

No computer control (mouse/keyboard) is executed in this module.
"""

import math
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

import config


def euclidean_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Calculates 2D normalized Euclidean distance between two landmark points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class GestureRecognizer:
    """
    Analyzes normalized hand landmarks to classify gestures and manage
    safety state toggles (Open Palm hold) and swipe motion tracking.
    """

    def __init__(self) -> None:
        # History queue of (timestamp, x_position) for detecting horizontal swipes
        # Holds approximately 15-20 recent samples (~0.4-0.5s of tracking)
        self.position_history: deque = deque(maxlen=20)
        
        # Open-palm hold timer for safety ON/PAUSED toggle
        self.palm_hold_start_time: Optional[float] = None
        self.palm_toggle_triggered: bool = False
        self.is_paused: bool = False

        # Swipe state and cooldown tracker
        self.last_swipe_time: float = 0.0
        self.active_swipe_gesture: Optional[str] = None
        self.swipe_display_until: float = 0.0

    def get_finger_states(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, bool]:
        """
        Determines whether each finger is extended (True) or folded (False).

        MediaPipe Landmark Indices:
            Wrist: 0
            Thumb:  1 (CMC),  2 (MCP),  3 (IP),  4 (TIP)
            Index:  5 (MCP),  6 (PIP),  7 (DIP), 8 (TIP)
            Middle: 9 (MCP), 10 (PIP), 11 (DIP), 12 (TIP)
            Ring:  13 (MCP), 14 (PIP), 15 (DIP), 16 (TIP)
            Pinky: 17 (MCP), 18 (PIP), 19 (DIP), 20 (TIP)
        """
        wrist = landmarks[0]

        # For non-thumb fingers: extended if TIP is higher than PIP (smaller y)
        # and distance from wrist to TIP is greater than distance from wrist to PIP
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

        # For thumb: extended if tip is far from pinky base and wrist
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        pinky_mcp = landmarks[17]
        thumb_extended = (
            euclidean_distance(thumb_tip, pinky_mcp) > euclidean_distance(thumb_ip, pinky_mcp) * 1.15
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

    def detect_swipe(self, current_x: float, current_time: float) -> Optional[str]:
        """
        Evaluates recent horizontal position history to detect deliberate swipe motions.
        """
        self.position_history.append((current_time, current_x))

        # Check if swipe cooldown is currently active
        if current_time - self.last_swipe_time < config.SWIPE_COOLDOWN:
            return None

        # Look back ~0.25 to ~0.40 seconds in the position history
        old_x = None
        for t, x in self.position_history:
            if 0.20 <= (current_time - t) <= 0.45:
                old_x = x
                break

        if old_x is None:
            return None

        # Calculate horizontal displacement
        dx = current_x - old_x

        if abs(dx) >= config.SWIPE_THRESHOLD:
            self.last_swipe_time = current_time
            self.position_history.clear()
            # Positive dx means movement to the right, negative dx means to the left
            if dx > 0:
                self.active_swipe_gesture = "SWIPE RIGHT"
            else:
                self.active_swipe_gesture = "SWIPE LEFT"
            self.swipe_display_until = current_time + 0.60
            return self.active_swipe_gesture

        return None

    def recognize(
        self,
        landmarks: Optional[List[Tuple[float, float, float]]],
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Processes normalized landmarks and returns recognized gesture and diagnostic details.

        Returns a dictionary containing:
            - 'gesture': Current recognized gesture name
            - 'status': "ON" or "PAUSED"
            - 'palm_hold_progress': float 0.0 to 1.0 (for UI hold bar)
            - 'volume_pct': float 0.0 to 100.0 (if three-finger volume active)
            - 'pinch_distance': float
            - 'fingers': Dict of finger extended booleans
        """
        if current_time is None:
            current_time = time.perf_counter()

        # Handle persistent visual display of recently triggered swipe
        if current_time < self.swipe_display_until and self.active_swipe_gesture:
            display_swipe = self.active_swipe_gesture
        else:
            display_swipe = None
            self.active_swipe_gesture = None

        # If no hand is detected in the current frame
        if landmarks is None or len(landmarks) < 21:
            # Reset open-palm timer if hand leaves the frame
            self.palm_hold_start_time = None
            self.palm_toggle_triggered = False
            self.position_history.clear()
            return {
                "gesture": display_swipe if display_swipe else "NO HAND",
                "status": "PAUSED" if self.is_paused else "ON",
                "palm_hold_progress": 0.0,
                "volume_pct": None,
                "pinch_distance": None,
                "fingers": None,
            }

        fingers = self.get_finger_states(landmarks)
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        pinch_dist = euclidean_distance(thumb_tip, index_tip)

        # Palm center / wrist x-coordinate used for swipe motion
        hand_x = landmarks[9][0]  # Middle finger MCP represents hand center well
        detected_swipe = self.detect_swipe(hand_x, current_time)
        if detected_swipe:
            display_swipe = detected_swipe

        num_fingers_up = sum([fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]])
        all_five_up = fingers["thumb"] and num_fingers_up == 4

        # ----------------------------------------------------------------------
        # 1. GESTURE 7: OPEN PALM (SAFETY TOGGLE HELD FOR 2 SECONDS)
        # ----------------------------------------------------------------------
        palm_progress = 0.0
        # Consider open palm if all 4 main fingers are up and thumb is extended
        is_open_palm = (num_fingers_up == 4 and fingers["thumb"])

        if is_open_palm:
            if self.palm_hold_start_time is None:
                self.palm_hold_start_time = current_time
                self.palm_toggle_triggered = False

            elapsed = current_time - self.palm_hold_start_time
            palm_progress = min(1.0, elapsed / config.OPEN_PALM_HOLD_SECONDS)

            # Trigger toggle once hold duration is reached
            if elapsed >= config.OPEN_PALM_HOLD_SECONDS and not self.palm_toggle_triggered:
                self.is_paused = not self.is_paused
                self.palm_toggle_triggered = True
                print(f"[AirControl Safety] Open palm hold completed. State toggled to: {'PAUSED' if self.is_paused else 'ON'}")
        else:
            self.palm_hold_start_time = None
            self.palm_toggle_triggered = False
            palm_progress = 0.0

        current_status = "PAUSED" if self.is_paused else "ON"

        # If a swipe was just detected, prioritize its display
        if display_swipe:
            return {
                "gesture": display_swipe,
                "status": current_status,
                "palm_hold_progress": palm_progress,
                "volume_pct": None,
                "pinch_distance": pinch_dist,
                "fingers": fingers,
            }

        # ----------------------------------------------------------------------
        # 2. GESTURE 6: FIST (ALL FINGERS FOLDED INTO PALM)
        # ----------------------------------------------------------------------
        # When all four fingers (index, middle, ring, pinky) are folded into palm
        if num_fingers_up == 0 and not fingers["thumb"]:
            return {
                "gesture": "FIST",
                "status": current_status,
                "palm_hold_progress": 0.0,
                "volume_pct": None,
                "pinch_distance": pinch_dist,
                "fingers": fingers,
            }

        # ----------------------------------------------------------------------
        # 3. GESTURE 2: PINCH (INDEX + THUMB)
        # ----------------------------------------------------------------------
        # Requires pinch distance < threshold and index finger extended or semi-extended
        if pinch_dist < config.PINCH_THRESHOLD and num_fingers_up > 0:
            return {
                "gesture": "PINCH",
                "status": current_status,
                "palm_hold_progress": 0.0,
                "volume_pct": None,
                "pinch_distance": pinch_dist,
                "fingers": fingers,
            }

        # ----------------------------------------------------------------------
        # 4. GESTURE 7: OPEN PALM
        # ----------------------------------------------------------------------
        if is_open_palm:
            return {
                "gesture": "OPEN PALM",
                "status": current_status,
                "palm_hold_progress": palm_progress,
                "volume_pct": None,
                "pinch_distance": pinch_dist,
                "fingers": fingers,
            }

        # ----------------------------------------------------------------------
        # 5. GESTURE 4: THREE FINGERS UP + THUMB-INDEX DISTANCE (VOLUME)
        # ----------------------------------------------------------------------
        # Index, Middle, Ring extended, Pinky folded
        if fingers["index"] and fingers["middle"] and fingers["ring"] and not fingers["pinky"]:
            # Map distance to 0% - 100% volume
            clamped_dist = max(config.VOLUME_MIN_DISTANCE, min(pinch_dist, config.VOLUME_MAX_DISTANCE))
            vol_range = config.VOLUME_MAX_DISTANCE - config.VOLUME_MIN_DISTANCE
            vol_pct = ((clamped_dist - config.VOLUME_MIN_DISTANCE) / vol_range) * 100.0

            return {
                "gesture": "THREE FINGERS (VOLUME)",
                "status": current_status,
                "palm_hold_progress": 0.0,
                "volume_pct": vol_pct,
                "pinch_distance": pinch_dist,
                "fingers": fingers,
            }

        # ----------------------------------------------------------------------
        # 6. GESTURE 3: INDEX + MIDDLE FINGERS UP (SCROLL)
        # ----------------------------------------------------------------------
        if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            return {
                "gesture": "INDEX + MIDDLE",
                "status": current_status,
                "palm_hold_progress": 0.0,
                "volume_pct": None,
                "pinch_distance": pinch_dist,
                "fingers": fingers,
            }

        # ----------------------------------------------------------------------
        # 7. GESTURE 1: INDEX FINGER UP (MOUSE MOVE)
        # ----------------------------------------------------------------------
        if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            return {
                "gesture": "INDEX FINGER UP",
                "status": current_status,
                "palm_hold_progress": 0.0,
                "volume_pct": None,
                "pinch_distance": pinch_dist,
                "fingers": fingers,
            }

        # Default fallback if hand pose is unrecognized
        return {
            "gesture": "UNKNOWN GESTURE",
            "status": current_status,
            "palm_hold_progress": 0.0,
            "volume_pct": None,
            "pinch_distance": pinch_dist,
            "fingers": fingers,
        }
