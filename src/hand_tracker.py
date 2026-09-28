"""
Hand Tracking Module for AirControl.

Encapsulates MediaPipe Hands initialization, landmark extraction,
and visual rendering of tracked landmarks on video frames.
AirControl V1 tracks strictly ONE hand to ensure deterministic behavior.
"""

import os
import sys
import contextlib
from typing import Optional, List, Tuple, Any
import cv2

# Suppress MediaPipe/TF low-level C++ logs before importing mediapipe
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import os
import sys
from typing import Optional, List, Tuple, Any
import cv2
import mediapipe as mp

import config


class HandTracker:
    """Wrapper around MediaPipe Hands for real-time hand landmark tracking."""

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MEDIAPIPE_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MEDIAPIPE_TRACKING_CONFIDENCE,
    ) -> None:
        """Initialize the MediaPipe Hands solution with configured confidence thresholds."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.last_results: Optional[Any] = None

    def find_hand_landmarks(self, frame_bgr: Any) -> Optional[List[Tuple[float, float, float]]]:
        """
        Processes a BGR image frame to locate hand landmarks.

        Args:
            frame_bgr: OpenCV image frame in BGR color space.

        Returns:
            A list of 21 (x, y, z) normalized landmark coordinates for the primary hand,
            or None if no hand is detected in the frame.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        self.last_results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not self.last_results or not self.last_results.multi_hand_landmarks:
            return None

        # Deterministically select only the FIRST detected hand
        primary_hand = self.last_results.multi_hand_landmarks[0]
        
        landmarks: List[Tuple[float, float, float]] = []
        for lm in primary_hand.landmark:
            landmarks.append((lm.x, lm.y, lm.z))

        return landmarks

    def draw_landmarks_on_frame(self, frame_bgr: Any) -> None:
        """Draws skeleton connections and landmark points directly onto the OpenCV frame."""
        if not self.last_results or not self.last_results.multi_hand_landmarks:
            return

        primary_hand = self.last_results.multi_hand_landmarks[0]
        self.mp_drawing.draw_landmarks(
            frame_bgr,
            primary_hand,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        """Releases MediaPipe pipeline resources cleanly."""
        if hasattr(self, "hands") and self.hands is not None:
            self.hands.close()
