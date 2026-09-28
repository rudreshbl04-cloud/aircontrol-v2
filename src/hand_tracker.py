"""
Two-Hand Tracking Module for AirControl V2.

Encapsulates MediaPipe Hands initialization configured for up to TWO hands,
extracting 21 3D landmarks per hand along with MediaPipe handedness labels
and classification confidence scores.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any
import cv2
import mediapipe as mp

import config


@dataclass
class RawHandData:
    """Raw hand tracking data extracted directly from MediaPipe."""
    landmarks: List[Tuple[float, float, float]]
    raw_label: str       # "Left" or "Right" as output by MediaPipe
    score: float          # Handedness confidence score [0.0, 1.0]
    center: Tuple[float, float]  # Normalized (x, y) center (approx. MCP cluster)


class HandTracker:
    """Wrapper around MediaPipe Hands for real-time two-hand landmark tracking."""

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MEDIAPIPE_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MEDIAPIPE_TRACKING_CONFIDENCE,
        model_complexity: int = config.MODEL_COMPLEXITY,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.last_results: Optional[Any] = None

    def set_model_complexity(self, complexity: int) -> None:
        """
        Dynamically switches model complexity (0 = Lite/Fast, 1 = Full/High Precision)
        and re-trains/re-initializes MediaPipe pipeline cleanly.
        """
        if complexity not in (0, 1) or complexity == self.model_complexity:
            return
        self.model_complexity = complexity
        config.MODEL_COMPLEXITY = complexity
        if hasattr(self, "hands") and self.hands is not None:
            self.hands.close()
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def toggle_model_complexity(self) -> int:
        """Toggles between Lite (0) and Full (1) model complexity."""
        new_complexity = 0 if self.model_complexity == 1 else 1
        self.set_model_complexity(new_complexity)
        return self.model_complexity

    def process_frame(self, frame_bgr: Any) -> List[RawHandData]:
        """
        Processes a BGR image frame to locate up to two hands.
        Uses fast downscaled inference for 25+ FPS while preserving normalized coordinates.

        Args:
            frame_bgr: OpenCV image frame in BGR color space.

        Returns:
            A list of RawHandData instances (0, 1, or 2 hands).
        """
        # Downscale for high-speed real-time inference (normalized coordinates remain identical)
        inf_frame = cv2.resize(frame_bgr, (config.INFERENCE_WIDTH, config.INFERENCE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(inf_frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        self.last_results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        detected_hands: List[RawHandData] = []

        if not self.last_results or not self.last_results.multi_hand_landmarks:
            return detected_hands

        # Extract landmarks and handedness for each detected hand
        num_hands = len(self.last_results.multi_hand_landmarks)
        for i in range(num_hands):
            hand_landmarks = self.last_results.multi_hand_landmarks[i]
            
            # Extract 21 landmark tuples
            landmarks: List[Tuple[float, float, float]] = []
            for lm in hand_landmarks.landmark:
                landmarks.append((lm.x, lm.y, lm.z))

            # Retrieve handedness classification if available
            raw_label = "Unknown"
            score = 0.5
            if self.last_results.multi_handedness and i < len(self.last_results.multi_handedness):
                classification = self.last_results.multi_handedness[i].classification[0]
                raw_label = classification.label  # "Left" or "Right"
                score = float(classification.score)

            # Calculate palm center (approximate using middle finger MCP landmark 9 and wrist landmark 0)
            center_x = (landmarks[0][0] + landmarks[9][0]) / 2.0
            center_y = (landmarks[0][1] + landmarks[9][1]) / 2.0

            detected_hands.append(
                RawHandData(
                    landmarks=landmarks,
                    raw_label=raw_label,
                    score=score,
                    center=(center_x, center_y),
                )
            )

        return detected_hands

    def draw_landmarks_for_hand(
        self,
        frame: Any,
        hand_idx: int,
        color_bgr: Tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """
        Draws landmark points and skeleton lines for a specific hand index.
        """
        if (
            not self.last_results
            or not self.last_results.multi_hand_landmarks
            or hand_idx >= len(self.last_results.multi_hand_landmarks)
        ):
            return

        hand_landmarks = self.last_results.multi_hand_landmarks[hand_idx]
        
        # Custom drawing spec with assigned role color
        landmark_spec = self.mp_drawing.DrawingSpec(color=color_bgr, thickness=2, circle_radius=3)
        connection_spec = self.mp_drawing.DrawingSpec(color=color_bgr, thickness=2)

        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            landmark_spec,
            connection_spec,
        )

    def close(self) -> None:
        """Releases MediaPipe pipeline resources cleanly."""
        if hasattr(self, "hands") and self.hands is not None:
            self.hands.close()
