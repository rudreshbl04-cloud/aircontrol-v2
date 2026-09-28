"""
Physical Hand Roles and Stability Manager for AirControl V2.

Responsible for:
- Mapping MediaPipe handedness to physical LEFT and RIGHT hands
- Maintaining stable role assignment across frames using position continuity
- Handling brief hand occlusions / tracking loss without abrupt jumps
- Detecting hand crossings and role ambiguities
- Suppressing actions and emitting warnings when hand assignment is unclear
- Triggering clean state resets when a hand disappears
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import config
from src.hand_tracker import RawHandData


@dataclass
class AssignedHand:
    """Represents a validated hand assigned to a physical role (LEFT or RIGHT)."""
    role: str                     # "RIGHT" or "LEFT"
    landmarks: List[Tuple[float, float, float]]
    confidence: float             # Handedness confidence [0.0, 1.0]
    center: Tuple[float, float]   # Normalized (x, y) center of palm
    raw_label: str                # Raw MediaPipe label ("Left" or "Right")
    raw_index: int                # Index in the raw detected hands list for drawing
    is_active: bool = True        # Whether currently visible in latest frame


@dataclass
class RoleFrameResult:
    """Complete role assignment status for a single video frame."""
    right_hand: Optional[AssignedHand] = None
    left_hand: Optional[AssignedHand] = None
    is_ambiguous: bool = False
    ambiguity_reason: str = ""
    hands_detected_count: int = 0
    right_just_lost: bool = False
    left_just_lost: bool = False


class HandRoleManager:
    """
    Manages physical LEFT and RIGHT hand identity, continuity,
    and ambiguity detection across consecutive video frames.
    """

    def __init__(
        self,
        confidence_threshold: float = config.ROLE_CONFIDENCE_THRESHOLD,
        occlusion_max_frames: int = config.HAND_OCCLUSION_MAX_FRAMES,
        ambiguity_threshold: float = config.AMBIGUITY_DISTANCE_THRESHOLD,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.occlusion_max_frames = occlusion_max_frames
        self.ambiguity_threshold = ambiguity_threshold

        # Persistent state
        self.last_right_center: Optional[Tuple[float, float]] = None
        self.last_left_center: Optional[Tuple[float, float]] = None

        self.right_missing_frames: int = 999
        self.left_missing_frames: int = 999

        # Flags indicating if hand was actively tracked in previous frame
        self.was_right_active: bool = False
        self.was_left_active: bool = False

    def _map_raw_label_to_physical(self, raw_label: str) -> str:
        """
        Maps MediaPipe's raw label to the user's physical hand.

        MediaPipe official convention:
        - When the image is mirrored (selfie mode, config.FLIP_CAMERA=True),
          MediaPipe's output 'Right' corresponds to the physical RIGHT hand,
          and 'Left' corresponds to the physical LEFT hand.
        - If not mirrored, the mapping is inverted.
        """
        if config.FLIP_CAMERA:
            return "RIGHT" if raw_label == "Right" else "LEFT"
        else:
            return "LEFT" if raw_label == "Right" else "RIGHT"

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculates 2D Euclidean distance between two normalized points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, raw_hands: List[RawHandData]) -> RoleFrameResult:
        """
        Updates role tracking with newly detected hands from the current frame.

        Args:
            raw_hands: List of 0, 1, or 2 RawHandData items from HandTracker.

        Returns:
            RoleFrameResult containing assigned hands and ambiguity flags.
        """
        result = RoleFrameResult(hands_detected_count=len(raw_hands))

        # ----------------------------------------------------------------------
        # Case 0: No hands detected in frame
        # ----------------------------------------------------------------------
        if not raw_hands:
            self.right_missing_frames += 1
            self.left_missing_frames += 1

            if self.was_right_active and self.right_missing_frames > self.occlusion_max_frames:
                result.right_just_lost = True
                self.was_right_active = False
                self.last_right_center = None

            if self.was_left_active and self.left_missing_frames > self.occlusion_max_frames:
                result.left_just_lost = True
                self.was_left_active = False
                self.last_left_center = None

            return result

        # ----------------------------------------------------------------------
        # Case 1: Ambiguity Check for Two Hands (Overlapping or Collision)
        # ----------------------------------------------------------------------
        if len(raw_hands) == 2:
            hand_dist = self._distance(raw_hands[0].center, raw_hands[1].center)
            if hand_dist < self.ambiguity_threshold:
                result.is_ambiguous = True
                result.ambiguity_reason = "Hand assignment unclear: hands overlapping"
                return result

        # Map each detected hand to candidate physical roles
        candidates: List[Tuple[str, RawHandData, int]] = []
        for idx, rh in enumerate(raw_hands):
            role = self._map_raw_label_to_physical(rh.raw_label)
            candidates.append((role, rh, idx))

        # ----------------------------------------------------------------------
        # Case 2: Exactly One Hand Detected
        # ----------------------------------------------------------------------
        if len(candidates) == 1:
            role, rh, idx = candidates[0]

            # Determine physical role with robust spatial & continuity heuristics
            # Respect raw MediaPipe classification when score meets confidence threshold
            assigned_role = role
            
            if rh.score < self.confidence_threshold:
                if self.last_right_center and self.last_left_center:
                    d_right = self._distance(rh.center, self.last_right_center)
                    d_left = self._distance(rh.center, self.last_left_center)
                    if min(d_right, d_left) < 0.35:
                        assigned_role = "RIGHT" if d_right < d_left else "LEFT"
                elif self.last_right_center and self._distance(rh.center, self.last_right_center) < 0.30:
                    assigned_role = "RIGHT"
                elif self.last_left_center and self._distance(rh.center, self.last_left_center) < 0.30:
                    assigned_role = "LEFT"
                else:
                    # Spatial screen hemisphere heuristic for low confidence single hand
                    if config.FLIP_CAMERA:
                        assigned_role = "RIGHT" if rh.center[0] >= 0.45 else "LEFT"
                    else:
                        assigned_role = "LEFT" if rh.center[0] >= 0.45 else "RIGHT"

            assigned = AssignedHand(
                role=assigned_role,
                landmarks=rh.landmarks,
                confidence=rh.score,
                center=rh.center,
                raw_label=rh.raw_label,
                raw_index=idx,
                is_active=True,
            )

            if assigned_role == "RIGHT":
                result.right_hand = assigned
                self.last_right_center = rh.center
                self.right_missing_frames = 0
                self.was_right_active = True

                self.left_missing_frames += 1
                if self.was_left_active and self.left_missing_frames > self.occlusion_max_frames:
                    result.left_just_lost = True
                    self.was_left_active = False
                    self.last_left_center = None
            else:
                result.left_hand = assigned
                self.last_left_center = rh.center
                self.left_missing_frames = 0
                self.was_left_active = True

                self.right_missing_frames += 1
                if self.was_right_active and self.right_missing_frames > self.occlusion_max_frames:
                    result.right_just_lost = True
                    self.was_right_active = False
                    self.last_right_center = None

            return result

        # ----------------------------------------------------------------------
        # Case 3: Two Hands Detected
        # ----------------------------------------------------------------------
        hand_a_role, hand_a_data, idx_a = candidates[0]
        hand_b_role, hand_b_data, idx_b = candidates[1]

        # Check if MediaPipe assigned distinct roles (one RIGHT, one LEFT)
        if hand_a_role != hand_b_role:
            # Clear distinct assignment
            right_cand = (hand_a_data, idx_a) if hand_a_role == "RIGHT" else (hand_b_data, idx_b)
            left_cand = (hand_b_data, idx_b) if hand_a_role == "RIGHT" else (hand_a_data, idx_a)

            result.right_hand = AssignedHand(
                role="RIGHT",
                landmarks=right_cand[0].landmarks,
                confidence=right_cand[0].score,
                center=right_cand[0].center,
                raw_label=right_cand[0].raw_label,
                raw_index=right_cand[1],
            )
            result.left_hand = AssignedHand(
                role="LEFT",
                landmarks=left_cand[0].landmarks,
                confidence=left_cand[0].score,
                center=left_cand[0].center,
                raw_label=left_cand[0].raw_label,
                raw_index=left_cand[1],
            )

            self.last_right_center = right_cand[0].center
            self.last_left_center = left_cand[0].center
            self.right_missing_frames = 0
            self.left_missing_frames = 0
            self.was_right_active = True
            self.was_left_active = True
            return result

        # Both hands were labeled with the same role by MediaPipe (Duplicate Conflict)
        # Attempt resolution via position continuity or horizontal spatial position
        if self.last_right_center and self.last_left_center:
            # Continuity match: assign based on proximity to previous locations
            d_a_r = self._distance(hand_a_data.center, self.last_right_center)
            d_a_l = self._distance(hand_a_data.center, self.last_left_center)
            d_b_r = self._distance(hand_b_data.center, self.last_right_center)
            d_b_l = self._distance(hand_b_data.center, self.last_left_center)

            if (d_a_r + d_b_l) < (d_a_l + d_b_r):
                assigned_right, assigned_left = (hand_a_data, idx_a), (hand_b_data, idx_b)
            else:
                assigned_right, assigned_left = (hand_b_data, idx_b), (hand_a_data, idx_a)
        else:
            # Spatial separation resolution:
            # In FLIP_CAMERA (mirrored selfie), hand with larger X is the physical RIGHT hand
            if config.FLIP_CAMERA:
                if hand_a_data.center[0] >= hand_b_data.center[0]:
                    assigned_right, assigned_left = (hand_a_data, idx_a), (hand_b_data, idx_b)
                else:
                    assigned_right, assigned_left = (hand_b_data, idx_b), (hand_a_data, idx_a)
            else:
                if hand_a_data.center[0] <= hand_b_data.center[0]:
                    assigned_right, assigned_left = (hand_a_data, idx_a), (hand_b_data, idx_b)
                else:
                    assigned_right, assigned_left = (hand_b_data, idx_b), (hand_a_data, idx_a)

        result.right_hand = AssignedHand(
            role="RIGHT",
            landmarks=assigned_right[0].landmarks,
            confidence=assigned_right[0].score,
            center=assigned_right[0].center,
            raw_label=assigned_right[0].raw_label,
            raw_index=assigned_right[1],
        )
        result.left_hand = AssignedHand(
            role="LEFT",
            landmarks=assigned_left[0].landmarks,
            confidence=assigned_left[0].score,
            center=assigned_left[0].center,
            raw_label=assigned_left[0].raw_label,
            raw_index=assigned_left[1],
        )
        self.last_right_center = assigned_right[0].center
        self.last_left_center = assigned_left[0].center
        self.right_missing_frames = 0
        self.left_missing_frames = 0
        self.was_right_active = True
        self.was_left_active = True
        return result

    def reset(self) -> None:
        """Resets all tracking histories and state."""
        self.last_right_center = None
        self.last_left_center = None
        self.right_missing_frames = 999
        self.left_missing_frames = 999
        self.was_right_active = False
        self.was_left_active = False
