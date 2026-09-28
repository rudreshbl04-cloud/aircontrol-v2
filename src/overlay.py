"""
HUD and Camera Overlay Module for AirControl V2.

Responsible for rendering:
- Left and Right physical hand cards with confidence scores
- Hand landmarks skeleton with distinct colors for Left vs Right
- ON / PAUSED safety status
- Mode indicator (DESKTOP, MEDIA, PRESENTATION)
- FPS counter
- Ambiguity warning banner ("Hand assignment unclear")
- Active camera region boundaries
- Low-light warning hints
"""

from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

import config
from src.hand_roles import RoleFrameResult, AssignedHand

# Color constants (BGR)
COLOR_RIGHT_HAND = (255, 200, 0)     # Vibrant Cyan-Blue for Right Hand
COLOR_LEFT_HAND = (0, 165, 255)      # Warm Amber-Orange for Left Hand
COLOR_AMBIGUOUS = (0, 100, 255)      # Red-Orange for Warnings
COLOR_PAUSED = (0, 140, 255)         # Orange for Paused state
COLOR_ACTIVE = (0, 255, 0)           # Bright Green for Active ON state
COLOR_HUD_BG = (22, 22, 22)          # Dark Slate Gray HUD background
COLOR_HUD_BORDER = (60, 60, 60)      # Subtle border


class CameraOverlay:
    """Manages all graphical HUD overlays on camera preview frames."""

    def __init__(self) -> None:
        pass

    def draw_active_region(self, frame: np.ndarray) -> None:
        """Draws the configured active region boundary for cursor navigation."""
        h, w, _ = frame.shape
        x_min = int(config.ACTIVE_AREA_LEFT * w)
        x_max = int(config.ACTIVE_AREA_RIGHT * w)
        y_min = int(config.ACTIVE_AREA_TOP * h)
        y_max = int(config.ACTIVE_AREA_BOTTOM * h)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (70, 70, 70), 1)
        cv2.putText(
            frame,
            "Active Cursor Region",
            (x_min + 6, y_min + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (110, 110, 110),
            1,
            cv2.LINE_AA,
        )

    def draw_hand_badge_card(
        self,
        frame: np.ndarray,
        hand: Optional[AssignedHand],
        role_name: str,
        card_origin: Tuple[int, int],
        card_color: Tuple[int, int, int],
    ) -> None:
        """Renders an individual HUD card for one physical hand."""
        x, y = card_origin
        card_w, card_h = 170, 62

        # Background card
        cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), COLOR_HUD_BG, -1)
        cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), card_color if hand else (50, 50, 50), 1)

        # Header: Role Name
        role_label = f"[{role_name} HAND]"
        cv2.putText(
            frame,
            role_label,
            (x + 10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            card_color if hand else (110, 110, 110),
            1,
            cv2.LINE_AA,
        )

        # Status & Confidence
        if hand:
            conf_pct = int(hand.confidence * 100)
            status_text = f"Active ({conf_pct}%)"
            sub_text = "Tracking 21 pts"
            text_color = (220, 220, 220)
        else:
            status_text = "Searching..."
            sub_text = "Not visible"
            text_color = (110, 110, 110)

        cv2.putText(
            frame,
            status_text,
            (x + 10, y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            text_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            sub_text,
            (x + 10, y + 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

    def render(
        self,
        frame: np.ndarray,
        roles: RoleFrameResult,
        fps: float,
        is_paused: bool = False,
        mode: str = "DESKTOP",
        is_low_light: bool = False,
    ) -> None:
        """
        Renders complete V2 HUD overlay on the given frame (modified in-place).
        """
        h, w, _ = frame.shape

        # 1. Active cursor region boundary
        self.draw_active_region(frame)

        # 2. Main System Header Card (Top Center / Left)
        header_w = 350
        header_h = 75
        cv2.rectangle(frame, (10, 10), (10 + header_w, 10 + header_h), COLOR_HUD_BG, -1)
        cv2.rectangle(frame, (10, 10), (10 + header_w, 10 + header_h), COLOR_HUD_BORDER, 1)

        # Application Title
        cv2.putText(
            frame,
            "AirControl V2 — Two-Hand Tracking",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        # Safety Status & Mode
        status_label = "PAUSED" if is_paused else "ON"
        status_color = COLOR_PAUSED if is_paused else COLOR_ACTIVE
        cv2.putText(
            frame,
            f"Status: {status_label}",
            (20, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            status_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Mode: {mode}",
            (175, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (285, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # 3. Hand Role Cards (Left Hand & Right Hand)
        card_y = 95
        # Left Hand card (Orange)
        self.draw_hand_badge_card(
            frame=frame,
            hand=roles.left_hand,
            role_name="LEFT",
            card_origin=(10, card_y),
            card_color=COLOR_LEFT_HAND,
        )
        # Right Hand card (Cyan)
        self.draw_hand_badge_card(
            frame=frame,
            hand=roles.right_hand,
            role_name="RIGHT",
            card_origin=(190, card_y),
            card_color=COLOR_RIGHT_HAND,
        )

        # 4. Ambiguity Warning Banner
        if roles.is_ambiguous:
            warning_text = f"WARNING: {roles.ambiguity_reason}"
            cv2.rectangle(frame, (10, card_y + 70), (10 + header_w, card_y + 102), (0, 0, 140), -1)
            cv2.rectangle(frame, (10, card_y + 70), (10 + header_w, card_y + 102), (0, 0, 255), 1)
            cv2.putText(
                frame,
                warning_text,
                (18, card_y + 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 5. Low-Light Warning (Bottom Left)
        if is_low_light:
            cv2.rectangle(frame, (10, h - 55), (320, h - 25), (0, 0, 140), -1)
            cv2.putText(
                frame,
                "Notice: Low lighting detected",
                (18, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 6. Bottom Navigation Bar
        cv2.putText(
            frame,
            "Keys: F8 = Pause/Resume | M = Mode | Q = Quit",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
