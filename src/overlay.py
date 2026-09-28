"""
HUD and Camera Overlay Module for AirControl V2.

Renders:
- Real-time action indicator banner showing what computer control is active
- Left and Right physical hand cards with detected pose, confidence %, and action badges
- On-screen quick gesture cheat-sheet guide (what each gesture does)
- Zoom in/out delta indicator
- Volume level gauge
- Mode indicator (DESKTOP, MEDIA, PRESENTATION)
- Safety state (ON / PAUSED)
- Ambiguity warning banner ("Hand assignment unclear")
- Active cursor boundary
"""

from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

import config
from src.hand_roles import RoleFrameResult, AssignedHand
from src.gestures import TwoHandGestureResult, HandPose

# High-contrast BGR colors
COLOR_RIGHT_HAND = (255, 200, 0)     # Vibrant Cyan-Blue for Right Hand
COLOR_LEFT_HAND = (0, 165, 255)      # Warm Amber-Orange for Left Hand
COLOR_ZOOM = (255, 0, 255)           # Magenta for Two-Hand Zoom
COLOR_ACTION_BG = (35, 35, 35)       # Card background
COLOR_ACTION_BORDER = (80, 80, 80)   # Card border
COLOR_ACTIVE_GREEN = (0, 255, 0)     # Active green
COLOR_PAUSED = (0, 140, 255)         # Orange for pause state


class CameraOverlay:
    """Manages graphical HUD overlays and gesture guidance on camera preview frames."""

    def __init__(self) -> None:
        pass

    def draw_active_region(self, frame: np.ndarray) -> None:
        """Draws the active camera region boundary for cursor navigation."""
        h, w, _ = frame.shape
        x_min = int(config.ACTIVE_AREA_LEFT * w)
        x_max = int(config.ACTIVE_AREA_RIGHT * w)
        y_min = int(config.ACTIVE_AREA_TOP * h)
        y_max = int(config.ACTIVE_AREA_BOTTOM * h)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (60, 60, 60), 1)
        cv2.putText(
            frame,
            "Active Cursor Zone",
            (x_min + 6, y_min + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

    def draw_gesture_cheatsheet(self, frame: np.ndarray) -> None:
        """Renders an on-screen visual cheatsheet guide so users always know which gesture to make."""
        h, w, _ = frame.shape
        guide_w = 245
        guide_h = 180
        gx = w - guide_w - 15
        gy = 10

        # Background container
        cv2.rectangle(frame, (gx, gy), (gx + guide_w, gy + guide_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (gx, gy), (gx + guide_w, gy + guide_h), (60, 60, 60), 1)

        cv2.putText(
            frame,
            "GESTURE CHEATSHEET",
            (gx + 10, gy + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        hints = [
            ("Right Index Up", "Move Mouse", COLOR_RIGHT_HAND),
            ("Right Pinch", "Click / Drag", COLOR_RIGHT_HAND),
            ("Right Mid Pinch", "Right Click", COLOR_RIGHT_HAND),
            ("Left Peace (2)", "Scroll Up/Down", COLOR_LEFT_HAND),
            ("Left 3-Fingers", "System Volume", COLOR_LEFT_HAND),
            ("Both at a Time", "Move + Scroll", COLOR_ACTIVE_GREEN),
            ("Both Pinch", "Screen Zoom", COLOR_ZOOM),
        ]

        text_y = gy + 40
        for gesture_txt, action_txt, color in hints:
            # Bullet dot
            cv2.circle(frame, (gx + 12, text_y - 4), 3, color, -1)
            cv2.putText(
                frame,
                f"{gesture_txt}:",
                (gx + 22, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                action_txt,
                (gx + 135, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                1,
                cv2.LINE_AA,
            )
            text_y += 19

    def draw_hand_card(
        self,
        frame: np.ndarray,
        hand: Optional[AssignedHand],
        pose: Optional[HandPose],
        role_name: str,
        origin: Tuple[int, int],
        card_color: Tuple[int, int, int],
    ) -> None:
        """Renders an interactive status card for one physical hand."""
        x, y = origin
        card_w, card_h = 190, 80

        # Background card
        cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), (22, 22, 22), -1)
        cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), card_color if hand else (50, 50, 50), 1)

        # Header: Role Name & Confidence
        conf_str = f"({int(hand.confidence * 100)}%)" if hand else "(searching)"
        cv2.putText(
            frame,
            f"[{role_name} HAND] {conf_str}",
            (x + 10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            card_color if hand else (110, 110, 110),
            1,
            cv2.LINE_AA,
        )

        if hand and pose:
            # Line 1: Detected Hand Pose
            cv2.putText(
                frame,
                f"Pose: {pose.name}",
                (x + 10, y + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            # Line 2: Connected Laptop Action Badge
            action_color = (0, 255, 0) if pose.action != "NONE" else (140, 140, 140)
            cv2.rectangle(frame, (x + 8, y + 52), (x + card_w - 8, y + 72), (38, 38, 38), -1)
            cv2.putText(
                frame,
                f">> Action: {pose.action}",
                (x + 12, y + 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                action_color,
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Show hand to camera",
                (x + 10, y + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

    def render(
        self,
        frame: np.ndarray,
        roles: RoleFrameResult,
        gestures: TwoHandGestureResult,
        fps: float,
        is_paused: bool = False,
        mode: str = "DESKTOP",
        is_low_light: bool = False,
    ) -> None:
        """Renders complete HUD overlay with action indicators and gesture guidance."""
        h, w, _ = frame.shape

        # 1. Active cursor zone boundary
        self.draw_active_region(frame)

        # 2. Main System HUD Header (Top Left)
        header_w = 400
        header_h = 75
        cv2.rectangle(frame, (10, 10), (10 + header_w, 10 + header_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (10 + header_w, 10 + header_h), (60, 60, 60), 1)

        # Title
        cv2.putText(
            frame,
            "AirControl V2 — Two-Hand Gesture Control",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        # Safety Status & Mode
        status_label = "PAUSED" if is_paused else "ON"
        status_color = COLOR_PAUSED if is_paused else COLOR_ACTIVE_GREEN
        cv2.putText(
            frame,
            f"Status: {status_label}",
            (20, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            status_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Mode: {mode}",
            (170, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (325, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    def render(
        self,
        frame: np.ndarray,
        roles: RoleFrameResult,
        gestures: TwoHandGestureResult,
        fps: float,
        is_paused: bool,
        mode: str,
        is_low_light: bool = False,
        failsafe_triggered: bool = False,
    ) -> None:
        """Renders complete graphical HUD overlay over preview frame."""
        h, w, _ = frame.shape

        # 0. Draw subtle active cursor navigation region
        self.draw_active_region(frame)

        # 1. Fingertip Reticles & Motion Indicators
        if roles.right_hand is not None and len(roles.right_hand.landmarks) > 8:
            tip = roles.right_hand.landmarks[8]
            cx, cy = int(tip[0] * w), int(tip[1] * h)
            if gestures.right and gestures.right.name == "INDEX UP":
                # Precision cursor crosshair
                cv2.circle(frame, (cx, cy), 12, COLOR_RIGHT_HAND, 2, cv2.LINE_AA)
                cv2.drawMarker(frame, (cx, cy), COLOR_RIGHT_HAND, cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
            elif gestures.right and gestures.right.name == "PINCH":
                # Filled circle on click
                cv2.circle(frame, (cx, cy), 14, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 16, (255, 255, 255), 2, cv2.LINE_AA)

        if roles.left_hand is not None and len(roles.left_hand.landmarks) > 8:
            tip = roles.left_hand.landmarks[8]
            cx, cy = int(tip[0] * w), int(tip[1] * h)
            if gestures.left and gestures.left.name == "INDEX + MIDDLE (PEACE)":
                # Scroll indicator
                cv2.circle(frame, (cx, cy), 12, COLOR_LEFT_HAND, 2, cv2.LINE_AA)
                arrow_dir = -8 if gestures.left.scroll_dy > 0 else (8 if gestures.left.scroll_dy < 0 else 0)
                if arrow_dir != 0:
                    cv2.arrowedLine(frame, (cx, cy), (cx, cy + arrow_dir), COLOR_LEFT_HAND, 2, tipLength=0.4)

        # 2. Main Title and Status Bar
        header_w = 380
        cv2.rectangle(frame, (10, 10), (10 + header_w, 85), (24, 24, 24), -1)
        cv2.rectangle(frame, (10, 10), (10 + header_w, 85), (60, 60, 60), 1)

        cv2.putText(
            frame,
            "AirControl V2 - Two-Hand Control",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Status Badge (ON / PAUSED)
        badge_text = "PAUSED" if is_paused else "ACTIVE"
        badge_color = COLOR_PAUSED if is_paused else COLOR_ACTIVE_GREEN
        cv2.rectangle(frame, (20, 44), (95, 66), badge_color, -1)
        cv2.putText(
            frame,
            badge_text,
            (26, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Mode Badge
        cv2.putText(
            frame,
            f"Mode: {mode}",
            (108, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Live FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (260, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        # Model complexity status badge
        model_lbl = "FULL" if getattr(config, "MODEL_COMPLEXITY", 1) == 1 else "LITE"
        cv2.putText(
            frame,
            f"Model: {model_lbl} (T)",
            (345, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 220, 100),
            1,
            cv2.LINE_AA,
        )

        # 3. Dedicated Hand Cards (Left & Right)
        card_y = 95
        # Left Hand card (Amber/Orange)
        self.draw_hand_card(
            frame=frame,
            hand=roles.left_hand,
            pose=gestures.left,
            role_name="LEFT",
            origin=(10, card_y),
            card_color=COLOR_LEFT_HAND,
        )
        # Right Hand card (Cyan/Blue)
        self.draw_hand_card(
            frame=frame,
            hand=roles.right_hand,
            pose=gestures.right,
            role_name="RIGHT",
            origin=(210, card_y),
            card_color=COLOR_RIGHT_HAND,
        )

        # 4. Prominent Action Banner (Below Hand Cards)
        action_y = card_y + 90
        cv2.rectangle(frame, (10, action_y), (10 + header_w, action_y + 35), (28, 28, 28), -1)
        cv2.rectangle(frame, (10, action_y), (10 + header_w, action_y + 35), (75, 75, 75), 1)

        summary_color = COLOR_ACTIVE_GREEN if gestures.priority_level != "IDLE" else (160, 160, 160)
        if gestures.priority_level == "ZOOM":
            summary_color = COLOR_ZOOM

        cv2.putText(
            frame,
            f"LIVE ACTION: {gestures.action_summary}",
            (20, action_y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            summary_color,
            1,
            cv2.LINE_AA,
        )

        # 5. On-Screen Gesture Cheatsheet (Top Right)
        self.draw_gesture_cheatsheet(frame)

        # 6. Ambiguity Warning Banner
        if roles.is_ambiguous:
            warn_y = action_y + 45
            cv2.rectangle(frame, (10, warn_y), (10 + header_w, warn_y + 30), (0, 0, 140), -1)
            cv2.rectangle(frame, (10, warn_y), (10 + header_w, warn_y + 30), (0, 0, 255), 1)
            cv2.putText(
                frame,
                f"WARNING: {roles.ambiguity_reason}",
                (20, warn_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 7. FailSafe Alert Banner
        if failsafe_triggered:
            fs_y = h - 85
            cv2.rectangle(frame, (10, fs_y), (w - 10, fs_y + 32), (0, 0, 180), -1)
            cv2.rectangle(frame, (10, fs_y), (w - 10, fs_y + 32), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "SAFETY PAUSE: Mouse reached screen corner. Press Space to resume.",
                (20, fs_y + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 8. Low-Light Warning (Bottom Left)
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

        # 9. Bottom Navigation Hints
        cv2.putText(
            frame,
            "Keys: Space = Pause | M = Mode | T = Model (Full/Lite) | R = Reset | S = Settings | Q = Quit",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
