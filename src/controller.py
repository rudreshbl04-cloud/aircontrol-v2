"""
Desktop Automation and Hardware Controller Module for AirControl V2.

Responsible for executing:
- Right Hand: Mouse cursor movement with active-region mapping, speed-adaptive smoothing, and dead zone
- Right Hand: Single left-click with release-to-click debounce
- Left Hand: Smooth vertical scrolling with dead zone and rate limiting
- Left Hand: System volume control via pycaw Windows Core Audio with graceful fallback
- Two-Hand: Screen zoom in / out with baseline displacement and rate limiting
- Media: Play / pause toggle via Space / Media keys
- Presentation: Slide next / previous navigation via arrow keys
- PyAutoGUI failsafe protection and error isolation
"""

import math
import time
from typing import Optional, Tuple, Any
import pyautogui

import config
from src.gestures import TwoHandGestureResult, HandPose

# Always ensure PyAutoGUI failsafe is enabled
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0  # Zero internal pause for instant, zero-latency OS automation


class AudioController:
    """Manages Windows system audio via pycaw with seamless fallback."""

    def __init__(self) -> None:
        self.is_available: bool = False
        self.vol_endpoint: Optional[Any] = None
        self.last_volume_pct: float = 50.0
        self.smoothed_volume_pct: float = 50.0
        self.last_update_time: float = 0.0

        self._init_pycaw()

    def _init_pycaw(self) -> None:
        """Initializes the Windows Core Audio endpoint."""
        try:
            from pycaw.pycaw import AudioUtilities
            speakers = AudioUtilities.GetSpeakers()
            if hasattr(speakers, "EndpointVolume"):
                # Modern pycaw API (v2024+)
                self.vol_endpoint = speakers.EndpointVolume
            else:
                # Legacy pycaw API
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import IAudioEndpointVolume
                interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.vol_endpoint = interface.QueryInterface(IAudioEndpointVolume)

            if self.vol_endpoint is not None:
                current_scalar = self.vol_endpoint.GetMasterVolumeLevelScalar()
                self.last_volume_pct = current_scalar * 100.0
                self.smoothed_volume_pct = self.last_volume_pct
                self.is_available = True
        except Exception as err:
            self.is_available = False
            self.vol_endpoint = None

    def set_volume(self, target_pct: float, current_time: Optional[float] = None) -> bool:
        """Smoothly adjusts system volume to target percentage (0.0 to 100.0)."""
        if not self.is_available or self.vol_endpoint is None:
            return False

        if current_time is None:
            current_time = time.perf_counter()

        # Exponential smoothing to prevent sudden audio spikes
        self.smoothed_volume_pct += config.VOLUME_SMOOTHING * (target_pct - self.smoothed_volume_pct)

        # Rate limit system volume updates to avoid excessive OS calls
        if abs(self.smoothed_volume_pct - self.last_volume_pct) >= (config.VOLUME_UPDATE_THRESHOLD * 100.0):
            if (current_time - self.last_update_time) >= 0.04:  # At most 25 updates per second
                clamped_scalar = max(0.0, min(1.0, self.smoothed_volume_pct / 100.0))
                try:
                    self.vol_endpoint.SetMasterVolumeLevelScalar(clamped_scalar, None)
                    self.last_volume_pct = self.smoothed_volume_pct
                    self.last_update_time = current_time
                    return True
                except Exception:
                    pass
        return False


class DesktopController:
    """Orchestrates desktop input automation safely and responsively."""

    def __init__(self) -> None:
        # Screen dimensions
        try:
            self.screen_width, self.screen_height = pyautogui.size()
        except Exception:
            self.screen_width, self.screen_height = 1920, 1080

        # Cursor smoothing state
        self.cursor_x: float = float(self.screen_width // 2)
        self.cursor_y: float = float(self.screen_height // 2)
        self.prev_target_x: float = float(self.screen_width // 2)
        self.prev_target_y: float = float(self.screen_height // 2)
        self.has_cursor_baseline: bool = False
        self.baseline_grace_frames: int = 0
        self.max_grace_frames: int = 6  # ~0.2s grace hold before dropping baseline

        # Jitter dead zone threshold in screen pixels
        self.jitter_dead_zone: float = 3.5

        # Drag & drop state
        self.is_dragging: bool = False
        self.drag_start_time: float = 0.0

        # Click state
        self.click_ready: bool = True
        self.last_click_time: float = 0.0
        self.last_right_click_time: float = 0.0

        # Scroll rate limiting
        self.last_scroll_time: float = 0.0

        # Zoom rate limiting
        self.last_zoom_time: float = 0.0

        # Presentation and Media rate limiting
        self.last_slide_time: float = 0.0
        self.last_play_pause_time: float = 0.0

        # Audio subsystem
        self.audio = AudioController()

    def update_cursor(self, fingertip_x: float, fingertip_y: float) -> Optional[Tuple[int, int]]:
        """
        Maps normalized index fingertip coordinates to full screen bounds
        using active camera region, precision dead-zone filtering, and continuous velocity smoothing.
        """
        # 1. Normalize position inside configured active region
        x_min, x_max = config.ACTIVE_AREA_LEFT, config.ACTIVE_AREA_RIGHT
        y_min, y_max = config.ACTIVE_AREA_TOP, config.ACTIVE_AREA_BOTTOM

        norm_x = (fingertip_x - x_min) / max(0.01, (x_max - x_min))
        norm_y = (fingertip_y - y_min) / max(0.01, (y_max - y_min))

        # 2. Clamp coordinates to valid screen boundaries [0.0, 1.0]
        clamped_x = max(0.0, min(1.0, norm_x))
        clamped_y = max(0.0, min(1.0, norm_y))

        # 3. Map to pixel screen coordinates
        target_x = clamped_x * self.screen_width
        target_y = clamped_y * self.screen_height

        self.baseline_grace_frames = 0

        if not self.has_cursor_baseline:
            self.cursor_x = target_x
            self.cursor_y = target_y
            self.prev_target_x = target_x
            self.prev_target_y = target_y
            self.has_cursor_baseline = True
            ix = int(max(2, min(self.screen_width - 3, self.cursor_x)))
            iy = int(max(2, min(self.screen_height - 3, self.cursor_y)))
            try:
                pyautogui.moveTo(ix, iy)
                return ix, iy
            except pyautogui.FailSafeException:
                raise
            except Exception:
                return None

        # 4. Precision dead-zone filter on input target motion
        target_delta = math.hypot(target_x - self.prev_target_x, target_y - self.prev_target_y)
        dist_to_cursor = math.hypot(target_x - self.cursor_x, target_y - self.cursor_y)

        if target_delta < self.jitter_dead_zone and dist_to_cursor < self.jitter_dead_zone:
            return int(self.cursor_x), int(self.cursor_y)

        if target_delta >= self.jitter_dead_zone:
            self.prev_target_x = target_x
            self.prev_target_y = target_y
        else:
            # Hand held steady: damp micro-tremor by anchoring to stable target
            target_x = self.prev_target_x
            target_y = self.prev_target_y

        dist = math.hypot(target_x - self.cursor_x, target_y - self.cursor_y)
        if dist < 0.75:
            self.cursor_x = target_x
            self.cursor_y = target_y
        else:
            # 5. Continuous velocity-adaptive smoothing
            base_factor = config.SMOOTHING_FACTOR
            speed_factor = 1.0 / (1.0 + math.exp(-0.06 * (dist - 28.0)))
            smooth_factor = 0.22 + (0.90 - 0.22) * speed_factor
            smooth_factor = max(0.18, min(0.92, smooth_factor * (base_factor / 0.35)))

            self.cursor_x += smooth_factor * (target_x - self.cursor_x)
            self.cursor_y += smooth_factor * (target_y - self.cursor_y)

        # Clamp target integers to prevent PyAutoGUI corner failsafe triggers
        ix = int(max(2, min(self.screen_width - 3, self.cursor_x)))
        iy = int(max(2, min(self.screen_height - 3, self.cursor_y)))

        try:
            pyautogui.moveTo(ix, iy)
            return ix, iy
        except pyautogui.FailSafeException:
            raise
        except Exception:
            return None

    def trigger_left_click(self, current_time: Optional[float] = None) -> bool:
        """Triggers a single left mouse click with release-to-click debouncing."""
        if current_time is None:
            current_time = time.perf_counter()

        if self.click_ready and (current_time - self.last_click_time >= config.CLICK_COOLDOWN):
            try:
                pyautogui.click()
                self.click_ready = False
                self.last_click_time = current_time
                return True
            except pyautogui.FailSafeException:
                raise
            except Exception:
                pass
        return False

    def trigger_right_click(self, current_time: Optional[float] = None) -> bool:
        """Triggers a single right mouse click with debouncing."""
        if current_time is None:
            current_time = time.perf_counter()

        if current_time - self.last_right_click_time >= config.CLICK_COOLDOWN:
            try:
                pyautogui.rightClick()
                self.last_right_click_time = current_time
                return True
            except pyautogui.FailSafeException:
                raise
            except Exception:
                pass
        return False

    def release_click_state(self) -> None:
        """Re-arms click readiness when pinch is released and drops any active drag."""
        self.click_ready = True
        if self.is_dragging:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
            self.is_dragging = False

    def trigger_scroll(self, scroll_dy: float, current_time: Optional[float] = None) -> bool:
        """Translates vertical hand movement into mouse wheel scrolling."""
        if current_time is None:
            current_time = time.perf_counter()

        if current_time - self.last_scroll_time < config.SCROLL_COOLDOWN:
            return False

        if abs(scroll_dy) < config.SCROLL_THRESHOLD:
            return False

        # Moving up (positive dy) scrolls up; moving down scrolls down
        direction = 1 if scroll_dy > 0 else -1
        # Proportional scrolling magnitude based on hand velocity
        scale = min(3.0, max(1.0, abs(scroll_dy) / 0.03))
        amount = int(direction * config.SCROLL_AMOUNT * scale)

        try:
            pyautogui.scroll(amount)
            self.last_scroll_time = current_time
            return True
        except pyautogui.FailSafeException:
            raise
        except Exception:
            return False

    def trigger_zoom(self, zoom_delta: float, current_time: Optional[float] = None) -> bool:
        """Executes application zoom via Ctrl + '+' / Ctrl + '-' shortcuts."""
        if current_time is None:
            current_time = time.perf_counter()

        # Rate limit zoom steps (at most 5 steps per second)
        if current_time - self.last_zoom_time < 0.20:
            return False

        if abs(zoom_delta) < 0.035:
            return False

        try:
            if zoom_delta > 0:
                pyautogui.hotkey("ctrl", "+")
            else:
                pyautogui.hotkey("ctrl", "-")

            self.last_zoom_time = current_time
            return True
        except pyautogui.FailSafeException:
            raise
        except Exception:
            return False

    def trigger_play_pause(self, current_time: Optional[float] = None) -> bool:
        """Sends Space keypress to play/pause media in compatible players."""
        if current_time is None:
            current_time = time.perf_counter()

        if current_time - self.last_play_pause_time >= config.PLAY_PAUSE_COOLDOWN:
            try:
                pyautogui.press("space")
                self.last_play_pause_time = current_time
                return True
            except pyautogui.FailSafeException:
                raise
            except Exception:
                pass
        return False

    def trigger_slide(self, direction: str, current_time: Optional[float] = None) -> bool:
        """Sends Left / Right arrow keypress to navigate presentation slides."""
        if current_time is None:
            current_time = time.perf_counter()

        if current_time - self.last_slide_time >= config.SWIPE_COOLDOWN:
            try:
                key = "right" if "RIGHT" in direction else "left"
                pyautogui.press(key)
                self.last_slide_time = current_time
                return True
            except pyautogui.FailSafeException:
                raise
            except Exception:
                pass
        return False

    def reset_cursor_baseline(self, force: bool = False) -> None:
        """Resets cursor tracking baseline when right hand is lost or explicitly reset."""
        if force:
            self.has_cursor_baseline = False
            self.baseline_grace_frames = 0
            if self.is_dragging:
                try:
                    pyautogui.mouseUp()
                except Exception:
                    pass
                self.is_dragging = False
            return

        self.baseline_grace_frames += 1
        if self.baseline_grace_frames > self.max_grace_frames:
            self.has_cursor_baseline = False
            if self.is_dragging:
                try:
                    pyautogui.mouseUp()
                except Exception:
                    pass
                self.is_dragging = False

    def execute(
        self,
        gestures: TwoHandGestureResult,
        roles: Optional[Any] = None,
        current_time: Optional[float] = None,
    ) -> None:
        """
        Executes arbitrated desktop automation actions based on current gesture frame result.
        Supports simultaneous two-hand actions, drag & drop, right-click, and failsafe safety.
        """
        if current_time is None:
            current_time = time.perf_counter()

        # ----------------------------------------------------------------------
        # Priority 1 & 2: Safety Pause or Ambiguous Hand Tracking (Zero Automation)
        # ----------------------------------------------------------------------
        if gestures.priority_level in ("SAFETY", "AMBIGUOUS"):
            self.reset_cursor_baseline(force=True)
            self.release_click_state()
            return

        # ----------------------------------------------------------------------
        # Priority 3: Two-Hand Zoom
        # ----------------------------------------------------------------------
        if gestures.is_zoom_active:
            self.trigger_zoom(gestures.zoom_delta, current_time)
            self.reset_cursor_baseline(force=True)
            self.release_click_state()
            return

        # ----------------------------------------------------------------------
        # Right Hand Actions: Cursor Movement, Left Click, Drag & Drop, Right Click
        # ----------------------------------------------------------------------
        cursor_hand = None
        if roles is not None:
            cursor_hand = roles.right_hand if roles.right_hand is not None else roles.left_hand

        if gestures.right is not None:
            if gestures.right.name == "PINCH":
                self.trigger_left_click(current_time)
                # Drag & Drop: Holding pinch and moving activates mouse down
                if (
                    getattr(config, "DRAG_ENABLED", True)
                    and cursor_hand is not None
                    and len(cursor_hand.landmarks) > 8
                ):
                    pinch_tip = cursor_hand.landmarks[8]
                    if not self.is_dragging:
                        if self.drag_start_time == 0.0:
                            self.drag_start_time = current_time
                        elif (current_time - self.drag_start_time) >= 0.20:
                            try:
                                pyautogui.mouseDown()
                                self.is_dragging = True
                            except pyautogui.FailSafeException:
                                raise
                            except Exception:
                                pass
                    if self.is_dragging:
                        self.update_cursor(pinch_tip[0], pinch_tip[1])

            elif gestures.right.name == "RIGHT PINCH":
                self.release_click_state()
                self.trigger_right_click(current_time)

            else:
                self.release_click_state()

            if gestures.right.name == "INDEX UP":
                self.drag_start_time = 0.0
                if cursor_hand is not None and len(cursor_hand.landmarks) > 8:
                    tip = cursor_hand.landmarks[8]
                    self.update_cursor(tip[0], tip[1])
                else:
                    self.reset_cursor_baseline(force=False)
            elif gestures.right.name != "PINCH" or not self.is_dragging:
                self.reset_cursor_baseline(force=False)
        else:
            self.reset_cursor_baseline(force=False)
            self.release_click_state()

        # ----------------------------------------------------------------------
        # Left Hand Actions: Scroll, Volume, Media, Swipe (Runs Simultaneously!)
        # ----------------------------------------------------------------------
        if gestures.left is not None:
            if gestures.left.name == "INDEX + MIDDLE (PEACE)":
                self.trigger_scroll(gestures.left.scroll_dy, current_time)

            elif gestures.left.name == "THREE FINGERS" and gestures.left.volume_pct is not None:
                self.audio.set_volume(gestures.left.volume_pct, current_time)

            elif gestures.left.name == "OPEN PALM" and gestures.left.action == "PLAY/PAUSE":
                self.trigger_play_pause(current_time)

            elif gestures.left.name == "SWIPE" and gestures.left.swipe_direction is not None:
                self.trigger_slide(gestures.left.swipe_direction, current_time)
