"""
AirControl Configuration Module.

All user-adjustable settings and parameters for computer vision,
tracking, active screen mapping, gesture thresholds, and action cooldowns.
"""

# ==============================================================================
# Camera & Video Capture Settings
# ==============================================================================
# Device index for the webcam (0 is typically the built-in webcam).
CAMERA_INDEX: int = 0

# Requested webcam capture resolution (width x height in pixels).
# Standard 640x480 or 1280x720 balances accuracy and real-time processing FPS.
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480

# Flip preview horizontally (selfie mode) so moving hand right moves cursor right.
FLIP_CAMERA: bool = True

# Target frame rate estimation / display
TARGET_FPS: int = 30

# Default active mode on startup ("DESKTOP", "MEDIA", or "PRESENTATION")
DEFAULT_MODE: str = "DESKTOP"

# Fast downscaled resolution for real-time MediaPipe inference (balances 25+ FPS & high landmark precision)
INFERENCE_WIDTH: int = 480
INFERENCE_HEIGHT: int = 360

# ==============================================================================
# MediaPipe Hand Tracking Settings (AirControl V2 Two-Hand Tracking)
# ==============================================================================
# Model complexity: 0 = Lite (fastest), 1 = Full (high accuracy, zero jitter)
MODEL_COMPLEXITY: int = 1

# Enable pinch-to-drag (holding pinch and moving hand drags windows/files/text)
DRAG_ENABLED: bool = True

# Enable right-click gesture (thumb-middle finger pinch on Right hand)
RIGHT_CLICK_ENABLED: bool = True

# Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful.
MEDIAPIPE_DETECTION_CONFIDENCE: float = 0.60

# Minimum confidence value ([0.0, 1.0]) for hand landmarks to be tracked successfully.
MEDIAPIPE_TRACKING_CONFIDENCE: float = 0.60

# Maximum number of hands to detect. AirControl V2 processes up to TWO hands simultaneously.
MAX_NUM_HANDS: int = 2

# Minimum confidence threshold for MediaPipe handedness classification
ROLE_CONFIDENCE_THRESHOLD: float = 0.55

# Number of frames a hand can be momentarily lost (e.g. fast motion) before resetting state
HAND_OCCLUSION_MAX_FRAMES: int = 5

# Normalized distance threshold below which two hands are considered overlapping/ambiguous
AMBIGUITY_DISTANCE_THRESHOLD: float = 0.08


# ==============================================================================
# Active Camera Region (Screen Mapping Bounds)
# ==============================================================================
# Normalized coordinates (0.0 to 1.0) defining the active region of the camera frame
# mapped to the full laptop screen. Only fingertip positions inside this box
# will navigate the desktop display, preventing users from having to reach edges of the camera.
ACTIVE_AREA_LEFT: float = 0.15    # 15% from the left boundary
ACTIVE_AREA_RIGHT: float = 0.85   # 85% from the left boundary (15% margin from right)
ACTIVE_AREA_TOP: float = 0.15     # 15% from the top boundary
ACTIVE_AREA_BOTTOM: float = 0.85  # 85% from the top boundary (15% margin from bottom)

# ==============================================================================
# Mouse Movement & Cursor Smoothing
# ==============================================================================
# Smoothing factor between 0.0 and 1.0 for cursor movement.
# Higher value (e.g. 0.7) means faster response with slight jitter.
# Lower value (e.g. 0.2) means smoother movement but slightly more latency.
SMOOTHING_FACTOR: float = 0.35

# ==============================================================================
# Gesture Recognition Thresholds
# ==============================================================================
# Normalized Euclidean distance threshold between thumb tip and index tip for pinch detection.
PINCH_THRESHOLD: float = 0.05

# Minimum normalized vertical movement required to register a scroll step.
SCROLL_THRESHOLD: float = 0.008

# Number of scroll units to apply per trigger.
SCROLL_AMOUNT: int = 35

# Minimum normalized horizontal displacement over recent frames to register a swipe.
SWIPE_THRESHOLD: float = 0.15

# Duration in seconds an open palm must be held continuously to toggle ON/PAUSED.
OPEN_PALM_HOLD_SECONDS: float = 2.0

# ==============================================================================
# Volume Control Settings (Windows pycaw)
# ==============================================================================
# Normalized distance between thumb tip and index tip corresponding to 0% volume.
VOLUME_MIN_DISTANCE: float = 0.03

# Normalized distance between thumb tip and index tip corresponding to 100% volume.
VOLUME_MAX_DISTANCE: float = 0.25

# Exponential smoothing factor for volume adjustments (0.0 to 1.0).
VOLUME_SMOOTHING: float = 0.25

# Minimum percentage change required before sending a system volume update call.
VOLUME_UPDATE_THRESHOLD: float = 0.02

# ==============================================================================
# Action Cooldowns (Debouncing to Prevent Accidental Multiple Triggers)
# ==============================================================================
# Cooldown between consecutive mouse clicks in seconds.
CLICK_COOLDOWN: float = 0.35

# Cooldown between consecutive scroll actions in seconds.
SCROLL_COOLDOWN: float = 0.06

# Cooldown between consecutive swipe slide changes in seconds.
SWIPE_COOLDOWN: float = 0.60

# Cooldown between consecutive play/pause toggle triggers in seconds.
PLAY_PAUSE_COOLDOWN: float = 0.80

# ==============================================================================
# Safety & Quality Checks
# ==============================================================================
# Brightness threshold (mean grayscale value 0-255) below which a low-light warning is shown.
LOW_LIGHT_THRESHOLD: float = 40.0
