"""
Central configuration constants for the Touchless Lock System.
Override these with CLI args or environment-specific config if needed.
"""

from pathlib import Path

# ----------------------
# Logging
# ----------------------
LOG_LEVEL = "INFO"

# ----------------------
# Camera / Video defaults
# ----------------------
DEFAULT_SOURCE = 0             # camera index or video file path
DEFAULT_WIDTH = None           # desired capture width
DEFAULT_HEIGHT = None          # desired capture height
DEFAULT_MAX_FPS = 30.0         # max processing FPS
DEFAULT_CAMERA_BUFFER_SIZE = None  # buffer size for camera queue
DEFAULT_CV2_THREADS = None     # number of OpenCV threads
DISPLAY_DEFAULT = True         # whether to show frames in window

# ----------------------
# MediaPipe / Detector defaults
# ----------------------
MAX_NUM_HANDS = 1
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.6

# ----------------------
# ROI / preprocessing
# ----------------------
ROI_SIZE = (96, 96)  # width, height for LBP/geometry extraction

# ----------------------
# Verification / Registration defaults
# ----------------------
DEFAULT_USE_GEOMETRY = True
DEFAULT_SIMILARITY_THRESHOLD = 0.975
REGISTRATION_TARGETS = 10      # number of frames to collect per user registration
VERIFICATION_TARGETS = 5       # number of frames to collect per verification attempt
VERIFICATION_COOLDOWN_SECONDS = 2.0  # cooldown after verification

# ----------------------
# LBP / feature extraction thresholds
# ----------------------
VARIANCE_THRESHOLD = 0.005     # minimum global variance to consider ROI valid

# ----------------------
# ESP32 / Hardware integration
# ----------------------
ESP32_ENABLED_DEFAULT = False
ESP32_PORT_DEFAULT = None       # serial port for ESP32, e.g., "/dev/ttyUSB0"

# ESP32-CAM specific optimizations
ESP32_SIMILARITY_THRESHOLD = 0.975  # Adjusted threshold for ESP32-CAM (0.991 should pass)
ESP32_VARIANCE_THRESHOLD = 0.001   # Higher variance threshold for ESP32-CAM
ESP32_USE_ENHANCED_PREPROCESSING = True  # Enable ESP32-specific image enhancement

# ----------------------
# Snapshot / saving
# ----------------------
SAVE_SNAPS_DEFAULT = False
SNAPS_DIR = Path("snapshots")  # directory to store snapshots

# ----------------------
# Database / storage
# ----------------------
DB_PATH = Path("palm_database.db")

# ----------------------
# UI / Windows
# ----------------------
WINDOW_NAME = "Touchless Lock System"
WINDOW_DEFAULT_RESOLUTION = (640, 480)
