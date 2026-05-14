"""Central configuration for the YOLO11 + EasyOCR beverage detector."""

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"
DATABASE_PATH = BASE_DIR / "database" / "beverages.json"
SCREENSHOT_DIR = BASE_DIR / "screenshot"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"

# Camera settings
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WINDOW_NAME = "YOLO11 EasyOCR Beverage Detection"

# YOLO inference settings
CONFIDENCE_THRESHOLD = 0.60
IOU_THRESHOLD = 0.45
TARGET_CLASS_NAME = "bottle"

# OCR settings
OCR_LANGUAGES = ["en"]
OCR_GPU = False
OCR_MIN_CONFIDENCE = 0.40
OCR_RESIZE_WIDTH = 640
OCR_ROTATION_ENABLED = True
OCR_ROTATION_ANGLES = [0, 90, 180, 270]

# Product matching settings
FUZZY_MATCH_THRESHOLD = 0.72

# Runtime behavior
DETECTION_COOLDOWN_SECONDS = 3
FREEZE_DURATION_SECONDS = 5
SCREENSHOT_ENABLED = True

# Sugar decision thresholds in grams
SUGAR_SAFE_MAX_EXCLUSIVE = 10
SUGAR_REASONABLE_MAX_INCLUSIVE = 20
