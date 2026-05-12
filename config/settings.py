"""Application settings for the Smart Beverage Detection System."""

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"
DATABASE_PATH = BASE_DIR / "database" / "beverages.json"
SCREENSHOT_DIR = BASE_DIR / "screenshot"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "Smart Beverage Detection System"

# YOLO inference settings
CONFIDENCE_THRESHOLD = 0.80
TARGET_CLASS_NAME = "bottle"
IOU_THRESHOLD = 0.45

# Runtime behaviour
FREEZE_DURATION_SECONDS = 4
DETECTION_COOLDOWN_SECONDS = 4
SCREENSHOT_ENABLED = True

# OCR settings
OCR_LANGUAGES = ["en"]
OCR_GPU = False
OCR_MIN_CONFIDENCE = 0.20
OCR_RESIZE_WIDTH = 640

# Product matching settings
FUZZY_MATCH_THRESHOLD = 0.72

# Sugar decision thresholds in grams
SUGAR_SAFE_MAX_EXCLUSIVE = 10
SUGAR_REASONABLE_MAX_INCLUSIVE = 20
