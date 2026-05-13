"""Screenshot persistence helpers."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def save_detection_screenshot(frame: np.ndarray, screenshot_dir: Path, product_name: str) -> Path:
    """Save a timestamped detection screenshot using a filesystem-safe filename."""
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Z0-9_-]+", "_", product_name.upper()).strip("_") or "UNKNOWN"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = screenshot_dir / f"{timestamp}_{safe_name}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Gagal menyimpan screenshot ke {path}")
    return path
