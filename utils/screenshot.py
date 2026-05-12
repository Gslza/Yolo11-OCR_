"""Screenshot persistence helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def save_detection_screenshot(frame: np.ndarray, screenshot_dir: Path, product_name: str) -> Path:
    """Save a timestamped screenshot for a recognized product."""
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "_".join(product_name.upper().split())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = screenshot_dir / f"{timestamp}_{safe_name}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Gagal menyimpan screenshot ke {path}")
    return path
