"""CSV logging helpers for beverage detections."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping


LOG_COLUMNS = [
    "timestamp",
    "confidence",
    "ocr_text",
    "product_name",
    "sugar_g",
    "status",
    "screenshot_path",
]


class DetectionLogger:
    """Append structured detection records to a daily CSV log file."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        date_label = datetime.now().strftime("%Y%m%d")
        self.log_path = self.log_dir / f"detections_{date_label}.csv"
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.log_path.exists() and self.log_path.stat().st_size > 0:
            return
        with self.log_path.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=LOG_COLUMNS)
            writer.writeheader()

    def log_detection(
        self,
        confidence: float,
        result: Mapping[str, object],
        screenshot_path: Path | None = None,
    ) -> None:
        """Write one detection row containing YOLO, OCR, product, and decision data."""
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "confidence": f"{confidence:.4f}",
            "ocr_text": result.get("ocr_text", ""),
            "product_name": result.get("name", ""),
            "sugar_g": result.get("sugar_g", ""),
            "status": result.get("status", ""),
            "screenshot_path": str(screenshot_path or ""),
        }
        with self.log_path.open("a", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=LOG_COLUMNS)
            writer.writerow(row)
