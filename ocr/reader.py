"""EasyOCR integration for beverage-label reading and product identification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np

from config import settings
from ocr.product_matcher import ProductMatcher
from ocr.text_cleaner import normalize_text

UNKNOWN_RESULT = {
    "name": "Tidak dikenali",
    "sugar_g": "-",
    "status": "Tidak dikenali",
    "ocr_text": "",
    "match_score": 0.0,
    "match_type": "none",
}


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Apply optional light preprocessing when OCR on the original crop is weak."""
    if crop is None or crop.size == 0:
        raise ValueError("Crop botol kosong sehingga OCR tidak dapat dijalankan.")

    height, width = crop.shape[:2]
    processed = crop
    if width < settings.OCR_RESIZE_WIDTH:
        scale = settings.OCR_RESIZE_WIDTH / max(width, 1)
        processed = cv2.resize(processed, (settings.OCR_RESIZE_WIDTH, int(height * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 45, 45)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 4.5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, sharpen_kernel)


class BeverageOCR:
    """Read text from a bottle crop, clean it, and match it to the product database."""

    def __init__(self, database_path: Path = settings.DATABASE_PATH) -> None:
        self.reader = easyocr.Reader(settings.OCR_LANGUAGES, gpu=settings.OCR_GPU)
        self.matcher = ProductMatcher(database_path)

    def read_text(self, crop: np.ndarray) -> str:
        """Read OCR text from the original crop first, then fallback to light preprocessing."""
        if crop is None or crop.size == 0:
            return ""

        raw_text = self._read_image(crop)
        cleaned = normalize_text(raw_text)
        if cleaned:
            return cleaned

        processed = preprocess_crop(crop)
        return normalize_text(self._read_image(processed))

    def analyze_crop(self, crop: np.ndarray) -> dict[str, object]:
        """Return a display-ready product result for one detected bottle crop."""
        ocr_text = self.read_text(crop)
        match = self.matcher.match(ocr_text)
        if match is not None:
            return match

        result = dict(UNKNOWN_RESULT)
        result["ocr_text"] = ocr_text
        return result

    def _read_image(self, image: np.ndarray) -> str:
        """Run EasyOCR and join words passing the configured confidence threshold."""
        ocr_results: list[Any] = self.reader.readtext(image)
        filtered_words = [
            str(text)
            for _, text, confidence in ocr_results
            if float(confidence) >= settings.OCR_MIN_CONFIDENCE
        ]
        return " ".join(filtered_words)
