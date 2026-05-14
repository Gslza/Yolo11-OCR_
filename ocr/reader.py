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
    "ocr_angle": 0,
}


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Apply optional light preprocessing when OCR on the original crop is weak."""
    if crop is None or crop.size == 0:
        raise ValueError("Crop botol kosong sehingga OCR tidak dapat dijalankan.")

    height, width = crop.shape[:2]
    processed = crop
    if width < settings.OCR_RESIZE_WIDTH:
        scale = settings.OCR_RESIZE_WIDTH / max(width, 1)
        processed = cv2.resize(
            processed,
            (settings.OCR_RESIZE_WIDTH, int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
    gray = cv2.bilateralFilter(gray, 5, 45, 45)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 4.5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, sharpen_kernel)


class BeverageOCR:
    """Read text from a bottle crop, clean it, and match it to the product database."""

    def __init__(self, database_path: Path = settings.DATABASE_PATH) -> None:
        self.reader = easyocr.Reader(settings.OCR_LANGUAGES, gpu=settings.OCR_GPU)
        self.matcher = ProductMatcher(database_path)

    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate an image by 0, 90, 180, or 270 degrees for vertical label OCR."""
        normalized_angle = angle % 360
        if normalized_angle == 0:
            return image
        if normalized_angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if normalized_angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if normalized_angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, normalized_angle, 1.0)
        return cv2.warpAffine(image, matrix, (width, height))

    def read_text_once(self, image: np.ndarray) -> dict[str, object]:
        """Run EasyOCR once and return text plus scoring metadata."""
        if image is None or image.size == 0:
            return {"text": "", "word_count": 0, "total_confidence": 0.0, "words": []}

        ocr_results: list[Any] = self.reader.readtext(image)
        words: list[str] = []
        total_confidence = 0.0
        for _, text, confidence in ocr_results:
            confidence_value = float(confidence)
            cleaned_word = normalize_text(str(text))
            if confidence_value < settings.OCR_MIN_CONFIDENCE or not cleaned_word:
                continue
            words.extend(cleaned_word.split())
            total_confidence += confidence_value

        normalized_text = normalize_text(" ".join(words))
        valid_words = normalized_text.split()
        return {
            "text": normalized_text,
            "word_count": len(valid_words),
            "total_confidence": total_confidence,
            "words": valid_words,
        }

    def read_text_with_rotation(self, crop: np.ndarray) -> dict[str, object]:
        """Try EasyOCR at configured rotations and select by word count then confidence."""
        if crop is None or crop.size == 0:
            return {"text": "", "word_count": 0, "total_confidence": 0.0, "angle": 0}

        angles = settings.OCR_ROTATION_ANGLES if settings.OCR_ROTATION_ENABLED else [0]
        best_result: dict[str, object] = {"text": "", "word_count": 0, "total_confidence": 0.0, "angle": 0}
        for angle in angles:
            rotated_crop = self.rotate_image(crop, int(angle))
            result = self.read_text_once(rotated_crop)
            result["angle"] = int(angle)
            if self._is_better_ocr_result(result, best_result):
                best_result = result

        if not best_result.get("text"):
            processed = preprocess_crop(crop)
            for angle in angles:
                rotated_crop = self.rotate_image(processed, int(angle))
                result = self.read_text_once(rotated_crop)
                result["angle"] = int(angle)
                result["preprocessed"] = True
                if self._is_better_ocr_result(result, best_result):
                    best_result = result
        return best_result

    def read_text(self, crop: np.ndarray) -> str:
        """Read OCR text from a crop using rotation-aware OCR."""
        return str(self.read_text_with_rotation(crop).get("text", ""))

    def analyze_crop(self, crop: np.ndarray) -> dict[str, object]:
        """Return a display-ready product result for one detected bottle crop."""
        ocr_result = self.read_text_with_rotation(crop)
        ocr_text = str(ocr_result.get("text", ""))
        match = self.matcher.match(ocr_text)
        if match is not None:
            match["ocr_angle"] = ocr_result.get("angle", 0)
            return match

        result = dict(UNKNOWN_RESULT)
        result["ocr_text"] = ocr_text
        result["ocr_angle"] = ocr_result.get("angle", 0)
        return result

    @staticmethod
    def _is_better_ocr_result(candidate: dict[str, object], current: dict[str, object]) -> bool:
        candidate_words = int(candidate.get("word_count", 0))
        current_words = int(current.get("word_count", 0))
        if candidate_words != current_words:
            return candidate_words > current_words
        return float(candidate.get("total_confidence", 0.0)) > float(current.get("total_confidence", 0.0))
