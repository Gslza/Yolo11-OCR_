"""OCR preprocessing, text normalization, and product matching."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np

from config import settings


@dataclass(frozen=True)
class Beverage:
    """One beverage record loaded from the JSON database."""

    name: str
    aliases: list[str]
    sugar_g: float
    status: str


def normalize_text(text: str) -> str:
    """Normalize OCR text for exact and fuzzy matching."""
    uppercase = text.upper()
    alphanumeric_spaces = re.sub(r"[^A-Z0-9\s]", " ", uppercase)
    return re.sub(r"\s+", " ", alphanumeric_spaces).strip()


def decide_status(sugar_g: float) -> str:
    """Apply the project sugar rules to produce a child-safety status."""
    if sugar_g > settings.SUGAR_REASONABLE_MAX_INCLUSIVE:
        return "Tidak Disarankan"
    if settings.SUGAR_SAFE_MAX_EXCLUSIVE <= sugar_g <= settings.SUGAR_REASONABLE_MAX_INCLUSIVE:
        return "Batas Wajar"
    return "Aman"


def load_beverage_database(database_path: Path) -> list[Beverage]:
    """Load and validate the beverage JSON database."""
    if not database_path.exists():
        raise FileNotFoundError(f"Database produk tidak ditemukan: {database_path}")

    with database_path.open("r", encoding="utf-8") as database_file:
        raw_data = json.load(database_file)

    if not isinstance(raw_data, list):
        raise ValueError("Database produk harus berupa list JSON.")

    beverages: list[Beverage] = []
    for item in raw_data:
        if not isinstance(item, dict):
            raise ValueError("Setiap item database harus berupa object JSON.")
        name = str(item["name"])
        aliases = [normalize_text(str(alias)) for alias in item.get("aliases", [])]
        sugar_g = float(item["sugar_g"])
        status = decide_status(sugar_g)
        beverages.append(Beverage(name=name, aliases=aliases, sugar_g=sugar_g, status=status))
    return beverages


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Prepare a cropped bottle image before EasyOCR reads the label."""
    if crop.size == 0:
        raise ValueError("Crop botol kosong sehingga OCR tidak dapat dijalankan.")

    height, width = crop.shape[:2]
    if width < settings.OCR_RESIZE_WIDTH:
        scale = settings.OCR_RESIZE_WIDTH / max(width, 1)
        crop = cv2.resize(crop, (settings.OCR_RESIZE_WIDTH, int(height * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    return cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )


class BeverageOCR:
    """Read bottle labels and match OCR output to known beverage products."""

    def __init__(self, database_path: Path) -> None:
        self.reader = easyocr.Reader(settings.OCR_LANGUAGES, gpu=settings.OCR_GPU)
        self.beverages = load_beverage_database(database_path)

    def read_text(self, crop: np.ndarray) -> str:
        """Run EasyOCR on a crop and return one normalized text string."""
        processed = preprocess_crop(crop)
        ocr_results: list[Any] = self.reader.readtext(processed)
        filtered_words = [
            str(text)
            for _, text, confidence in ocr_results
            if float(confidence) >= settings.OCR_MIN_CONFIDENCE
        ]
        return normalize_text(" ".join(filtered_words))

    def identify_product(self, ocr_text: str) -> dict[str, object] | None:
        """Identify a product using exact matching first, then fuzzy matching."""
        normalized_ocr = normalize_text(ocr_text)
        if not normalized_ocr:
            return None

        # Exact matching protects high-confidence OCR names from fuzzy false positives.
        for beverage in self.beverages:
            search_terms = [normalize_text(beverage.name), *beverage.aliases]
            if any(term and term in normalized_ocr for term in search_terms):
                return self._to_result(beverage, normalized_ocr, 1.0, "exact")

        best_match: tuple[Beverage, float] | None = None
        for beverage in self.beverages:
            for alias in [normalize_text(beverage.name), *beverage.aliases]:
                score = SequenceMatcher(None, normalized_ocr, alias).ratio()
                if best_match is None or score > best_match[1]:
                    best_match = (beverage, score)

        if best_match and best_match[1] >= settings.FUZZY_MATCH_THRESHOLD:
            return self._to_result(best_match[0], normalized_ocr, best_match[1], "fuzzy")
        return None

    def analyze_crop(self, crop: np.ndarray) -> dict[str, object] | None:
        """Run OCR and product identification for a bottle crop."""
        ocr_text = self.read_text(crop)
        product = self.identify_product(ocr_text)
        if product is None:
            return {"ocr_text": ocr_text, "name": "Tidak dikenali", "status": "Tidak dikenali"}
        return product

    @staticmethod
    def _to_result(beverage: Beverage, ocr_text: str, match_score: float, match_type: str) -> dict[str, object]:
        return {
            "name": beverage.name,
            "aliases": beverage.aliases,
            "sugar_g": beverage.sugar_g,
            "status": beverage.status,
            "ocr_text": ocr_text,
            "match_score": round(match_score, 4),
            "match_type": match_type,
        }
