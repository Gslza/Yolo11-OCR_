"""Database loading and safe product matching for cleaned OCR text."""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from config import settings
from ocr.text_cleaner import is_text_too_generic, normalize_text

SPECIFIC_BRAND_KEYWORDS = (
    "ABC",
    "GOLDA",
    "FANTA",
    "COCA",
    "COLA",
    "COKE",
    "SPRITE",
    "POCARI",
    "SOSRO",
    "ULTRA",
    "MIZONE",
    "FRESTEA",
    "ADEM",
    "PUCUK",
    "TEBS",
)


@dataclass(frozen=True)
class Beverage:
    """One validated beverage record loaded from the JSON database."""

    name: str
    aliases: tuple[str, ...]
    sugar_g: float
    status: str


def decide_status(sugar_g: float) -> str:
    """Return the safety status derived from sugar content in grams."""
    if sugar_g < settings.SUGAR_SAFE_MAX_EXCLUSIVE:
        return "Aman"
    if sugar_g <= settings.SUGAR_REASONABLE_MAX_INCLUSIVE:
        return "Batas Wajar"
    return "Tidak Disarankan"


def load_beverage_database(database_path: Path = settings.DATABASE_PATH) -> list[Beverage]:
    """Load and validate the beverage database JSON list."""
    if not database_path.exists():
        print(f"Peringatan: database produk tidak ditemukan: {database_path}")
        return []

    with database_path.open("r", encoding="utf-8") as database_file:
        raw_data: Any = json.load(database_file)

    if not isinstance(raw_data, list):
        raise ValueError("Database produk harus berupa JSON list.")

    beverages: list[Beverage] = []
    for index, item in enumerate(raw_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item database ke-{index} harus berupa object JSON.")
        missing_fields = {"name", "aliases", "sugar_g"} - set(item)
        if missing_fields:
            raise ValueError(f"Item database ke-{index} kurang field: {', '.join(sorted(missing_fields))}")
        if not isinstance(item["aliases"], list):
            raise ValueError(f"Field aliases item ke-{index} harus berupa list.")

        name = normalize_text(str(item["name"]))
        aliases = tuple(
            alias_text
            for alias in item["aliases"]
            if (alias_text := normalize_text(str(alias)))
        )
        sugar_g = float(item["sugar_g"])
        beverages.append(Beverage(name=name, aliases=aliases, sugar_g=sugar_g, status=decide_status(sugar_g)))
    return beverages


class ProductMatcher:
    """Match cleaned OCR output to beverage records with exact and fuzzy logic."""

    def __init__(self, database_path: Path = settings.DATABASE_PATH) -> None:
        self.beverages = load_beverage_database(database_path)

    def match(self, text: str) -> dict[str, object] | None:
        """Return a product result dictionary, or None when matching is unsafe."""
        ocr_text = normalize_text(text)
        if is_text_too_generic(ocr_text):
            return None

        exact_match = self._exact_match(ocr_text)
        if exact_match is not None:
            beverage, term = exact_match
            return self._to_result(beverage, ocr_text, 1.0, "exact" if term == beverage.name else f"exact:{term}")

        best_brand_match = self._specific_brand_match(ocr_text)
        if best_brand_match is not None:
            beverage, term = best_brand_match
            return self._to_result(beverage, ocr_text, 0.98, f"brand:{term}")

        best_beverage: Beverage | None = None
        best_score = 0.0
        best_term = ""
        for beverage in self.beverages:
            for term in self._terms_for(beverage):
                score = max(
                    SequenceMatcher(None, ocr_text, term).ratio(),
                    SequenceMatcher(None, ocr_text.replace(" ", ""), term.replace(" ", "")).ratio(),
                )
                if score > best_score:
                    best_beverage = beverage
                    best_score = score
                    best_term = term

        if best_beverage is not None and best_score >= settings.FUZZY_MATCH_THRESHOLD:
            return self._to_result(best_beverage, ocr_text, best_score, f"fuzzy:{best_term}")
        return None

    def _exact_match(self, ocr_text: str) -> tuple[Beverage, str] | None:
        compact_ocr = ocr_text.replace(" ", "")
        for beverage in self.beverages:
            for term in self._terms_for(beverage):
                if is_text_too_generic(term):
                    continue
                compact_term = term.replace(" ", "")
                if term in ocr_text or compact_term in compact_ocr:
                    return beverage, term
        return None

    def _specific_brand_match(self, ocr_text: str) -> tuple[Beverage, str] | None:
        words = set(ocr_text.split())
        compact_ocr = ocr_text.replace(" ", "")
        for keyword in SPECIFIC_BRAND_KEYWORDS:
            if keyword not in words and keyword not in compact_ocr:
                continue
            for beverage in self.beverages:
                terms = self._terms_for(beverage)
                if any(keyword in term.split() or keyword in term.replace(" ", "") for term in terms):
                    return beverage, keyword
        return None

    @staticmethod
    def _terms_for(beverage: Beverage) -> tuple[str, ...]:
        return (beverage.name, *beverage.aliases)

    @staticmethod
    def _to_result(beverage: Beverage, ocr_text: str, score: float, match_type: str) -> dict[str, object]:
        return {
            "name": beverage.name,
            "sugar_g": beverage.sugar_g,
            "status": beverage.status,
            "ocr_text": ocr_text,
            "match_score": round(score, 4),
            "match_type": match_type,
        }
