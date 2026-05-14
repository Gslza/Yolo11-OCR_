"""Text normalization helpers for beverage-label OCR output."""

from __future__ import annotations

import re

GENERIC_WORDS = {
    "COFFEE",
    "TEA",
    "MILK",
    "ORANGE",
    "STRAWBERRY",
    "LEMON",
    "DRINK",
    "WATER",
    "SWEAT",
    "MALT",
    "CHOCO",
    "SPARKLING",
}

OCR_CORRECTIONS = {
    "A8C": "ABC",
    "C0CA": "COCA",
    "C0LA": "COLA",
    "G0LDA": "GOLDA",
    "FANT4": "FANTA",
    "F4NTA": "FANTA",
    "GOLD4": "GOLDA",
    "SPR1TE": "SPRITE",
    "POCAR1": "POCARI",
    "M1ZONE": "MIZONE",
    "MIZ0NE": "MIZONE",
    "M1LK": "MILK",
    "B0TOL": "BOTOL",
    "S4RI": "SARI",
    "H4RUM": "HARUM",
}

_ZERO_CONTEXT_PATTERN = re.compile(r"(?<=[A-Z])0|0(?=[A-Z])")
_ONE_CONTEXT_PATTERN = re.compile(r"(?<=[A-Z])1|1(?=[A-Z])")


def normalize_text(text: str) -> str:
    """Normalize OCR text to uppercase words and fix common OCR confusions."""
    uppercase = str(text or "").upper().replace("|", "I")
    alphanumeric_spaces = re.sub(r"[^A-Z0-9\s]", " ", uppercase)
    compacted = re.sub(r"\s+", " ", alphanumeric_spaces).strip()

    corrected_words = [OCR_CORRECTIONS.get(word, word) for word in compacted.split()]
    corrected = " ".join(corrected_words)
    corrected = _ZERO_CONTEXT_PATTERN.sub("O", corrected)
    corrected = _ONE_CONTEXT_PATTERN.sub("I", corrected)
    corrected = corrected.replace("8", "B").replace("4", "A")
    corrected_words = [OCR_CORRECTIONS.get(word, word) for word in corrected.split()]
    return re.sub(r"\s+", " ", " ".join(corrected_words)).strip()


def is_text_too_generic(text: str) -> bool:
    """Return True when OCR text is empty or only contains generic beverage words."""
    normalized = normalize_text(text)
    if not normalized:
        return True
    words = normalized.split()
    return bool(words) and all(word in GENERIC_WORDS for word in words)
