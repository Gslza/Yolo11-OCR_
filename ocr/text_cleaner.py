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
}

_OCR_TOKEN_TRANSLATION = str.maketrans({"|": "I", "8": "B", "4": "A"})
_ZERO_CONTEXT_PATTERN = re.compile(r"(?<=[A-Z])0|0(?=[A-Z])")
_ONE_CONTEXT_PATTERN = re.compile(r"(?<=[A-Z])1|1(?=[A-Z])")


def normalize_text(text: str) -> str:
    """Normalize OCR text while keeping numbers that may be part of product names.

    The function fixes common label OCR mistakes in a conservative way: ``0`` and
    ``1`` are only converted when adjacent to letters, while ``4`` and ``8`` are
    converted because they frequently represent ``A`` and ``B`` in beverage brand
    words such as ``FANT4`` and ``A8C``.
    """
    uppercase = str(text or "").upper().translate(_OCR_TOKEN_TRANSLATION)
    letter_context_fixed = _ZERO_CONTEXT_PATTERN.sub("O", uppercase)
    letter_context_fixed = _ONE_CONTEXT_PATTERN.sub("I", letter_context_fixed)
    alphanumeric_spaces = re.sub(r"[^A-Z0-9\s]", " ", letter_context_fixed)
    return re.sub(r"\s+", " ", alphanumeric_spaces).strip()


def is_text_too_generic(text: str) -> bool:
    """Return True when OCR text is empty or only contains generic beverage words."""
    normalized = normalize_text(text)
    if not normalized:
        return True
    words = normalized.split()
    return bool(words) and all(word in GENERIC_WORDS for word in words)
