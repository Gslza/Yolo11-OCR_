"""OpenCV drawing helpers for detection and result overlays."""

from __future__ import annotations

from typing import Mapping, Sequence

import cv2
import numpy as np


GREEN = (0, 180, 0)
YELLOW = (0, 215, 255)
RED = (0, 0, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 120, 0)


def status_color(status: str) -> tuple[int, int, int]:
    """Return a BGR color that represents a beverage safety status."""
    normalized = status.upper()
    if normalized == "AMAN":
        return GREEN
    if normalized == "BATAS WAJAR":
        return YELLOW
    return RED


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Sequence[int],
    confidence: float,
    label: str = "bottle",
) -> np.ndarray:
    """Draw the YOLO bounding box and confidence on a frame."""
    x1, y1, x2, y2 = [int(value) for value in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    return frame


def draw_info_panel(
    frame: np.ndarray,
    result: Mapping[str, object] | None,
    confidence: float | None = None,
) -> np.ndarray:
    """Draw detection, OCR, product, sugar, and decision information."""
    panel_height = 170
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), BLACK, -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    lines = ["Smart Beverage Detection System"]
    if confidence is not None:
        lines.append(f"YOLO confidence: {confidence:.2f}")

    if result:
        product_name = result.get("name", "Tidak dikenali")
        ocr_text = result.get("ocr_text", "-")
        sugar = result.get("sugar_g", "-")
        status = result.get("status", "Tidak dikenali")
        lines.extend(
            [
                f"Produk: {product_name}",
                f"OCR: {ocr_text}",
                f"Gula: {sugar} g",
                f"Status: {status}",
            ]
        )
    else:
        lines.append("Arahkan label botol ke kamera. Tekan 'q' untuk keluar.")

    y = 30
    for index, line in enumerate(lines):
        color = WHITE
        if line.startswith("Status:"):
            color = status_color(line.replace("Status:", "").strip())
        if index == 0:
            color = BLUE
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        y += 28
    return frame


def draw_freeze_banner(frame: np.ndarray, seconds_left: float) -> np.ndarray:
    """Draw a freeze-mode banner so users know the result is being held."""
    text = f"HASIL DI-FREEZE ({seconds_left:.1f}s)"
    cv2.rectangle(frame, (0, frame.shape[0] - 45), (frame.shape[1], frame.shape[0]), BLACK, -1)
    cv2.putText(
        frame,
        text,
        (20, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        YELLOW,
        2,
    )
    return frame
