"""OpenCV drawing helpers for the beverage detection UI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

GREEN = (0, 180, 0)
YELLOW = (0, 220, 255)
RED = (0, 0, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 140, 0)
GRAY = (70, 70, 70)


def status_color(status: str) -> tuple[int, int, int]:
    """Return a BGR color that represents a beverage safety status."""
    normalized = str(status).upper()
    if normalized == "AMAN":
        return GREEN
    if normalized == "BATAS WAJAR":
        return YELLOW
    if normalized == "TIDAK DISARANKAN":
        return RED
    return WHITE


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Sequence[float],
    confidence: float,
    label: str = "bottle",
) -> np.ndarray:
    """Draw the YOLO bounding box and confidence label on a frame."""
    x1, y1, x2, y2 = [int(value) for value in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
    text = f"{label}: {confidence:.2f}"
    cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + 170, y1), GREEN, -1)
    cv2.putText(frame, text, (x1 + 5, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, BLACK, 2)
    return frame


def draw_info_panel(
    frame: np.ndarray,
    result: Mapping[str, object] | None,
    confidence: float | None = None,
) -> np.ndarray:
    """Draw product name, OCR text, sugar, status, match score, and YOLO confidence."""
    panel_height = 190
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), BLACK, -1)
    cv2.addWeighted(overlay, 0.66, frame, 0.34, 0, frame)

    lines = ["YOLO11 + EasyOCR Beverage Detection"]
    if confidence is not None:
        lines.append(f"YOLO confidence: {confidence:.2f}")

    if result:
        product_name = result.get("name", "Tidak dikenali")
        ocr_text = result.get("ocr_text", "-") or "-"
        sugar = result.get("sugar_g", "-")
        status = result.get("status", "Tidak dikenali")
        score = result.get("match_score", 0.0)
        match_type = result.get("match_type", "none")
        lines.extend(
            [
                f"Produk: {product_name}",
                f"OCR: {ocr_text}",
                f"Gula: {sugar} g | Status: {status}",
                f"Match: {match_type} ({float(score):.2f})" if isinstance(score, (int, float)) else f"Match: {match_type}",
            ]
        )
    else:
        lines.append("Arahkan label botol ke kamera. Tekan 'q' untuk keluar.")

    y = 30
    for index, line in enumerate(lines):
        color = BLUE if index == 0 else WHITE
        if "Status:" in line:
            color = status_color(line.split("Status:", 1)[1].strip())
        cv2.putText(frame, line[:105], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 28
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw the current frames-per-second counter."""
    text = f"FPS: {fps:.1f}"
    x = max(10, frame.shape[1] - 135)
    cv2.rectangle(frame, (x - 8, 10), (frame.shape[1] - 10, 42), GRAY, -1)
    cv2.putText(frame, text, (x, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    return frame


def draw_freeze_banner(frame: np.ndarray, seconds_left: float) -> np.ndarray:
    """Draw a freeze-mode banner so users know the recognized result is being held."""
    text = f"HASIL DI-FREEZE ({seconds_left:.1f}s)"
    cv2.rectangle(frame, (0, frame.shape[0] - 48), (frame.shape[1], frame.shape[0]), BLACK, -1)
    cv2.putText(
        frame,
        text,
        (20, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        YELLOW,
        2,
    )
    return frame
