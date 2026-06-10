"""Real-time YOLO11 + EasyOCR beverage detection application."""

from __future__ import annotations

import platform
import sys
import threading
import time
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from config import settings
from ocr.reader import BeverageOCR
from utils.display import draw_bounding_box, draw_fps, draw_freeze_banner, draw_info_panel
from utils.logger import DetectionLogger
from utils.screenshot import save_detection_screenshot


def ensure_runtime_paths() -> None:
    """Create output folders and validate required runtime files."""
    settings.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not settings.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model YOLO tidak ditemukan di {settings.MODEL_PATH}. "
            "Letakkan file best.pt pada folder model/."
        )
    if not settings.DATABASE_PATH.exists():
        print(f"[WARN] Database produk tidak ditemukan di {settings.DATABASE_PATH}; matching akan dinonaktifkan.")


def open_camera() -> cv2.VideoCapture:
    """Open the configured webcam and apply requested resolution."""
    backend = cv2.CAP_DSHOW if platform.system().lower() == "windows" else 0
    capture = cv2.VideoCapture(settings.CAMERA_INDEX, backend) if backend else cv2.VideoCapture(settings.CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"Kamera index {settings.CAMERA_INDEX} tidak tersedia atau sedang digunakan.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
    return capture


def safe_crop(frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
    """Crop a YOLO bounding box while clamping coordinates to frame boundaries."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def is_bottle_detection(model: YOLO, class_id: int) -> bool:
    """Return True when the detection class is the target bottle class."""
    names = model.names
    if isinstance(names, dict):
        class_name = str(names.get(class_id, class_id)).lower()
        class_count = len(names)
    else:
        class_name = str(names[class_id]).lower() if class_id < len(names) else str(class_id)
        class_count = len(names)
    return class_name == settings.TARGET_CLASS_NAME.lower() or class_count == 1


def find_best_bottle_detection(model: YOLO, frame: np.ndarray) -> tuple[list[float], float] | None:
    """Run YOLO and return the highest-confidence bottle detection."""
    yolo_results = model.predict(
        frame,
        conf=settings.CONFIDENCE_THRESHOLD,
        iou=settings.IOU_THRESHOLD,
        verbose=False,
    )

    best_detection: tuple[list[float], float] | None = None
    for result in yolo_results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            if confidence < settings.CONFIDENCE_THRESHOLD or not is_bottle_detection(model, class_id):
                continue
            bbox = box.xyxy[0].tolist()
            if best_detection is None or confidence > best_detection[1]:
                best_detection = (bbox, confidence)
    return best_detection


class OCRWorker:
    """Run OCR analysis in a background thread so the camera loop stays responsive."""

    def __init__(self, ocr_engine: BeverageOCR) -> None:
        self._engine = ocr_engine
        self._thread: threading.Thread | None = None
        self._result: dict[str, object] | None = None
        self._lock = threading.Lock()
        self._busy = False
        self._last_submit_time = 0.0

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, crop: np.ndarray) -> None:
        """Send a crop for background OCR if the worker is idle and cooldown elapsed."""
        now = time.monotonic()
        if self.busy or (now - self._last_submit_time) < settings.DETECTION_COOLDOWN_SECONDS:
            return
        self._last_submit_time = now
        with self._lock:
            self._busy = True
            self._result = None
        thread = threading.Thread(target=self._run, args=(crop.copy(),), daemon=True)
        thread.start()

    def _run(self, crop: np.ndarray) -> None:
        try:
            result = self._engine.analyze_crop(crop)
            with self._lock:
                self._result = result
        finally:
            with self._lock:
                self._busy = False

    def collect(self) -> dict[str, object] | None:
        """Return the completed OCR result and clear it, or None if not ready."""
        with self._lock:
            result = self._result
            self._result = None
            return result


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    ocr_worker: OCRWorker,
    logger: DetectionLogger,
    last_result: dict[str, object] | None,
) -> tuple[np.ndarray, dict[str, object] | None, float | None, bool]:
    """Run YOLO every frame; OCR runs in a background thread without blocking."""
    rendered = frame.copy()
    detection = find_best_bottle_detection(model, frame)

    if detection is None:
        return draw_info_panel(rendered, None), None, None, False

    bbox, confidence = detection
    draw_bounding_box(rendered, bbox, confidence, settings.TARGET_CLASS_NAME)

    # Collect any completed background OCR result
    result = last_result
    recognized_now = False
    ocr_result = ocr_worker.collect()
    if ocr_result is not None:
        result = ocr_result
        recognized_now = bool(result.get("name") != "Tidak dikenali")

    # Submit a new OCR job when the worker is idle (cooldown enforced internally)
    crop = safe_crop(frame, bbox)
    if crop is not None:
        ocr_worker.submit(crop)

    rendered = draw_info_panel(rendered, result, confidence)
    if recognized_now and result is not None:
        screenshot_path = None
        if settings.SCREENSHOT_ENABLED:
            screenshot_path = save_detection_screenshot(rendered, settings.SCREENSHOT_DIR, str(result["name"]))
            if isinstance(result, dict) and screenshot_path:
                result["screenshot_path"] = str(screenshot_path)
                result["screenshot_filename"] = screenshot_path.name
        logger.log_detection(confidence, result, screenshot_path)
    return rendered, result, confidence, recognized_now


def main() -> int:
    """Application entry point. Run with: python detect.py"""
    capture = None
    try:
        ensure_runtime_paths()
        model = YOLO(str(settings.MODEL_PATH))
        ocr_engine = BeverageOCR(settings.DATABASE_PATH)
        ocr_worker = OCRWorker(ocr_engine)
        logger = DetectionLogger(settings.LOG_DIR)
        capture = open_camera()

        last_result: dict[str, object] | None = None
        freeze_until = 0.0
        freeze_frame: np.ndarray | None = None
        fps = 0.0
        previous_frame_time = time.monotonic()

        while True:
            now = time.monotonic()
            elapsed = now - previous_frame_time
            previous_frame_time = now
            if elapsed > 0:
                fps = (0.90 * fps) + (0.10 * (1.0 / elapsed)) if fps else 1.0 / elapsed

            if freeze_frame is not None and time.monotonic() < freeze_until:
                display_frame = freeze_frame.copy()
                seconds_left = max(0.0, freeze_until - time.monotonic())
                display_frame = draw_freeze_banner(display_frame, seconds_left)
            else:
                freeze_frame = None
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Gagal membaca frame dari kamera.")

                display_frame, product, _, recognized_now = process_frame(
                    frame,
                    model,
                    ocr_worker,
                    logger,
                    last_result,
                )
                last_result = product
                if recognized_now:
                    freeze_frame = display_frame.copy()
                    freeze_until = time.monotonic() + settings.FREEZE_DURATION_SECONDS

            draw_fps(display_frame, fps)
            cv2.imshow(settings.WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
