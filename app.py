"""Flask web backend for the YOLO11 + EasyOCR Beverage Detection Dashboard."""

from __future__ import annotations

import csv
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory

# Include the current directory in python path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parent))

from config import settings
from detect import find_best_bottle_detection, process_frame, OCRWorker, ensure_runtime_paths, open_camera
from ocr.reader import BeverageOCR
from utils.logger import DetectionLogger, LOG_COLUMNS
from utils.display import draw_fps, draw_freeze_banner
from ultralytics import YOLO

# Initialize Flask app
# Serving static files from the 'web' directory
app = Flask(__name__, static_folder='web', static_url_path='')

# Global thread-safe structures for camera streaming and status updates
latest_jpeg_frame = None
latest_frame_lock = threading.Lock()

current_status = {
    "fps": 0.0,
    "active_detection": {
        "state": "idle",
        "yolo_confidence": None,
        "ocr_text": None,
        "ocr_angle": None,
        "product": None
    }
}
current_status_lock = threading.Lock()

sse_clients: list[queue.Queue] = []
sse_clients_lock = threading.Lock()

stop_event = threading.Event()


def get_today_csv_path() -> Path:
    """Return the Path to today's CSV detection log."""
    date_label = datetime.now().strftime("%Y%m%d")
    return settings.LOG_DIR / f"detections_{date_label}.csv"


def get_stats_from_csv() -> dict[str, int]:
    """Parse today's CSV detection log and compile summary counts."""
    stats = {"total": 0, "safe": 0, "warning": 0, "danger": 0}
    csv_path = get_today_csv_path()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return stats

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                stats["total"] += 1
                status = str(row.get("status", "")).strip()
                if status == "Aman":
                    stats["safe"] += 1
                elif status == "Batas Wajar":
                    stats["warning"] += 1
                elif status == "Tidak Disarankan":
                    stats["danger"] += 1
    except Exception as err:
        print(f"[WARN] Gagal membaca statistik dari CSV: {err}", file=sys.stderr)
    return stats


def read_history_from_csv() -> list[dict[str, object]]:
    """Parse today's CSV detection log and return the 20 most recent events."""
    history_events: list[dict[str, object]] = []
    csv_path = get_today_csv_path()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return history_events

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # Safely parse numeric types
                def safe_float(val: str | None) -> float | None:
                    try:
                        return float(val) if val else None
                    except ValueError:
                        return None

                def safe_int(val: str | None) -> int | None:
                    try:
                        return int(float(val)) if val else None
                    except ValueError:
                        return None

                full_path_str = row.get("screenshot_path", "")
                screenshot_fn = Path(full_path_str).name if full_path_str else ""
                screenshot_url = f"/screenshot/{screenshot_fn}" if screenshot_fn else ""

                event = {
                    "timestamp": row.get("timestamp", ""),
                    "product_name": row.get("product_name", ""),
                    "ocr_text": row.get("ocr_text", ""),
                    "sugar_g": safe_int(row.get("sugar_g")) if row.get("sugar_g") != "-" else "-",
                    "status": row.get("status", ""),
                    "yolo_confidence": safe_float(row.get("yolo_confidence")),
                    "match_score": safe_float(row.get("match_score")),
                    "match_type": row.get("match_type", ""),
                    "screenshot_path": screenshot_url,
                }
                history_events.append(event)
    except Exception as err:
        print(f"[WARN] Gagal membaca riwayat dari CSV: {err}", file=sys.stderr)

    # Return events reversed (newest first), capped at 20
    return history_events[::-1][:20]


def broadcast_sse(data: dict[str, object]) -> None:
    """Safely broadcast JSON payload to all active SSE client queues."""
    with sse_clients_lock:
        for client_queue in list(sse_clients):
            try:
                client_queue.put_nowait(data)
            except queue.Full:
                # Queue full, client might have lagged, pop oldest to make room
                try:
                    client_queue.get_nowait()
                    client_queue.put_nowait(data)
                except (queue.Empty, queue.Full):
                    pass


class CameraThread(threading.Thread):
    """Background thread to handle camera acquisition, YOLO inference, OCR processing, and CSV logging."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.name = "CameraThread"

    def run(self) -> None:
        global latest_jpeg_frame, current_status
        capture = None
        
        print("[INFO] Memulai Thread Kamera dan Deteksi...")
        try:
            ensure_runtime_paths()
            
            # Load models
            print("[INFO] Memuat YOLO model...")
            model = YOLO(str(settings.MODEL_PATH))
            
            print("[INFO] Memuat EasyOCR engine...")
            ocr_engine = BeverageOCR(settings.DATABASE_PATH)
            ocr_worker = OCRWorker(ocr_engine)
            logger = DetectionLogger(settings.LOG_DIR)
            
            print("[INFO] Membuka kamera...")
            capture = open_camera()

            last_result: dict[str, object] | None = None
            freeze_until = 0.0
            freeze_frame: np.ndarray | None = None
            fps = 0.0
            previous_frame_time = time.monotonic()

            while not stop_event.is_set():
                now = time.monotonic()
                elapsed = now - previous_frame_time
                previous_frame_time = now
                if elapsed > 0:
                    fps = (0.90 * fps) + (0.10 * (1.0 / elapsed)) if fps else 1.0 / elapsed

                recognized_now = False
                confidence = None

                if freeze_frame is not None and time.monotonic() < freeze_until:
                    # Serve frozen frame with banner
                    display_frame = freeze_frame.copy()
                    seconds_left = max(0.0, freeze_until - time.monotonic())
                    display_frame = draw_freeze_banner(display_frame, seconds_left)
                    time.sleep(0.03)  # Reduce CPU load during freeze
                else:
                    freeze_frame = None
                    ok, frame = capture.read()
                    if not ok:
                        print("[ERROR] Gagal membaca frame dari kamera. Menunggu...", file=sys.stderr)
                        time.sleep(0.5)
                        continue

                    # Run YOLO and coordinate with background OCR
                    display_frame, product, confidence, recognized_now = process_frame(
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

                # Draw FPS on the display frame
                draw_fps(display_frame, fps)

                # Encode display frame to JPEG
                ok, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, settings.STREAM_QUALITY])
                if ok:
                    with latest_frame_lock:
                        latest_jpeg_frame = jpeg.tobytes()

                # Determine active detection state
                active_detection = {
                    "state": "idle",
                    "yolo_confidence": None,
                    "ocr_text": None,
                    "ocr_angle": None,
                    "product": None
                }

                # If we are in the freeze period, we hold the recognized product status
                if freeze_frame is not None:
                    if last_result:
                        active_detection["state"] = "recognized"
                        active_detection["product"] = last_result
                        active_detection["yolo_confidence"] = 1.0  # static for freeze
                    else:
                        active_detection["state"] = "idle"
                else:
                    # Otherwise, use live inference results
                    is_bottle_detected = (confidence is not None)
                    if is_bottle_detected:
                        prod_name = last_result.get("name") if last_result else None
                        if prod_name and prod_name != "Tidak dikenali":
                            active_detection["state"] = "recognized"
                            active_detection["product"] = last_result
                        else:
                            active_detection["state"] = "scanning"
                            active_detection["ocr_text"] = last_result.get("ocr_text") if last_result else ""
                            active_detection["ocr_angle"] = last_result.get("ocr_angle") if last_result else 0
                        active_detection["yolo_confidence"] = confidence
                    else:
                        active_detection["state"] = "idle"

                # Update global thread-safe state
                with current_status_lock:
                    current_status = {
                        "fps": fps,
                        "active_detection": active_detection
                    }

                # Construct real-time update broadcast
                update_payload = {
                    "fps": fps,
                    "active_detection": active_detection
                }

                # If a product was logged this frame, broadcast it as a new event
                if recognized_now and last_result:
                    screenshot_fn = last_result.get("screenshot_filename", "")
                    screenshot_url = f"/screenshot/{screenshot_fn}" if screenshot_fn else ""
                    new_log = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "product_name": last_result.get("name", ""),
                        "ocr_text": last_result.get("ocr_text", ""),
                        "sugar_g": last_result.get("sugar_g", "-"),
                        "status": last_result.get("status", ""),
                        "yolo_confidence": f"{confidence:.4f}" if confidence is not None else "1.0000",
                        "match_score": last_result.get("match_score", ""),
                        "match_type": last_result.get("match_type", ""),
                        "screenshot_path": screenshot_url
                    }
                    update_payload["new_log_event"] = new_log
                    update_payload["stats"] = get_stats_from_csv()

                broadcast_sse(update_payload)
                time.sleep(0.01)

        except Exception as error:
            print(f"[FATAL ERROR] Kegagalan pada thread kamera: {error}", file=sys.stderr)
        finally:
            if capture is not None:
                capture.release()
            print("[INFO] Thread Kamera dan Deteksi berhenti.")


# Flask Routes
@app.route("/screenshot/<filename>")
def get_screenshot(filename):
    """Serve saved detection screenshots."""
    return send_from_directory(str(settings.SCREENSHOT_DIR), filename)


@app.route("/")
def index():
    """Serve the single-page dashboard HTML."""
    return app.send_static_file("index.html")


@app.route("/video_feed")
def video_feed():
    """Stream MJPEG frames to the client."""
    def generate():
        while not stop_event.is_set():
            with latest_frame_lock:
                frame_bytes = latest_jpeg_frame

            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # Cap streaming rate to ~30 FPS
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/events")
def events():
    """Server-Sent Events endpoint for real-time status and telemetry."""
    def event_stream():
        client_queue = queue.Queue(maxsize=100)
        with sse_clients_lock:
            sse_clients.append(client_queue)

        # Broadcast initial status payload immediately upon connection
        with current_status_lock:
            initial_status = dict(current_status)
            initial_status["stats"] = get_stats_from_csv()

        yield f"data: {json.dumps(initial_status)}\n\n"

        try:
            while True:
                data = client_queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except GeneratorExit:
            pass
        finally:
            with sse_clients_lock:
                if client_queue in sse_clients:
                    sse_clients.remove(client_queue)

    return Response(event_stream(), mimetype='text/event-stream')


@app.route("/api/stats")
def api_stats():
    """Retrieve sugar classification metrics for today's logs."""
    return jsonify(get_stats_from_csv())


@app.route("/api/history")
def api_history():
    """Retrieve today's most recent 20 logged detections."""
    return jsonify(read_history_from_csv())


if __name__ == "__main__":
    # Start the camera detection thread only once (handling Werkzeug reloader behavior)
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        camera_thread = CameraThread()
        camera_thread.start()

    try:
        print(f"[INFO] Menjalankan server dashboard di http://localhost:{settings.WEB_PORT}")
        app.run(host=settings.WEB_HOST, port=settings.WEB_PORT, debug=False, threaded=True)
    finally:
        stop_event.set()
