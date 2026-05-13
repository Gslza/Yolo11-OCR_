# YOLO11 EasyOCR Beverage Detection

Project ini menjalankan pipeline **YOLO11 + EasyOCR** untuk mendeteksi botol minuman dari webcam, membaca teks label, membersihkan hasil OCR, mencocokkan produk ke database JSON, lalu menampilkan kadar gula dan status keamanan konsumsi.

## Alur Sistem

1. Load model YOLO dari `model/best.pt`.
2. Buka webcam dengan OpenCV.
3. YOLO mendeteksi objek `bottle` pada setiap frame.
4. Sistem memilih bounding box botol dengan confidence terbaik.
5. Crop area botol secara aman agar koordinat tidak keluar frame.
6. EasyOCR membaca teks label dari crop.
7. Teks OCR dibersihkan oleh `ocr/text_cleaner.py`.
8. `ocr/product_matcher.py` mencocokkan teks ke `database/beverages.json` menggunakan exact matching dan fuzzy matching.
9. OpenCV menampilkan bounding box, OCR text, nama produk, kadar gula, status, confidence YOLO, dan FPS.
10. Jika produk dikenali, frame di-freeze beberapa detik, screenshot disimpan, dan log deteksi ditulis.

## Struktur Folder

```text
.
├── config/
│   ├── __init__.py
│   └── settings.py              # Path, kamera, threshold YOLO/OCR, cooldown, freeze
├── database/
│   └── beverages.json           # Database produk, alias OCR, dan kadar gula
├── logs/                        # Log CSV hasil deteksi
├── model/
│   └── best.pt                  # Letakkan model YOLO11 terlatih di sini
├── ocr/
│   ├── __init__.py
│   ├── reader.py                # Integrasi EasyOCR + ProductMatcher
│   ├── text_cleaner.py          # Normalisasi OCR dan filter kata umum
│   └── product_matcher.py       # Matching database dan decision status gula
├── output/                      # Folder output tambahan
├── screenshot/                  # Screenshot otomatis produk dikenali
├── utils/
│   ├── __init__.py
│   ├── display.py               # Overlay OpenCV, FPS, panel info, freeze banner
│   ├── logger.py                # Logger CSV
│   └── screenshot.py            # Penyimpanan screenshot aman
├── detect.py                    # Entry point real-time webcam
├── README.md
└── requirements.txt
```

## Instalasi

Gunakan Python 3.10+ jika memungkinkan.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pada Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Model YOLO

File model tidak disertakan di repository. Letakkan model YOLO11 hasil training dengan nama berikut:

```text
model/best.pt
```

Aplikasi akan berhenti dengan pesan error yang jelas jika file tersebut belum tersedia.

## Menjalankan Program

Jalankan dari root repository:

```bash
python detect.py
```

Kontrol:

- Arahkan label botol ke kamera.
- Pastikan pencahayaan cukup dan label terlihat jelas.
- Tekan tombol `q` pada window OpenCV untuk keluar.

## Database Produk

Database berada di `database/beverages.json` dan berbentuk JSON list. Setiap item wajib memiliki:

- `name`: nama produk utama.
- `aliases`: daftar alias brand/varian dan kemungkinan salah baca OCR.
- `sugar_g`: kadar gula dalam gram.

Contoh:

```json
{
  "name": "ABC COFFEE",
  "aliases": ["ABC", "A8C", "ABC COFFEE", "ABC CHOCO MALT COFFEE"],
  "sugar_g": 17
}
```

Field `status` tidak perlu ditulis karena status dihitung otomatis:

- `sugar_g < 10`: **Aman**
- `10 <= sugar_g <= 20`: **Batas Wajar**
- `sugar_g > 20`: **Tidak Disarankan**

Alias dibuat spesifik agar kata umum seperti `COFFEE`, `TEA`, `MILK`, `ORANGE`, atau `STRAWBERRY` tidak langsung menyebabkan salah identifikasi produk.

## Konfigurasi Penting

Pengaturan utama ada di `config/settings.py`.

| Setting | Default | Keterangan |
| --- | --- | --- |
| `MODEL_PATH` | `model/best.pt` | Path model YOLO11 |
| `DATABASE_PATH` | `database/beverages.json` | Path database produk |
| `CAMERA_INDEX` | `0` | Index webcam OpenCV |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `640` / `480` | Resolusi webcam |
| `CONFIDENCE_THRESHOLD` | `0.60` | Minimum confidence YOLO |
| `IOU_THRESHOLD` | `0.45` | Threshold IoU YOLO |
| `TARGET_CLASS_NAME` | `bottle` | Class target deteksi |
| `OCR_LANGUAGES` | `["en"]` | Bahasa OCR Latin label minuman |
| `OCR_GPU` | `False` | EasyOCR CPU by default |
| `OCR_MIN_CONFIDENCE` | `0.40` | Minimum confidence OCR per kata |
| `FUZZY_MATCH_THRESHOLD` | `0.72` | Ambang fuzzy matching |
| `DETECTION_COOLDOWN_SECONDS` | `3` | Jeda agar OCR tidak berjalan setiap frame |
| `FREEZE_DURATION_SECONDS` | `5` | Durasi freeze saat produk dikenali |
| `SCREENSHOT_ENABLED` | `True` | Screenshot otomatis |

## Output Runtime

- **Window OpenCV** menampilkan bounding box botol, confidence YOLO, FPS, OCR text, nama produk, kadar gula, status, dan banner freeze.
- **Screenshot** produk dikenali disimpan ke folder `screenshot/`.
- **Log CSV** disimpan ke folder `logs/` dengan kolom timestamp, nama produk, OCR text, gula, status, confidence YOLO, dan path screenshot.

## Catatan Matching OCR

- `A8C` dinormalisasi menjadi `ABC` dan cocok ke **ABC COFFEE**.
- `G0LDA` dinormalisasi menjadi `GOLDA` dan cocok ke **GOLDA**.
- `C0CA C0LA` dinormalisasi menjadi `COCA COLA`.
- OCR yang hanya berisi kata umum seperti `COFFEE` atau `TEA` akan ditampilkan sebagai **Tidak dikenali**.
