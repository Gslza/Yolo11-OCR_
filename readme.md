# Smart Beverage Detection System

Sistem ini adalah aplikasi **real-time computer vision** berbasis **YOLO11**, **EasyOCR**, dan **OpenCV** untuk mendeteksi botol minuman dari webcam, membaca teks label produk, mencocokkan produk dengan database JSON, menampilkan kadar gula, dan memberikan status keamanan konsumsi untuk anak-anak.

> Project ini hanya berisi kode **inference application**. Kode training, dataset preparation, dan annotation pipeline tidak disertakan karena model YOLO11 diasumsikan sudah selesai dilatih.

## Fitur Utama

- Deteksi botol minuman real-time dari webcam menggunakan model YOLO11 `model/best.pt`.
- Validasi confidence YOLO dengan default threshold `0.80`.
- Crop otomatis area botol berdasarkan bounding box YOLO dengan proteksi batas frame.
- OCR label botol menggunakan EasyOCR.
- Normalisasi teks OCR ke uppercase, menghapus karakter spesial, dan merapikan spasi.
- Identifikasi produk dengan exact matching terlebih dahulu, lalu fuzzy matching jika exact matching gagal.
- Database produk berbasis JSON untuk `FANTA`, `COCA COLA`, `TEH BOTOL`, dan `SPRITE`.
- Decision system kadar gula:
  - `> 20g`: **Tidak Disarankan**
  - `10g - 20g`: **Batas Wajar**
  - `< 10g`: **Aman**
- Freeze screen setelah produk dikenali agar hasil mudah dibaca.
- Screenshot otomatis untuk produk yang berhasil dikenali.
- Logging hasil deteksi ke file CSV harian.
- Error handling untuk model tidak ditemukan, kamera tidak tersedia, database tidak valid, crop kosong, dan kegagalan screenshot.

## Struktur Project

```text
.
├── config/
│   ├── __init__.py
│   └── settings.py          # Konfigurasi path, kamera, threshold, OCR, dan runtime
├── database/
│   └── beverages.json       # Database produk minuman dan kadar gula
├── logs/                    # Output log CSV harian saat aplikasi berjalan
├── model/
│   └── best.pt              # Letakkan model YOLO11 hasil training di sini
├── ocr/
│   ├── __init__.py
│   └── reader.py            # Preprocessing OCR, normalisasi teks, matching produk
├── output/                  # Folder output tambahan jika dibutuhkan
├── screenshot/              # Screenshot otomatis hasil deteksi
├── utils/
│   ├── __init__.py
│   ├── display.py           # Overlay OpenCV dan tampilan freeze
│   ├── logger.py            # Logger CSV
│   └── screenshot.py        # Penyimpanan screenshot
├── detect.py                # Entry point aplikasi real-time
├── readme.md
└── requirements.txt
```

## Prasyarat

- Python 3.10 atau lebih baru direkomendasikan.
- Webcam aktif dan dapat diakses OpenCV.
- File model YOLO11 hasil training bernama `best.pt`.
- Sistem operasi dengan dukungan OpenCV GUI untuk menampilkan window monitor.

## Instalasi

1. Buat dan aktifkan virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   Pada Windows PowerShell:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependency inference:

   ```bash
   pip install -r requirements.txt
   ```

3. Letakkan model YOLO11 terlatih di path berikut:

   ```text
   model/best.pt
   ```

## Menjalankan Aplikasi

Jalankan aplikasi dari root repository:

```bash
python detect.py
```

Kontrol aplikasi:

- Arahkan label botol ke kamera.
- Pastikan label cukup terang dan terlihat jelas.
- Tekan tombol `q` pada window OpenCV untuk keluar.

## Alur Kerja Sistem

1. Webcam membaca frame video real-time.
2. YOLO11 menjalankan inference pada frame.
3. Sistem memilih deteksi `bottle` terbaik dengan confidence minimal `0.80`.
4. Bounding box botol di-crop secara aman dari frame.
5. Crop diproses untuk OCR: resize, grayscale, filter, sharpen, dan threshold.
6. EasyOCR membaca teks label.
7. Teks OCR dinormalisasi dan dibersihkan.
8. Sistem mencocokkan teks dengan database produk menggunakan exact matching dan fuzzy matching.
9. Jika produk dikenali, sistem mengambil kadar gula dari database.
10. Decision system menentukan status keamanan berdasarkan kadar gula.
11. Sistem menyimpan screenshot, menulis log CSV, dan menampilkan freeze screen.

## Konfigurasi Penting

Pengaturan utama berada di `config/settings.py`.

| Setting | Default | Keterangan |
| --- | --- | --- |
| `MODEL_PATH` | `model/best.pt` | Path model YOLO11 hasil training |
| `DATABASE_PATH` | `database/beverages.json` | Path database produk JSON |
| `CAMERA_INDEX` | `0` | Index webcam OpenCV |
| `CONFIDENCE_THRESHOLD` | `0.80` | Minimum confidence YOLO untuk lanjut OCR |
| `TARGET_CLASS_NAME` | `bottle` | Nama class target |
| `FREEZE_DURATION_SECONDS` | `4` | Durasi freeze screen setelah produk dikenali |
| `DETECTION_COOLDOWN_SECONDS` | `4` | Proteksi agar deteksi tidak berulang terus-menerus |
| `SCREENSHOT_ENABLED` | `True` | Mengaktifkan screenshot otomatis |
| `OCR_LANGUAGES` | `["en"]` | Bahasa OCR EasyOCR |
| `FUZZY_MATCH_THRESHOLD` | `0.72` | Ambang fuzzy matching produk |

## Format Database Produk

Database berada di `database/beverages.json` dengan format list object:

```json
{
  "name": "FANTA",
  "aliases": ["FANTA", "FANTA ORANGE", "FANT4"],
  "sugar_g": 23,
  "status": "Tidak Disarankan"
}
```

Catatan:

- Field `aliases` digunakan untuk membantu pencocokan OCR ketika teks tidak terbaca sempurna.
- Status akhir tetap dihitung ulang dari `sugar_g` oleh decision system agar konsisten dengan rule gula.

## Output Runtime

- Screenshot produk dikenali disimpan ke folder `screenshot/` dengan nama file berbasis timestamp.
- Log deteksi disimpan ke folder `logs/` dengan format `detections_YYYYMMDD.csv`.
- Informasi yang dicatat meliputi timestamp, confidence YOLO, hasil OCR, nama produk, kadar gula, status, dan path screenshot.

## Troubleshooting

### Model YOLO tidak ditemukan

Pastikan file model tersedia di:

```text
model/best.pt
```

### Kamera tidak tersedia

- Pastikan webcam tidak sedang digunakan aplikasi lain.
- Ubah `CAMERA_INDEX` di `config/settings.py` jika kamera Anda bukan index `0`.

### OCR tidak mengenali produk

- Pastikan label menghadap kamera dan cukup terang.
- Tambahkan alias OCR baru pada `database/beverages.json` jika pola salah baca sering muncul.
- Sesuaikan `OCR_MIN_CONFIDENCE` atau `FUZZY_MATCH_THRESHOLD` di `config/settings.py`.

### Window OpenCV tidak muncul

- Pastikan menjalankan aplikasi di environment desktop, bukan headless server.
- Jika menggunakan Linux server, pastikan display/GUI tersedia.

## Catatan Pengembangan

- Aplikasi dapat dijalankan langsung dengan `python detect.py` setelah dependency terpasang dan `model/best.pt` tersedia.
- Project ini tidak menyediakan training script karena fokusnya adalah inference dari model yang sudah dilatih.
- Untuk menambah produk baru, edit `database/beverages.json`, tambahkan `name`, `aliases`, `sugar_g`, dan `status`.
