# Mini Project Integrasi — Verifikasi Ijazah (OCR + Deteksi Tanda Tangan)

Tugas Pertemuan 7 — Pengolahan Citra Digital.
Prototype menerima citra ijazah dan menghasilkan **nomor ijazah** (OCR) serta status **tanda tangan kepala sekolah** (PRESENT / ABSENT).

```
Input : ijazah_001.jpg
Output:
Nomor Ijazah : DN-123456789
Tanda Tangan : PRESENT
```

## Pipeline

```
Citra Ijazah → Grayscale → Image Enhancement
        ├── Area Nomor        → Enhancement → OCR (Tesseract) → Nomor Ijazah
        └── Area Tanda Tangan → Thresholding → Morphology → Signature Detection
                                   ↓
                            Hasil Verifikasi
```

## Penjelasan Metode

| Tahap | Metode | Alasan |
|---|---|---|
| Grayscale | `cv2.cvtColor` BGR→GRAY | Mengurangi dimensi, warna tidak dibutuhkan untuk teks/tinta |
| Image Enhancement | None, Histogram Equalization, CLAHE, Unsharp Masking, Denoise (NLMeans)+CLAHE | Memperbaiki kontras/ketajaman sebelum OCR; dibandingkan lewat CER |
| ROI | Crop area relatif (`roi.json`) | Memisahkan area nomor dan tanda tangan |
| OCR | Tesseract (`--psm 7`, whitelist `A-Z0-9-/`), resize 2x, regex pola nomor | Satu baris teks; whitelist menekan salah baca karakter |
| Thresholding | Gaussian blur + Otsu (invers) | Memisahkan tinta tanda tangan dari kertas secara otomatis |
| Morphology | Opening (2×2) lalu Closing (7×7) | Opening membuang noise kecil, closing menyambung goresan putus |
| Signature Detection | Connected components (area ≥ 80 px), rasio piksel tinta ≥ 1% → `PRESENT` | Tanda tangan berupa goresan besar yang terhubung; area kosong hanya berisi noise |

## Cara Menjalankan

1. Install Tesseract OCR
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki (lalu tambahkan ke PATH, atau set `pytesseract.pytesseract.tesseract_cmd`)
   - Linux: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`
2. Install dependensi Python
   ```bash
   pip install -r requirements.txt
   ```
3. (Opsional, disarankan) Tentukan area nomor & tanda tangan dengan mouse. Hasil disimpan di `roi.json`.
   ```bash
   python main.py data/ijazah_001.jpg --select-roi
   ```
4. Jalankan prototype
   ```bash
   python main.py data/ijazah_001.jpg --enhancement clahe --debug
   ```
   `--debug` menyimpan citra hasil antara ke folder `debug/`.

> **Catatan orientasi:** citra ijazah dari dosen tersimpan dalam posisi miring 90°. Gunakan `--rotate 90`
> (searah jarum jam) di semua perintah (`--select-roi`, `main.py`, `evaluate.py`) agar teks terbaca tegak.

## Evaluasi Metode Enhancement (CER)

1. Taruh citra ijazah di `data/` dan buat `data/ground_truth.csv`:
   ```csv
   filename,nomor_ijazah
   ijazah_001.jpg,DN-123456789
   ```
2. Jalankan (atau, karena semua citra adalah ijazah yang sama, cukup beri satu nomor tanpa CSV):
   ```bash
   python evaluate.py --images data/ --truth data/ground_truth.csv --rotate 90
   python evaluate.py --images data/ --number "15 - 006559" --rotate 90
   ```

CER = (S + D + I) / N, dengan S = substitusi, D = penghapusan, I = penyisipan, N = jumlah karakter referensi. Semakin kecil semakin baik.

### Hasil (isi setelah menjalankan evaluate.py)

| Metode | Rata-rata CER |
|---|---|
| none | ... |
| hist_eq | ... |
| clahe | ... |
| sharpen | ... |
| denoise_clahe | ... |

**Kesimpulan:** metode `...` paling efektif karena memiliki CER terendah (`...`). *(Jelaskan mengapa, mis. CLAHE menaikkan kontras lokal tanpa memperbesar noise, sedangkan histogram equalization global bisa merusak area dengan pencahayaan tidak merata.)*

## Struktur Repo

```
├── main.py          # CLI prototype
├── pipeline.py      # grayscale, enhancement, OCR, deteksi tanda tangan
├── evaluate.py      # perhitungan CER per metode
├── roi.json         # koordinat area (dibuat via --select-roi)
├── data/            # citra ijazah + ground_truth.csv
└── requirements.txt
```

## Keterbatasan

- Posisi area nomor/tanda tangan bergantung layout ijazah (atur lewat `roi.json`).
- Deteksi tanda tangan berbasis rasio tinta; stempel atau teks cetak di dalam ROI bisa menyebabkan false positive. Sesuaikan `min_ink_ratio`.
