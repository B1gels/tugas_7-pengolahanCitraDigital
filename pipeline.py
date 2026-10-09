"""Pipeline verifikasi ijazah.

Citra Ijazah -> Grayscale -> Image Enhancement -> (Area Nomor | Area Tanda Tangan)
  Area Nomor        : Enhancement -> OCR -> Nomor Ijazah
  Area Tanda Tangan : Thresholding -> Morphology -> Signature Detection
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# Lokasi ROI dalam koordinat relatif (x, y, w, h) terhadap ukuran citra (0..1).
# Sesuaikan dengan layout ijazah Anda, atau buat otomatis dengan: python main.py --select-roi
DEFAULT_ROI = {
    "nomor": [0.60, 0.05, 0.38, 0.10],
    "tanda_tangan": [0.55, 0.70, 0.40, 0.22],
}

ENHANCEMENT_METHODS = ["none", "hist_eq", "clahe", "sharpen", "denoise_clahe"]

# Parameter deteksi tanda tangan
MIN_INK_RATIO = 0.01     # minimal rasio piksel tinta terhadap luas ROI
MIN_COMP_AREA = 80       # komponen terhubung lebih kecil dari ini dianggap noise
MIN_CONTRAST = 30        # minimal selisih gelap (tinta) vs kertas; di bawahnya = area kosong


# ---------------------------------------------------------------- utilitas
def load_roi(path="roi.json"):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return DEFAULT_ROI


def crop_relative(img, roi):
    h, w = img.shape[:2]
    x, y, rw, rh = roi
    x0, y0 = int(x * w), int(y * h)
    return img[y0:y0 + int(rh * h), x0:x0 + int(rw * w)]


def rotate_image(img, rotate=0):
    """Putar citra searah jarum jam: 0, 90, 180, atau 270 derajat."""
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(img, codes[rotate]) if rotate in codes else img


# ---------------------------------------------------------- 1. grayscale
def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------- 2. enhancement
def enhance(gray, method="clahe"):
    """Meningkatkan kualitas citra grayscale dengan metode yang dipilih."""
    if method == "none":
        return gray
    if method == "hist_eq":
        return cv2.equalizeHist(gray)
    if method == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if method == "sharpen":  # unsharp masking
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        return cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    if method == "denoise_clahe":
        den = cv2.fastNlMeansDenoising(gray, None, h=10)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(den)
    raise ValueError(f"Metode enhancement tidak dikenal: {method}")


# ---------------------------------------------------------- 3. OCR nomor
NUMBER_PATTERN = re.compile(r"\d+\s*-\s*\d{4,}|[A-Z]{1,3}\s*-\s*\d{4,}")


def ocr_raw(roi_gray):
    """OCR mentah dari area nomor (dipakai juga untuk hitung CER)."""
    # perbesar agar teks kecil lebih terbaca
    roi = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/"
    return pytesseract.image_to_string(roi, config=cfg).strip()


def extract_number(text):
    """Ambil pola nomor ijazah (mis. DN-123456789) dari teks hasil OCR."""
    text = text.upper().replace(" ", "")
    m = NUMBER_PATTERN.search(text)
    return m.group(0).replace(" ", "") if m else text


def read_diploma_number(gray_enhanced, roi_nomor):
    area = crop_relative(gray_enhanced, roi_nomor)
    return extract_number(ocr_raw(area))


# ---------------------------------------------- 4. deteksi tanda tangan
def signature_contrast(gray_enhanced, roi_ttd):
    """Selisih kecerahan kertas (median) dan piksel tergelap pada ROI tanda tangan."""
    area = crop_relative(gray_enhanced, roi_ttd)
    blur = cv2.GaussianBlur(area, (5, 5), 0)
    return float(np.median(blur) - np.percentile(blur, 0.5))


def detect_signature(gray_enhanced, roi_ttd, min_ink_ratio=MIN_INK_RATIO,
                     min_comp_area=MIN_COMP_AREA, min_contrast=MIN_CONTRAST):
    """Thresholding -> Morphology -> Signature Detection.

    Return: (status, ink_ratio, mask)
    """
    area = crop_relative(gray_enhanced, roi_ttd)
    blur = cv2.GaussianBlur(area, (5, 5), 0)

    # Otsu selalu menemukan ambang walau area hanya berisi noise, sehingga area
    # kosong bisa salah dianggap tanda tangan. Pastikan ada tinta yang cukup gelap.
    if float(np.median(blur) - np.percentile(blur, 0.5)) < min_contrast:
        return "ABSENT", 0.0, np.zeros_like(blur)

    # Thresholding (Otsu, invers: tinta = putih)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphology: open (buang noise kecil) lalu close (sambung goresan putus)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k_close)

    # Signature detection: simpan komponen terhubung yang cukup besar
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    mask = np.zeros_like(clean)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_comp_area:
            mask[labels == i] = 255

    ink_ratio = float(np.count_nonzero(mask)) / mask.size
    status = "PRESENT" if ink_ratio >= min_ink_ratio else "ABSENT"
    return status, ink_ratio, mask


# --------------------------------------------------------------- pipeline
def run_pipeline(image_path, enhancement="clahe", roi=None, debug_dir=None, rotate=0):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Gagal membaca citra: {image_path}")
    img = rotate_image(img, rotate)
    roi = roi or load_roi()

    gray = to_gray(img)
    enhanced = enhance(gray, enhancement)

    nomor = read_diploma_number(enhanced, roi["nomor"])
    ttd, ratio, mask = detect_signature(enhanced, roi["tanda_tangan"])
    contrast = signature_contrast(enhanced, roi["tanda_tangan"])

    if debug_dir:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        cv2.imwrite(str(d / f"{stem}_enhanced.png"), enhanced)
        cv2.imwrite(str(d / f"{stem}_nomor_roi.png"), crop_relative(enhanced, roi["nomor"]))
        cv2.imwrite(str(d / f"{stem}_ttd_mask.png"), mask)

    return {"nomor_ijazah": nomor, "tanda_tangan": ttd, "ink_ratio": round(ratio, 4),
            "kontras": round(contrast, 1)}