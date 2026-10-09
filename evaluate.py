"""Evaluasi CER (Character Error Rate) untuk setiap metode enhancement.

Semua citra ijazah sama (hanya beda degradasi), jadi cukup beri satu nomor:
    python evaluate.py --images data --number "571012022000056" --rotate 90

Atau pakai CSV ground truth (filename,nomor_ijazah):
    python evaluate.py --images data --truth data/ground_truth.csv --rotate 90

Metrik utama : CER (lebih kecil lebih baik)
Pemutus seri : rata-rata confidence OCR Tesseract (lebih besar lebih baik)
"""
import argparse
import csv
from pathlib import Path

import cv2
import pytesseract

from pipeline import (ENHANCEMENT_METHODS, crop_relative, enhance, extract_number,
                      load_roi, ocr_raw, rotate_image, to_gray)

IMG_EXT = {".jpg", ".jpeg", ".png"}


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference, hypothesis):
    """CER = (S + D + I) / N, dihitung lewat edit distance."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def ocr_confidence(roi_gray):
    """Rata-rata confidence (0-100) Tesseract pada area nomor."""
    roi = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789-"
    data = pytesseract.image_to_data(roi, config=cfg, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, t in zip(data["conf"], data["text"]) if t.strip() and float(c) >= 0]
    return sum(confs) / len(confs) if confs else 0.0


def norm(s):
    return s.upper().replace(" ", "").strip()


def load_truth(args):
    if args.number:
        files = sorted(p.name for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXT)
        return [{"filename": f, "nomor_ijazah": args.number} for f in files]
    with open(args.truth, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data")
    ap.add_argument("--truth", default="data/ground_truth.csv")
    ap.add_argument("--number", help="nomor ijazah yang benar (dipakai untuk semua citra)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--out", default="results_cer.csv")
    args = ap.parse_args()

    roi = load_roi()
    truth = load_truth(args)

    rows, summary = [], {}
    for method in ENHANCEMENT_METHODS:
        cers, confs = [], []
        for t in truth:
            img = cv2.imread(str(Path(args.images) / t["filename"]))
            if img is None:
                print(f"[skip] {t['filename']} tidak ditemukan")
                continue
            gray = to_gray(rotate_image(img, args.rotate))
            area = crop_relative(enhance(gray, method), roi["nomor"])
            pred = norm(extract_number(ocr_raw(area)))
            conf = ocr_confidence(area)
            ref = norm(t["nomor_ijazah"])
            score = cer(ref, pred)
            cers.append(score)
            confs.append(conf)
            rows.append([method, t["filename"], ref, pred, round(score, 4), round(conf, 1)])
        if cers:
            summary[method] = (sum(cers) / len(cers), sum(confs) / len(confs))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "filename", "reference", "prediction", "cer", "confidence"])
        w.writerows(rows)

    print("\nPerbandingan metode enhancement")
    print(f"{'metode':<16}{'CER':>9}{'confidence':>13}")
    print("-" * 38)
    # urut: CER terkecil dulu, lalu confidence terbesar
    ranked = sorted(summary.items(), key=lambda kv: (round(kv[1][0], 4), -kv[1][1]))
    for m, (c, cf) in ranked:
        print(f"{m:<16}{c:>9.4f}{cf:>13.1f}")

    if ranked:
        best_cer = round(ranked[0][1][0], 4)
        tied = [m for m, (c, _) in ranked if round(c, 4) == best_cer]
        print(f"\nCER terendah : {best_cer:.4f} -> {', '.join(tied)}")
        if len(tied) > 1:
            print(f"Seri pada CER; berdasarkan confidence tertinggi: {ranked[0][0]} "
                  f"({ranked[0][1][1]:.1f})")
        else:
            print(f"Metode terbaik: {ranked[0][0]}")
        print(f"Detail per citra: {args.out}")


if __name__ == "__main__":
    main()