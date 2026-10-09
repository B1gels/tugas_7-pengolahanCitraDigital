"""Jalankan pipeline pada semua citra di sebuah folder.

    python run_all.py --images data --rotate 90
"""
import argparse
from pathlib import Path

from pipeline import ENHANCEMENT_METHODS, load_roi, run_pipeline

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser(description="Proses semua citra ijazah dalam satu folder")
    ap.add_argument("--images", default="data")
    ap.add_argument("--enhancement", default="denoise_clahe", choices=ENHANCEMENT_METHODS)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    args = ap.parse_args()

    roi = load_roi()
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"{'File':<34}{'Nomor Ijazah':<20}{'Tanda Tangan':<14}{'Rasio':>7}{'Kontras':>9}")
    print("-" * 84)
    for p in files:
        r = run_pipeline(p, args.enhancement, roi, rotate=args.rotate)
        print(f"{p.name:<34}{r['nomor_ijazah']:<20}{r['tanda_tangan']:<14}"
              f"{r['ink_ratio']:>7}{r['kontras']:>9}")


if __name__ == "__main__":
    main()