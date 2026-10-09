"""Membuat citra uji TANPA tanda tangan (simulasi) dari citra yang ada.

Area tanda tangan (sesuai roi.json) ditimpa dengan warna kertas, sehingga
sistem seharusnya menjawab ABSENT.

    python make_absent_sample.py data/01_HighQuality_Enhanced.jpg --rotate 90
    python main.py docs/sample_absent.jpg --rotate 90
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from pipeline import load_roi, rotate_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--out", default="docs/sample_absent.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    img = rotate_image(img, args.rotate)       # samakan orientasi dengan ROI
    h, w = img.shape[:2]
    x, y, rw, rh = load_roi()["tanda_tangan"]
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = x0 + int(rw * w), y0 + int(rh * h)

    paper = np.median(img.reshape(-1, 3), axis=0).astype(np.uint8)  # warna kertas dominan
    img[y0:y1, x0:x1] = paper
    img = rotate_image(img, (360 - args.rotate) % 360)              # kembalikan orientasi asli

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"Tersimpan: {args.out}")


if __name__ == "__main__":
    main()