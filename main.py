"""CLI: python main.py ijazah_001.jpg [--enhancement clahe] [--debug]"""
import argparse
import json

import cv2

from pipeline import ENHANCEMENT_METHODS, load_roi, rotate_image, run_pipeline


def select_roi_interactive(image_path, rotate=0, out="roi.json"):
    """Pilih area nomor & tanda tangan dengan mouse, simpan ke roi.json."""
    img = rotate_image(cv2.imread(image_path), rotate)
    # perkecil tampilan agar muat di layar (ROI disimpan relatif, jadi tetap akurat)
    max_w, max_h = 1000, 650
    scale = min(max_w / img.shape[1], max_h / img.shape[0], 1.0)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    result = {}
    for key, title in [("nomor", "Pilih AREA NOMOR lalu ENTER"),
                       ("tanda_tangan", "Pilih AREA TANDA TANGAN lalu ENTER")]:
        x, y, rw, rh = cv2.selectROI(title, img, showCrosshair=False)
        result[key] = [round(x / w, 4), round(y / h, 4), round(rw / w, 4), round(rh / h, 4)]
    cv2.destroyAllWindows()
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"ROI disimpan ke {out}: {result}")


def main():
    ap = argparse.ArgumentParser(description="Verifikasi ijazah: OCR nomor + deteksi tanda tangan")
    ap.add_argument("image", help="path citra ijazah, mis. ijazah_001.jpg")
    ap.add_argument("--enhancement", default="denoise_clahe", choices=ENHANCEMENT_METHODS)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="putar citra searah jarum jam sebelum diproses")
    ap.add_argument("--select-roi", action="store_true", help="pilih ROI manual dengan mouse")
    ap.add_argument("--debug", action="store_true", help="simpan citra hasil antara ke folder debug/")
    args = ap.parse_args()

    if args.select_roi:
        select_roi_interactive(args.image, args.rotate)

    res = run_pipeline(args.image, args.enhancement, load_roi(),
                       debug_dir="debug" if args.debug else None, rotate=args.rotate)
    print(f"Nomor Ijazah : {res['nomor_ijazah']}")
    print(f"Tanda Tangan : {res['tanda_tangan']}")
    if args.debug:
        print(f"[debug] rasio_tinta={res['ink_ratio']}  kontras={res['kontras']}")


if __name__ == "__main__":
    main()