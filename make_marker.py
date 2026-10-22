#!/usr/bin/env python3
"""Write a printable ArUco marker PNG at an exact physical size.

The PNG is rendered at 300 DPI, so if it is printed at 100% scale the black
square measures exactly --size-mm millimetres. Measure it with a ruler after
printing and pass the real size to the scanner if your printer scaled it.
"""
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from objectlens.measure import get_dictionary

DPI = 300
PX_PER_MM = DPI / 25.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--size-mm", type=float, default=40.0)
    ap.add_argument("--out", default="marker.png")
    args = ap.parse_args()

    marker_px = int(round(args.size_mm * PX_PER_MM))
    quiet_px = int(round(10 * PX_PER_MM))          # 10 mm white border
    label_h = int(round(12 * PX_PER_MM))
    img = np.full((marker_px + 2 * quiet_px + label_h,
                   marker_px + 2 * quiet_px), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(get_dictionary(), args.id,
                                           marker_px)
    img[quiet_px:quiet_px + marker_px, quiet_px:quiet_px + marker_px] = marker

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(6 * PX_PER_MM))
    except OSError:
        font = ImageFont.load_default()
    text = f"ArUco id {args.id} - {args.size_mm:.0f} mm - print at 100%"
    draw.text((quiet_px, marker_px + 2 * quiet_px + 8), text, fill=0, font=font)
    pil.save(args.out, dpi=(DPI, DPI))
    print(f"wrote {args.out} ({pil.size[0]}x{pil.size[1]} px @ {DPI} DPI, "
          f"marker = {args.size_mm:.0f} mm)")


if __name__ == "__main__":
    main()
