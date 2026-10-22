"""Synthetic test scenes with exact known ground truth.

Test A: a 40 mm ArUco marker next to a 100 x 60 mm rectangle, rendered at
8 px/mm with noise and blur, like a decent webcam photo.

Test B: a bottle-like solid of revolution with elliptical cross-section
(60 x 110 x 42 mm), rendered from 12 angles (30 deg steps) at 6 px/mm.
Silhouettes are drawn analytically, so the ground truth is exact.
"""
import os

import cv2
import numpy as np

from objectlens.measure import get_dictionary

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ---------- Test A: measurement scene ----------
MEASURE_TRUTH = {"marker_mm": 40.0, "width_mm": 100.0, "height_mm": 60.0,
                 "px_per_mm": 8.0}


def render_measure_scene(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ppm = MEASURE_TRUTH["px_per_mm"]
    img = np.full((1200, 1600, 3), 235, np.uint8)
    # object: dark blue rounded rectangle, 100 x 60 mm
    ow, oh = int(100 * ppm), int(60 * ppm)
    cv2.rectangle(img, (500, 500), (500 + ow, 500 + oh), (120, 60, 30), -1)
    # marker: 40 mm, bottom-left of the object
    mpx = int(MEASURE_TRUTH["marker_mm"] * ppm)
    marker = cv2.aruco.generateImageMarker(get_dictionary(), 0, mpx)
    img[700:700 + mpx, 120:120 + mpx] = cv2.cvtColor(marker,
                                                    cv2.COLOR_GRAY2BGR)
    # make it less perfect: noise + slight blur
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 3, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imwrite(path, img)
    return path


# ---------- Test B: bottle visual hull ----------
BOTTLE_TRUTH = {"x_mm": 60.0, "y_mm": 110.0, "z_mm": 42.0,
                "px_per_mm": 6.0, "cx": 500, "cy": 820}


def _profile(y):
    """Radius scale factor along the bottle height y (mm from base)."""
    if y < 60:
        return 1.0
    if y < 90:  # smooth shoulder taper down to the neck
        t = (y - 60) / 30
        return 1.0 - 0.62 * (t * t * (3 - 2 * t))
    return 0.38


def _half_width(theta_deg, y):
    """Silhouette half width (mm) of the elliptical cross-section."""
    rx, rz = 30 * _profile(y), 21 * _profile(y)
    th = np.deg2rad(theta_deg)
    return np.sqrt((rx * np.cos(th)) ** 2 + (rz * np.sin(th)) ** 2)


def render_bottle_views(folder):
    os.makedirs(folder, exist_ok=True)
    ppm, cx, cy = BOTTLE_TRUTH["px_per_mm"], BOTTLE_TRUTH["cx"], \
        BOTTLE_TRUTH["cy"]
    heights = np.arange(0, BOTTLE_TRUTH["y_mm"] + 0.5, 0.5)
    paths = []
    for i in range(12):
        theta = i * 30
        img = np.full((900, 1000, 3), 240, np.uint8)
        left, right = [], []
        for y in heights:
            hw = _half_width(theta, y) * ppm
            row = int(round(cy - y * ppm))
            left.append((int(round(cx - hw)), row))
            right.append((int(round(cx + hw)), row))
        poly = np.array(left + right[::-1], np.int32)
        cv2.fillPoly(img, [poly], (80, 80, 90))
        if i == 0:
            # ArUco marker standing on the ground line, left of the bottle
            mpx = int(40 * ppm)
            marker = cv2.aruco.generateImageMarker(get_dictionary(), 0, mpx)
            img[cy - mpx:cy, 40:40 + mpx] = cv2.cvtColor(
                marker, cv2.COLOR_GRAY2BGR)
        p = os.path.join(folder, f"view_{i:02d}.png")
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    print(render_measure_scene(os.path.join(DATA, "measure_scene.png")))
    for p in render_bottle_views(os.path.join(DATA, "bottle")):
        print(p)
