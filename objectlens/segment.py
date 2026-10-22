"""Silhouette segmentation for multi-view reconstruction."""
import cv2
import numpy as np

from .measure import estimate_background_color


def segment_by_background(image, dist_thresh=30):
    """Mask of pixels that differ from the plain background colour."""
    imgf = image.astype(np.float32)
    bg = estimate_background_color(image)
    dist = np.linalg.norm(imgf - bg.reshape(1, 1, -1), axis=2)
    mask = (dist > dist_thresh).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask * 0
    best = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [best], -1, 255, -1)
    return out


def segment_by_bgdiff(frame, background, thresh=35):
    """Mask by absolute difference against an empty-scene background frame."""
    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def segment_grabcut(image, iterations=5):
    """GrabCut with a border-inset rectangle as the initial guess."""
    h, w = image.shape[:2]
    margin_x, margin_y = w // 12, h // 12
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd, fgd, iterations,
                cv2.GC_INIT_WITH_RECT)
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
    return out.astype(np.uint8)


METHODS = {
    "background": segment_by_background,
    "grabcut": segment_grabcut,
}


def segment(image, method="background"):
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, use {sorted(METHODS)}")
    return METHODS[method](image)
