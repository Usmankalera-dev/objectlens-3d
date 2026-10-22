"""ArUco marker detection and true-size object measurement.

Idea: an ArUco marker of known physical size (mm) sits in the same plane as
the object. The marker tells us how many pixels equal one millimetre at that
plane, so a pixel bounding box of the object becomes a real measurement.
"""
import cv2
import numpy as np

DICT_NAME = cv2.aruco.DICT_4X4_50


def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(DICT_NAME)


def detect_markers(image):
    """Return {marker_id: corners(4x2 float32)} for all markers in image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = cv2.aruco.ArucoDetector(get_dictionary(),
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    found = {}
    if ids is not None:
        for marker_id, c in zip(ids.flatten(), corners):
            found[int(marker_id)] = c.reshape(4, 2).astype(np.float32)
    return found


def marker_scale_px_per_mm(corners, marker_mm):
    """Mean edge length of the marker in px divided by its real size in mm."""
    pts = corners.reshape(4, 2)
    edges = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    return float(np.mean(edges) / marker_mm)


def estimate_background_color(image):
    """Background colour = median of the four image corners patches."""
    h, w = image.shape[:2]
    p = max(4, min(h, w) // 20)
    patches = [image[:p, :p], image[:p, -p:], image[-p:, :p], image[-p:, -p:]]
    return np.median(np.stack([pt.reshape(-1, image.shape[2]).mean(axis=0)
                               for pt in patches]), axis=0)


def segment_object_mask(image, exclude_boxes=(), dist_thresh=30):
    """Segment the foreground object on a plain background.

    Foreground = pixels whose colour differs from the background colour
    (estimated from the image corners). Marker areas can be excluded with
    boxes (x, y, w, h). Returns the cleaned mask of the largest blob.
    """
    imgf = image.astype(np.float32)
    bg = estimate_background_color(image)
    dist = np.linalg.norm(imgf - bg.reshape(1, 1, -1), axis=2)
    mask = (dist > dist_thresh).astype(np.uint8) * 255
    for (x, y, w, h) in exclude_boxes:
        pad = 6
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
        mask[y0:y1, x0:x1] = 0
    # clean small noise, close small holes
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


def yolo_object_box(image):
    """Optional fallback: largest YOLO detection box (needs ultralytics)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    model = YOLO("yolov8n.pt")
    result = model(image, verbose=False)[0]
    best, best_area = None, 0
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best, best_area = (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), area
    return best


def measure_object(image, marker_mm=40.0, marker_id=None, use_yolo=False):
    """Measure the object next to an ArUco marker. Returns a report dict."""
    markers = detect_markers(image)
    if not markers:
        return {"ok": False, "error": "no ArUco marker found"}
    mid = marker_id if marker_id in markers else sorted(markers)[0]
    corners = markers[mid]
    px_per_mm = marker_scale_px_per_mm(corners, marker_mm)

    x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
    if use_yolo:
        box = yolo_object_box(image)
        if box is None:
            return {"ok": False, "error": "YOLO found no object"}
    else:
        mask = segment_object_mask(image, exclude_boxes=[(x, y, w, h)])
        box = cv2.boundingRect(mask) if mask.any() else None
        if box is None:
            return {"ok": False, "error": "no object segmented"}
    ox, oy, ow, oh = [int(v) for v in box]
    return {
        "ok": True,
        "marker_id": int(mid),
        "marker_mm": float(marker_mm),
        "px_per_mm": round(px_per_mm, 4),
        "marker_box_px": [x, y, w, h],
        "object_box_px": [ox, oy, ow, oh],
        "width_mm": round(ow / px_per_mm, 2),
        "height_mm": round(oh / px_per_mm, 2),
    }
