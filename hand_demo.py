#!/usr/bin/env python3
"""Hand tracking proof module - MediaPipe HandLandmarker (Tasks API).

Draws the 21 hand landmarks and their connections on an image or on every
frame of a video. The model file (models/hand_landmarker.task) is downloaded
automatically on first run.

    python hand_demo.py photo.jpg
    python hand_demo.py clip.mp4 --out outputs/hands.mp4
"""
import argparse
import os
import urllib.request

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "hand_landmarker.task")

# connections between the 21 hand landmarks (palm + 5 fingers)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm edge
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("downloading hand model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        num_hands=2, min_hand_detection_confidence=0.4)
    return vision.HandLandmarker.create_from_options(options)


def draw_hands(image_bgr, result):
    """Draw landmarks for every detected hand; returns hands drawn."""
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    for hand in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (0, 255, 160), 2)
        for p in pts:
            cv2.circle(out, p, 4, (255, 120, 0), -1)
    return out, len(result.hand_landmarks)


def run_on_image(landmarker, path, out_path):
    import mediapipe as mp
    image = cv2.imread(path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=rgb))
    drawn, n = draw_hands(image, result)
    cv2.imwrite(out_path, drawn)
    return n


def run_on_video(landmarker, path, out_path):
    import mediapipe as mp
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    frames, frames_with_hands, ts = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, ts)
        ts += int(1000 / fps)
        drawn, n = draw_hands(frame, result)
        frames_with_hands += n > 0
        writer.write(drawn)
        frames += 1
    cap.release()
    writer.release()
    return frames, frames_with_hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="image or video file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    is_video = args.input.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    out = args.out or ("outputs/hands.mp4" if is_video
                       else "outputs/hands.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    landmarker = make_landmarker()
    if is_video:
        frames, with_hands = run_on_video(landmarker, args.input, out)
        print(f"video: {frames} frames, hands visible in {with_hands}")
    else:
        n = run_on_image(landmarker, args.input, out)
        print(f"image: {n} hand(s) detected")
    print(f"annotated -> {out}")


if __name__ == "__main__":
    main()
