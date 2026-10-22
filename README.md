# ObjectLens 3D — webcam object scanner with *true* millimetre measurements

**🔴 Live demo: https://usmankalera-dev.github.io/objectlens-3d/** (open it, allow the webcam, hold an object up)

Most webcam "3D scanners" guess the object's size. ObjectLens 3D calibrates
every scan against a printed **ArUco marker of known size**, so measurements
come out in real millimetres. Rotate the object, shoot ~12 views, and the
pipeline carves a **3D visual hull** (point cloud or mesh, PLY) whose
dimensions are measured, not estimated. MediaPipe hand tracking is included
as a live module, and the static demo site runs hands + object detection
right in the browser.

**Stack:** Python · OpenCV (ArUco, GrabCut) · NumPy voxel carving ·
scikit-image marching cubes · MediaPipe HandLandmarker · three.js ·
TensorFlow.js COCO-SSD (browser demo)

## Layout

```
objectlens/measure.py   ArUco detection, px-per-mm scale, object measurement
objectlens/segment.py   silhouette segmentation (background colour / GrabCut / bg-diff)
objectlens/hull.py      voxel visual-hull carving, dimensions, PLY export
scan.py                 CLI: `measure` and `hull` subcommands
make_marker.py          printable ArUco marker PNG (300 DPI, exact mm)
hand_demo.py            MediaPipe hand landmarks on an image or video
tests/                  synthetic scenes with exact ground truth + runner
site/                   static live-demo website (no build step)
models/                 hand_landmarker.task (auto-downloaded on first run)
```

## Quick start

```bash
pip install -r requirements.txt

# 1. print the calibration marker (40 mm square, print at 100%, check with a ruler)
python make_marker.py --size-mm 40 --out marker.png

# 2. measure an object in a single photo (marker beside the object, same plane)
python scan.py measure photo.jpg --marker-mm 40 --annotate out.png

# 3. reconstruct in 3D: object rotated ~30° between shots, plain background,
#    marker visible in at least one view
python scan.py hull view_*.png --step 30 --marker-mm 40 --mesh \
    --out outputs/sample.ply --report outputs/report.json

# 4. hand tracking proof module
python hand_demo.py photo_or_clip.mp4
```

Every scan writes a JSON report: dimensions (mm), voxel count, surface
points, views used, and the px-per-mm scale that was measured from the marker.

## Demo results — measured against exact ground truth

No webcam exists on the build machine, so the pipeline was verified on
**synthetic scenes rendered with exact known dimensions** (tests in
`tests/`, re-run any time with `python tests/run_tests.py`):

| Test | Quantity | Recovered | Ground truth | Error |
|---|---|---|---|---|
| A · single photo | object width | 100.69 mm | 100 mm | **0.69 mm (0.7%)** |
| A · single photo | object height | 60.56 mm | 60 mm | **0.56 mm (0.9%)** |
| A · single photo | scale | 7.975 px/mm | 8.000 px/mm | 0.3% |
| A · marker rotated 15° | scale | 7.984 px/mm | 8.000 px/mm | 0.21% |
| B · visual hull (12 views) | width (X) | 60.87 mm | 60 mm | **0.87 mm (1.4%)** |
| B · visual hull (12 views) | height (Y) | 111.15 mm | 110 mm | **1.15 mm (1.0%)** |
| B · visual hull (12 views) | depth (Z) | 42.34 mm | 42 mm | **0.34 mm (0.8%)** |

Test B object: a bottle-like solid of revolution with elliptical
cross-section, 60 × 110 × 42 mm; hull = 70,516 voxels at 96³,
23,178-face marching-cubes mesh. Hand tracking: `hand_demo.py` was run on a
real photograph of two open hands (Wikimedia Commons) — **both hands
detected, all 21 landmarks per hand drawn** (`outputs/hands_annotated.png`).

These are **synthetic-scene numbers**. They prove the geometry and scale
math end-to-end; a real webcam, real lighting and a real printed marker
will add error on top (see limitations).

## Live demo site

`site/` is fully static — serve it with anything:

```bash
cd site && python -m http.server 8000
```

- **Live scanner:** webcam with MediaPipe hand landmarks and COCO-SSD object
  boxes drawn in real time (needs a camera + internet for the CDN models).
- **Sample scan:** the PLY from Test B in a three.js viewer (rotate/zoom)
  with its measured dimensions, straight from the scan report JSON.
- On a machine without a camera the live panel shows a placeholder — that is
  expected; the sample scan is the pipeline's real output.

## Honest limitations

- **Single-plane scale.** The marker calibrates the plane it lies in. If the
  object sits much nearer/farther than the marker, sizes scale by the depth
  ratio. Keep marker and object in the same plane.
- **A visual hull is an outer shell.** Silhouette carving can never recover
  concavities — dents, holes, threads, the inside of a cup. It is not a
  LiDAR/depth scan.
- **Segmentation needs cooperation.** Plain, contrasting background and even
  light. Busy backgrounds, shadows and same-colour objects break masks, and
  a bad mask becomes a bad hull.
- **Occlusions stay unknown.** Anything hidden in every view (e.g. the very
  bottom face) cannot be reconstructed.
- **Print scale matters.** If your printer scales the marker, measure the
  printed square and pass the real size with `--marker-mm`.
- **Synthetic tests ≠ real-world accuracy.** The table above is exact-truth
  rendered scenes, the honest best case. Expect a few mm more error with a
  real camera.

## Credits

Built by **Usman Kalera** (github.com/Usmankalera-dev). Test photograph of
hands: Wikimedia Commons ("Human-Hands-Front-Back.jpg").
