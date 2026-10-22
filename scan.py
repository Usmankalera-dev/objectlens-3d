#!/usr/bin/env python3
"""ObjectLens 3D command line.

Measure one photo (ArUco marker + object):
    python scan.py measure photo.jpg --marker-mm 40

Reconstruct from rotated views (object turned ~--step degrees between shots,
marker visible in at least the first view for scale):
    python scan.py hull view_*.png --step 30 --marker-mm 40 \
        --out outputs/sample.ply --report outputs/report.json
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objectlens import hull as hull_mod          # noqa: E402
from objectlens import segment as segment_mod    # noqa: E402
from objectlens.measure import (detect_markers,  # noqa: E402
                                marker_scale_px_per_mm, measure_object)


def cmd_measure(args):
    image = cv2.imread(args.image)
    if image is None:
        print(f"cannot read {args.image}")
        return 1
    report = measure_object(image, marker_mm=args.marker_mm,
                            marker_id=args.marker_id, use_yolo=args.yolo)
    print(json.dumps(report, indent=2))
    if args.annotate and report.get("ok"):
        x, y, w, h = report["object_box_px"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 160), 3)
        label = f"{report['width_mm']:.1f} x {report['height_mm']:.1f} mm"
        cv2.putText(image, label, (x, max(30, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 160), 2)
        cv2.imwrite(args.annotate, image)
        print(f"annotated image -> {args.annotate}")
    return 0 if report.get("ok") else 2


def _scale_and_axis(images, args):
    """Find px-per-mm and the rotation axis column / ground row.

    Scale comes from the ArUco marker in the first view that has one.
    Axis column defaults to the centre of the first silhouette (the marker
    only gives scale; it does not sit on the rotation axis). Ground row
    defaults to the bottom of the first silhouette.
    """
    px_per_mm = args.px_per_mm
    for img in images:
        markers = detect_markers(img)
        if markers:
            mid = sorted(markers)[0]
            if px_per_mm is None:
                px_per_mm = marker_scale_px_per_mm(markers[mid],
                                                   args.marker_mm)
            break
    if px_per_mm is None:
        raise SystemExit("no ArUco marker found and --px-per-mm not given")
    mask = segment_mod.segment(images[0], args.seg)
    ys, xs = np.nonzero(mask)
    cx = args.cx if args.cx is not None else float((xs.min() + xs.max()) / 2)
    cy = args.cy if args.cy is not None else float(ys.max())
    return px_per_mm, cx, cy


def cmd_hull(args):
    paths = sorted(p for pat in args.images for p in glob.glob(pat))
    if not paths:
        print("no images matched")
        return 1
    images = [cv2.imread(p) for p in paths]
    px_per_mm, cx, cy = _scale_and_axis(images, args)
    masks = [segment_mod.segment(img, args.seg) for img in images]
    angles = [i * args.step for i in range(len(images))]
    voxels, origin, voxel_mm = hull_mod.carve(
        masks, angles, px_per_mm, cx, cy, grid=args.grid,
        extent_mm=args.extent_mm)
    dims = hull_mod.hull_dimensions(voxels, voxel_mm)
    points, _ = hull_mod.surface_points(voxels, origin, voxel_mm)
    report = {
        "mode": "visual_hull",
        "views_used": len(images),
        "angles_deg": angles,
        "px_per_mm": round(float(px_per_mm), 4),
        "axis_cx_px": round(float(cx), 1),
        "ground_cy_px": round(float(cy), 1),
        "grid": args.grid,
        "voxel_mm": round(float(voxel_mm), 3),
        "voxel_count": int(voxels.sum()),
        "surface_points": int(len(points)),
        "dimensions_mm": {"width_x": dims["x_mm"], "height_y": dims["y_mm"],
                          "depth_z": dims["z_mm"]},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.mesh:
        verts, faces = hull_mod.mesh_from_voxels(voxels, origin, voxel_mm)
        hull_mod.write_ply_mesh(args.out, verts, faces)
        report["mesh_vertices"] = int(len(verts))
        report["mesh_faces"] = int(len(faces))
    else:
        hull_mod.write_ply_points(args.out, points)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"PLY -> {args.out}\nreport -> {args.report}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="ObjectLens 3D scanner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="measure an object next to a marker")
    m.add_argument("image")
    m.add_argument("--marker-mm", type=float, default=40.0)
    m.add_argument("--marker-id", type=int, default=None)
    m.add_argument("--yolo", action="store_true",
                   help="use YOLO box instead of contour segmentation")
    m.add_argument("--annotate", metavar="OUT.png")
    m.set_defaults(fn=cmd_measure)

    h = sub.add_parser("hull", help="visual hull from rotated views")
    h.add_argument("images", nargs="+", help="view images (globs allowed)")
    h.add_argument("--step", type=float, default=30.0,
                   help="degrees the object was rotated between views")
    h.add_argument("--marker-mm", type=float, default=40.0)
    h.add_argument("--px-per-mm", type=float, default=None)
    h.add_argument("--cx", type=float, default=None,
                   help="rotation axis column in px (default: marker/image)")
    h.add_argument("--cy", type=float, default=None,
                   help="ground row in px (default: bottom of silhouette)")
    h.add_argument("--grid", type=int, default=96)
    h.add_argument("--extent-mm", type=float, default=None)
    h.add_argument("--seg", default="background",
                   choices=["background", "grabcut"])
    h.add_argument("--mesh", action="store_true",
                   help="export a marching-cubes mesh instead of points")
    h.add_argument("--out", default="outputs/sample.ply")
    h.add_argument("--report", default="outputs/report.json")
    h.set_defaults(fn=cmd_hull)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
