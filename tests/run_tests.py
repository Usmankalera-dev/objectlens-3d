#!/usr/bin/env python3
"""End-to-end tests of ObjectLens 3D on synthetic scenes with ground truth.

Run:  python tests/run_tests.py
Prints the recovered values next to the truth and the absolute errors.
Everything is also exercised through the real CLI (scan.py) as a user runs it.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import make_test_data as td  # noqa: E402

PY = sys.executable
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def run_cli(*argv):
    cmd = [PY, os.path.join(ROOT, "scan.py"), *argv]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise SystemExit(f"CLI failed: {' '.join(argv)}")
    return res.stdout


def err(name, got, truth):
    e = abs(got - truth)
    print(f"  {name:<12} got {got:>8.2f} mm | truth {truth:>6.1f} mm | "
          f"error {e:5.2f} mm ({100 * e / truth:.1f}%)")
    return e


def main():
    scene = td.render_measure_scene(os.path.join(td.DATA, "measure_scene.png"))
    views = td.render_bottle_views(os.path.join(td.DATA, "bottle"))

    print("== Test A: single-photo measurement (ArUco scale) ==")
    out = run_cli("measure", scene, "--marker-mm", "40",
                  "--annotate", os.path.join(OUT, "measure_annotated.png"))
    rep = json.loads(out[out.index("{"):out.rindex("}") + 1])
    t = td.MEASURE_TRUTH
    err("width", rep["width_mm"], t["width_mm"])
    err("height", rep["height_mm"], t["height_mm"])
    print(f"  px_per_mm    got {rep['px_per_mm']:.3f} | truth "
          f"{t['px_per_mm']:.3f} | error "
          f"{abs(rep['px_per_mm'] - t['px_per_mm']):.3f}")

    print("== Test B: visual hull from 12 views ==")
    report_path = os.path.join(OUT, "sample-report.json")
    ply_path = os.path.join(OUT, "sample.ply")
    out = run_cli("hull", os.path.join(td.DATA, "bottle", "view_*.png"),
                  "--step", "30", "--marker-mm", "40", "--grid", "96",
                  "--mesh", "--out", ply_path, "--report", report_path)
    rep = json.load(open(report_path))
    dims = rep["dimensions_mm"]
    t = td.BOTTLE_TRUTH
    err("width (x)", dims["width_x"], t["x_mm"])
    err("height (y)", dims["height_y"], t["y_mm"])
    err("depth (z)", dims["depth_z"], t["z_mm"])
    print(f"  voxels {rep['voxel_count']}, surface points "
          f"{rep['surface_points']}, mesh faces {rep.get('mesh_faces')}")
    print("ALL TESTS RAN OK")


if __name__ == "__main__":
    main()
