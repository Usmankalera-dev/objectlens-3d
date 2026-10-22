"""Voxel visual-hull carving from silhouettes, plus PLY export.

Geometry convention (shared by the renderer, the CLI and the tests):
- The object stands on a turntable and rotates about the vertical Y axis.
- The camera is fixed, roughly at object height, looking horizontally.
- In every view image, the rotation axis is at column `cx` and the ground
  (base of the object, y = 0) is at row `cy`. Up in the image = +Y.
- Projection is treated as orthographic with scale `px_per_mm`, which the
  ArUco marker provides for real scans. View angle theta rotates the object
  around Y:  u = x*cos(theta) + z*sin(theta),  v = -y (in mm, before scale).
"""
import numpy as np


def _project(pts_mm, theta_deg, px_per_mm, cx, cy):
    """Project (N,3) mm points to image (u, v) pixel coordinates."""
    th = np.deg2rad(theta_deg)
    u = (pts_mm[:, 0] * np.cos(th) + pts_mm[:, 2] * np.sin(th)) * px_per_mm + cx
    v = cy - pts_mm[:, 1] * px_per_mm
    return u, v


def carve(masks, angles_deg, px_per_mm, cx, cy, grid=96, extent_mm=None):
    """Carve a voxel visual hull. Returns (voxels bool array, origin, voxel_mm).

    The volume is a box of side `extent_mm`: x and z run from -extent/2 to
    +extent/2 around the rotation axis, y runs from 0 (ground) to extent.
    If extent_mm is None it is estimated from the silhouettes.
    """
    masks = [np.asarray(m) > 0 for m in masks]
    h_img, w_img = masks[0].shape
    if extent_mm is None:
        # widest silhouette half-width in mm + margin -> cube half side
        half = 0.0
        top = 0.0
        for m in masks:
            ys, xs = np.nonzero(m)
            if len(xs):
                half = max(half, (xs.max() - cx) / px_per_mm,
                           (cx - xs.min()) / px_per_mm)
                top = max(top, (cy - ys.min()) / px_per_mm)
        extent_mm = 2 * max(half, top / 2) * 1.15
    voxel_mm = extent_mm / grid
    # voxel centres
    ax = (np.arange(grid) + 0.5) * voxel_mm - extent_mm / 2   # x and z
    ay = (np.arange(grid) + 0.5) * voxel_mm                    # y from ground
    gx, gy, gz = np.meshgrid(ax, ay, ax, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    keep = np.ones(len(pts), dtype=bool)
    for mask, theta in zip(masks, angles_deg):
        u, v = _project(pts[keep], theta, px_per_mm, cx, cy)
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        inside = (ui >= 0) & (ui < w_img) & (vi >= 0) & (vi < h_img)
        hit = np.zeros(len(ui), dtype=bool)
        hit[inside] = mask[vi[inside], ui[inside]]
        idx = np.nonzero(keep)[0]
        keep[idx[~hit]] = False
        if not keep.any():
            break
    voxels = keep.reshape(grid, grid, grid)
    origin = np.array([-extent_mm / 2, 0.0, -extent_mm / 2])
    return voxels, origin, voxel_mm


def hull_dimensions(voxels, voxel_mm):
    """Overall size of the carved volume in mm along (x, y, z)."""
    idx = np.nonzero(voxels)
    if len(idx[0]) == 0:
        return {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0}
    dims = {}
    for name, axis in zip(("x_mm", "y_mm", "z_mm"), idx):
        dims[name] = round(float((axis.max() - axis.min() + 1) * voxel_mm), 2)
    return dims


def surface_points(voxels, origin, voxel_mm):
    """Centres (mm) of voxels that touch empty space = the point cloud."""
    padded = np.pad(voxels, 1)
    empty_neighbour = np.zeros_like(voxels)
    for shift in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                  (0, 0, -1)):
        s = tuple(slice(1 + shift[i], 1 + shift[i] + voxels.shape[i])
                  for i in range(3))
        empty_neighbour |= ~padded[s]
    surf = voxels & empty_neighbour
    idx = np.argwhere(surf)
    return origin + (idx + 0.5) * voxel_mm, surf


def write_ply_points(path, points, color=(0, 255, 160)):
    """Write an ASCII PLY point cloud."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\n"
                "property uchar blue\nend_header\n")
        r, g, b = color
        for x, y, z in points:
            f.write(f"{x:.2f} {y:.2f} {z:.2f} {r} {g} {b}\n")


def write_ply_mesh(path, vertices, faces, color=(0, 200, 160)):
    """Write an ASCII PLY triangle mesh (vertices in mm)."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\n"
                "property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        r, g, b = color
        for x, y, z in vertices:
            f.write(f"{x:.2f} {y:.2f} {z:.2f} {r} {g} {b}\n")
        for a, b_, c in faces:
            f.write(f"3 {a} {b_} {c}\n")


def mesh_from_voxels(voxels, origin, voxel_mm):
    """Marching-cubes mesh of the voxel volume (needs scikit-image)."""
    from skimage import measure
    volume = voxels.astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(volume, level=0.5)
    # marching cubes returns (x, y, z) in index space matching array axes
    verts_mm = origin + verts * voxel_mm
    return verts_mm, faces
