#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone visualization script (EXR/PNG depth friendly):
- Reads a deformed mesh (PLY/OBJ)
- Reads a GT depth image (EXR float or PNG16) + intrinsics + pose
- Backprojects depth to world point cloud (float pipeline, avoids mm-quantization banding)
- Shows mesh and point cloud together with different colors

Dependencies: open3d, numpy, imageio (or opencv)
"""

import os
import sys
import numpy as np
import open3d as o3d
import imageio

# ===================== USER INPUTS =====================
# Mesh & depth paths
mesh_path   = "./data/halfDef/1/results/deformed_mesh_f1.ply"
# mesh_path   = "./data/halfDef/1/mesh.obj"
# depth_path can be .exr (float) or .png (uint16)
depth_path  = "./data/halfDef/1/results/1.png"  # change to .exr if available

# Camera pose
pose_txt    = "./data/halfDef/1/results/poses_gt.txt"  # 4x4 or 3x4 matrix
pose_format = "tcw"              # "twc" (T_wc) or "tcw" (T_cw)

# Camera intrinsics (pixels)
fx, fy = 155.0030, 155.0030
cx, cy = 160.0, 120.0

# Depth handling
# If using PNG16 in millimeters, set depth_scale = 1000.0 (mm->m).
# If using EXR where value = (mm * 1/5), then exr_to_mm = 5.0 as per your dataset.
depth_scale_png = 1000.0   # PNG16: value / 1000 -> meters
exr_to_mm       = 5.0      # EXR float to millimeters multiplier (EXR×5 → mm)
mm_to_m         = 1.0 / 1000.0

# Visualization / filtering
depth_trunc_m = 0.0        # meters; 0 or negative disables truncation
subsample     = 1          # take 1 of every N pixels

# Treat these depth values as invalid
invalidate_zero = True
invalidate_nan  = True

# Colors
pc_color   = [0.0, 1.0, 0.0]  # green for point cloud
mesh_color = [1.0, 0.0, 0.0]  # red for mesh
mesh_alpha = 0.6              # may not be supported in fallback vis

# (Optional) quick stats on unique depth levels (rounded) for diagnosing banding
print_depth_stats = True
stats_round_decimals = 6  # round to 1e-6 m when counting unique values

# ===================== HELPERS =====================

def load_pose_txt(path: str) -> np.ndarray:
    """Load 4x4 or 3x4 pose matrix from txt (space or comma separated)."""
    arr = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',') if ',' in line else line.split()
            arr.extend([float(x) for x in parts if x])
    arr = np.array(arr, dtype=np.float64)
    if arr.size == 16:
        T = arr.reshape(4, 4)
    elif arr.size >= 12:
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = arr.reshape(3, 4)
    else:
        raise RuntimeError("Pose txt must contain 12 or 16 numbers")
    return T


def load_depth_meters(path: str) -> np.ndarray:
    """Load depth image (EXR float or PNG16) and return meters as float32 without quantization losses.
    - EXR: interpret as float, multiply by exr_to_mm, then convert mm->m
    - PNG16: interpret as uint16 millimeters and divide by depth_scale_png
    """
    ext = os.path.splitext(path)[1].lower()
    depth_any = imageio.imread(path)

    if ext == ".exr":
        depth_exr = depth_any.astype(np.float32)
        if depth_exr.ndim == 3:
            depth_exr = depth_exr[..., 0]  # take first channel
        depth_m = depth_exr * float(exr_to_mm) * float(mm_to_m)
    else:
        # PNG / others
        if depth_any.dtype == np.uint16:
            depth_m = depth_any.astype(np.float32) / float(depth_scale_png)
        else:
            # Assume already meters in float
            depth_m = depth_any.astype(np.float32)
    return depth_m


def backproject_depth_to_world(depth_m: np.ndarray, T: np.ndarray,
                               pose_fmt: str,
                               fx: float, fy: float, cx: float, cy: float,
                               subsample: int = 1,
                               depth_trunc_m: float = 0.0,
                               invalidate_zero: bool = True,
                               invalidate_nan: bool = True) -> np.ndarray:
    """Return (N,3) world points from depth in meters.
    pose_fmt: 'twc' (world-from-camera) or 'tcw' (camera-from-world)
    """
    H, W = depth_m.shape

    # grid
    us = np.arange(0, W, max(1, int(subsample)))
    vs = np.arange(0, H, max(1, int(subsample)))
    UU, VV = np.meshgrid(us, vs)

    Z = depth_m[VV, UU]

    mask = np.ones_like(Z, dtype=bool)
    if invalidate_nan:
        mask &= np.isfinite(Z)
    if invalidate_zero:
        mask &= (Z > 0)
    if depth_trunc_m > 0:
        mask &= (Z <= depth_trunc_m)

    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float64)

    UU, VV, Z = UU[mask], VV[mask], Z[mask]

    # backproject to camera frame
    X = (UU - cx) / fx * Z
    Y = (VV - cy) / fy * Z
    xc = np.stack([X, Y, Z], axis=0)  # 3xN

    # transform to world
    R, t = T[:3, :3], T[:3, 3]
    if pose_fmt.lower() == "twc":
        xw = (R @ xc) + t.reshape(3, 1)
    elif pose_fmt.lower() == "tcw":
        xw = R.T @ (xc - t.reshape(3, 1))
    else:
        raise ValueError("pose_format must be 'twc' or 'tcw'")

    pts_w = xw.T.astype(np.float64)
    return pts_w


# ===================== MAIN =====================
if __name__ == "__main__":
    # ---- Load mesh ----
    if not os.path.isfile(mesh_path):
        print(f"[ERROR] Mesh not found: {mesh_path}")
        sys.exit(1)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        print(f"[ERROR] Failed to load mesh: {mesh_path}")
        sys.exit(1)
    mesh.compute_vertex_normals()

    # ---- Load depth (meters) ----
    if not os.path.isfile(depth_path):
        print(f"[ERROR] Depth not found: {depth_path}")
        sys.exit(1)
    depth_m = load_depth_meters(depth_path)

    # ---- Optional stats to diagnose banding ----
    if print_depth_stats:
        valid = np.isfinite(depth_m) & (depth_m > 0)
        if np.any(valid):
            # random subsample to keep memory reasonable
            vals = depth_m[valid]
            if vals.size > 200000:
                rng = np.random.default_rng(0)
                idx = rng.choice(vals.size, size=200000, replace=False)
                vals = vals[idx]
            uniq = np.unique(np.round(vals, stats_round_decimals)).size
            print(f"[STAT] approx unique depth levels (rounded 1e-{stats_round_decimals} m): {uniq}")
        else:
            print("[STAT] no valid positive depth values found.")

    # ---- Load pose ----
    if not os.path.isfile(pose_txt):
        print(f"[ERROR] Pose file not found: {pose_txt}")
        sys.exit(1)
    T = load_pose_txt(pose_txt)

    # ---- Backproject ----
    pts_w = backproject_depth_to_world(depth_m, T, pose_format, fx, fy, cx, cy,
                                       subsample=subsample,
                                       depth_trunc_m=depth_trunc_m,
                                       invalidate_zero=invalidate_zero,
                                       invalidate_nan=invalidate_nan)
    if pts_w.shape[0] == 0:
        print("[WARN] No valid 3D points after masking.")

    # ---- Build point cloud ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_w)
    pcd.paint_uniform_color(pc_color)

    # ---- Visualization ----
    try:
        vis = o3d.visualization.O3DVisualizer("Mesh + GT Depth Point Cloud", 1024, 768)
        vis.add_geometry("mesh", mesh)
        vis.add_geometry("gt_pc", pcd)

        mat_mesh = o3d.visualization.rendering.MaterialRecord()
        mat_mesh.shader = "defaultLit"
        mat_mesh.base_color = [mesh_color[0], mesh_color[1], mesh_color[2], mesh_alpha]
        vis.modify_geometry_material("mesh", mat_mesh)

        mat_pc = o3d.visualization.rendering.MaterialRecord()
        mat_pc.shader = "defaultUnlit"
        mat_pc.point_size = 2.0
        vis.modify_geometry_material("gt_pc", mat_pc)

        o3d.visualization.gui.Application.instance.initialize()
        o3d.visualization.gui.Application.instance.add_window(vis)
        o3d.visualization.gui.Application.instance.run()
    except Exception as e:
        print("[Info] Fallback viewer, alpha not supported. Using draw_geometries().")
        o3d.visualization.draw_geometries([mesh, pcd])