#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rasterize a mesh to a depth image (software z-buffer, NumPy-only) and
compute RMSE vs. an input depth PNG. No occlusion handling beyond the
standard nearest-surface z-buffer (i.e., we assume the scene is not
self-occluded for your current data; if there is, the z-buffer still
resolves it).

- Intrinsics: fx, fy, cx, cy (pixels)
- Pose file: supports comma/space separated 3x4 or 4x4; accepts T_cw or T_wc
- Depth PNG: expects your µm/mm/m pipeline with optional sidecar JSON
  {"unit": "um|mm|m", "scale": <optional multiplier>} as previously used

Dependencies: numpy, imageio, open3d (for mesh IO only)

Notes on speed:
- This is a pure-NumPy triangle rasterizer; on 320x240 it’s typically fine
  for ~60k triangles, but not real‑time. See FAST_MODE below for a quick
  vertex splat approximation; or use OpenGL/PyTorch3D for high performance.
"""

import os
import json
import math
import numpy as np
import imageio.v2 as imageio
import open3d as o3d

# ===================== 用户配置区 =====================
# 输入深度（来自相机 / 仿真）
# png_path    = "./data/halfDef/5/depth/1.png"      # *_depth16_um.png with sidecar JSON is supported
png_path   = "./data/halfDef/5/results/2.png"
# 网格与位姿
mesh_path   = "./data/halfDef/5/mesh.obj"         # or PLY/OBJ
# mesh_path  = "./data/halfDef/5/deformed_mesh_f1.ply"
mesh_unit   = "mm"                                  # "m" or "mm"
pose_txt    = "./data/halfDef/5/results/poses_gt.txt"  # 3x4 or 4x4, comma/space
pose_format = "tcw"                                  # "twc" (T_wc) or "tcw" (T_cw)

# 相机内参（像素）
fx, fy = 155.0030, 155.0030
cx, cy = 160.0, 120.0

# 运行选项
FAST_MODE = False    # True: 仅顶点投影的快速近似（更快但稀疏、不精确）
STRIDE    = 1        # 光栅化时像素步进（>1 可加速，精度略降）
Z_NEAR    = 1e-6     # 近裁剪，避免 z<=0

# =====================================================


def _load_sidecar_json(png_path):
    base, _ = os.path.splitext(png_path)
    sidecar = base + ".json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def load_depth_um_png(png_path, return_unit="m"):
    """读取 *_depth16_um.png（uint16）为米或毫米；若有 sidecar JSON 则按其 unit/scaling 解释。"""
    depth_u16 = imageio.imread(png_path).astype(np.uint16)
    depth = depth_u16.astype(np.float32)

    meta = _load_sidecar_json(png_path)
    depth_m = None
    if meta:
        unit = str(meta.get("unit", "um")).lower()
        scale = float(meta.get("scale", 1.0))
        depth *= scale
        if unit.startswith("um"):
            depth_m = depth * 1e-6
        elif unit.startswith("mm"):
            depth_m = depth * 1e-3
        elif unit in ("m", "meter", "meters"):
            depth_m = depth
        else:
            depth_m = depth * 1e-6  # 安全兜底
    else:
        depth_m = depth * 1e-6     # 无 sidecar 默认 µm

    mask = np.isfinite(depth_m) & (depth_m > 0)
    depth_m = np.where(mask, depth_m, np.nan).astype(np.float32)

    if return_unit == "mm":
        return depth_m * 1000.0, mask, meta
    return depth_m, mask, meta


def load_pose_txt(path, pose_format="tcw"):
    """读取 3x4 或 4x4 位姿；自动适配逗号/空格分隔。
    返回 T_wc（世界←相机）。
    """
    mat = None
    try:
        mat = np.loadtxt(path, delimiter=",")
    except Exception:
        pass
    if mat is None or (mat.ndim == 1 and mat.size == 0):
        mat = np.loadtxt(path)

    if mat.size == 12:
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = mat.reshape(3, 4)
    elif mat.size == 16:
        T = mat.reshape(4, 4)
    else:
        raise RuntimeError(f"Unsupported pose shape from {path}: got {mat.shape}, expect 3x4 or 4x4")

    if pose_format.lower() == "tcw":
        return np.linalg.inv(T)
    return T


def world_to_cam(verts_w, T_cw):
    V = verts_w.shape[0]
    verts_h = np.concatenate([verts_w, np.ones((V, 1), dtype=verts_w.dtype)], axis=1)
    verts_c = (T_cw @ verts_h.T).T[:, :3]
    return verts_c


def project(verts_c, fx, fy, cx, cy):
    z = verts_c[:, 2]
    valid = z > Z_NEAR
    x = np.empty_like(z, dtype=np.float32); x.fill(np.nan)
    y = np.empty_like(z, dtype=np.float32); y.fill(np.nan)
    x[valid] = (verts_c[valid, 0] * fx / z[valid]) + cx
    y[valid] = (verts_c[valid, 1] * fy / z[valid]) + cy
    return np.stack([x, y, z.astype(np.float32)], axis=1), valid


# ----------------- Triangle Rasterization (NumPy) -----------------

def _edge_fn(ax, ay, bx, by, px, py):
    return (py - ay) * (bx - ax) - (px - ax) * (by - ay)


def rasterize_triangles(verts_c, faces, h, w, stride=1):
    """Software z-buffer. Returns depth in *meters* (z in camera frame).
    verts_c: (V,3) camera coords; faces: (F,3) int
    """
    depth = np.full((h, w), np.inf, dtype=np.float32)

    # Project once
    proj, valid = project(verts_c, fx, fy, cx, cy)  # (V,3) with NaNs for invalid

    for f in faces:
        i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
        # Skip if any vertex behind camera
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue

        x0, y0, z0 = proj[i0]
        x1, y1, z1 = proj[i1]
        x2, y2, z2 = proj[i2]

        # Bounding box (clamped)
        xmin = int(max(0, math.floor(min(x0, x1, x2))))
        xmax = int(min(w - 1, math.ceil(max(x0, x1, x2))))
        ymin = int(max(0, math.floor(min(y0, y1, y2))))
        ymax = int(min(h - 1, math.ceil(max(y0, y1, y2))))
        if xmax < xmin or ymax < ymin:
            continue

        # Precompute area
        area = _edge_fn(x0, y0, x1, y1, x2, y2)
        if area == 0:
            continue

        # Iterate pixels (optionally strided)
        for py in range(ymin, ymax + 1, stride):
            # X scanline vectorized
            xs = np.arange(xmin, xmax + 1, stride, dtype=np.float32)
            pys = np.full_like(xs, py, dtype=np.float32)

            w0 = _edge_fn(x1, y1, x2, y2, xs, pys)
            w1 = _edge_fn(x2, y2, x0, y0, xs, pys)
            w2 = _edge_fn(x0, y0, x1, y1, xs, pys)

            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) if area > 0 else (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
            if not np.any(inside):
                continue

            # Barycentric (unnormalized -> normalized)
            w0i = w0[inside]
            w1i = w1[inside]
            w2i = w2[inside]
            denom = (w0i + w1i + w2i)
            a0 = w0i / denom
            a1 = w1i / denom
            a2 = w2i / denom

            # Perspective-correct depth: linear in screen space is sufficient for z-buffer
            zi = a0 * z0 + a1 * z1 + a2 * z2

            xs_i = xs[inside].astype(np.int32)
            ys_i = np.full(xs_i.shape, py, dtype=np.int32)

            # Z-test and update
            old = depth[ys_i, xs_i]
            upd = zi < old
            if np.any(upd):
                depth[ys_i[upd], xs_i[upd]] = zi[upd]

    depth[~np.isfinite(depth)] = np.nan
    depth[depth <= 0] = np.nan
    return depth


# ----------------- Fast vertex splat (optional) -----------------

def rasterize_points_nearest(verts_c, h, w):
    depth = np.full((h, w), np.inf, dtype=np.float32)
    proj, valid = project(verts_c, fx, fy, cx, cy)
    pts = proj[valid]
    xs = np.rint(pts[:, 0]).astype(np.int32)
    ys = np.rint(pts[:, 1]).astype(np.int32)
    zs = pts[:, 2]
    m = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys, zs = xs[m], ys[m], zs[m]
    # z-min write
    for x, y, z in zip(xs, ys, zs):
        if z < depth[y, x]:
            depth[y, x] = z
    depth[~np.isfinite(depth)] = np.nan
    depth[depth <= 0] = np.nan
    return depth


# ----------------- RMSE -----------------

def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan, 0
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d * d))), int(m.sum())


# ===================== 主流程 =====================
if __name__ == "__main__":
    # Load input depth as METERS for computation/comparison
    depth_m, mask_in, meta = load_depth_um_png(png_path, return_unit="m")
    H, W = depth_m.shape
    print(f"[INFO] Loaded input depth: shape={depth_m.shape}, unit=m, meta={meta}")

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("Mesh has no vertices or faces.")

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int32)

    # Convert mesh units -> meters
    mu = mesh_unit.lower()
    if mu == "mm":
        V_m = V * 1e-3
    elif mu == "m":
        V_m = V.copy()
    else:
        raise ValueError("mesh_unit must be 'm' or 'mm'")

    # Pose T_wc, then T_cw for projection
    T_wc = load_pose_txt(pose_txt, pose_format)
    T_cw = np.linalg.inv(T_wc)

    # Transform to camera
    V_c = world_to_cam(V_m.astype(np.float64), T_cw)

    # Rasterize
    if FAST_MODE:
        depth_mesh_m = rasterize_points_nearest(V_c, H, W)
        print("[INFO] FAST_MODE=True (vertex splat)")
    else:
        depth_mesh_m = rasterize_triangles(V_c, F, H, W, stride=STRIDE)
        print(f"[INFO] Rasterized triangles with stride={STRIDE}")

    # RMSE (in meters)
    rmse_m, npx = rmse(depth_mesh_m, depth_m)
    rmse_mm = rmse_m * 1000.0 if np.isfinite(rmse_m) else np.nan

    # Basic stats for sanity check
    def _stats(name, D):
        finite = D[np.isfinite(D)]
        if finite.size == 0:
            return f"{name}: no finite pixels"
        return (f"{name}: min={finite.min():.6f} m, max={finite.max():.6f} m, "
                f"mean={finite.mean():.6f} m, median={np.median(finite):.6f} m, count={finite.size}")

    print(_stats("[Stats] input", depth_m))
    print(_stats("[Stats] mesh ", depth_mesh_m))
    print(f"[METRIC] RMSE = {rmse_m:.6f} m  ({rmse_mm:.3f} mm)  over {npx} overlapping pixels")

    # Optional: save the rasterized depth for inspection (as float32 .npy)
    # np.save("mesh_depth_m.npy", depth_mesh_m)

    # Optional: quick visualization of the two depth maps difference
    try:
        import matplotlib.pyplot as plt
        diff = depth_mesh_m - depth_m
        vmax = np.nanpercentile(np.abs(diff), 99)
        plt.figure(); plt.imshow(depth_mesh_m, vmin=np.nanmin(depth_mesh_m), vmax=np.nanmax(depth_mesh_m)); plt.title("Mesh depth (m)"); plt.colorbar();
        plt.figure(); plt.imshow(depth_m, vmin=np.nanmin(depth_m), vmax=np.nanmax(depth_m)); plt.title("Input depth (m)"); plt.colorbar();
        plt.figure(); plt.imshow(diff, vmin=-vmax, vmax=vmax); plt.title("Mesh - Input (m)"); plt.colorbar();
        plt.show()
    except Exception:
        pass