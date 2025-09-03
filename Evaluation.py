#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone mesh/point metrics script (no CLI params; edit config block).
Computes:
  (1) Point-to-mesh surface distances (RMSE/mean/median/p95) using Open3D RaycastingScene if available.
      Falls back to nearest-neighbor against uniformly-sampled mesh points when RaycastingScene missing.
  (2) Symmetric Chamfer distance (mean/RMSE/p95-sum) via bidirectional NN on sampled mesh points.

Notes:
- Depth PNG assumed to encode depth in µm (micrometers) by default; if a sidecar JSON exists (same stem),
  we read {"unit": "um"/"mm"/"m"} to convert properly.
- Pose file supports comma or whitespace separated 3x4 / 4x4. Use pose_format = "tcw" (T_cw) or "twc".
- Units: point cloud is aligned to mesh_unit ("m" or "mm") before metrics.
- Outputs a small JSON report next to the PNG path (same stem + "_metrics.json").
"""

import os
import json
import numpy as np

# Prefer imageio v2 API
try:
    import imageio.v2 as imageio
except Exception:
    import imageio

import open3d as o3d

# ========== USER CONFIG ==========
png_path    = "./data/halfDef/5/results/2.png"       # depth PNG (uint16), default unit µm unless sidecar says otherwise
pose_txt    = "./data/halfDef/5/results/poses_gt.txt"  # 3x4 or 4x4 pose
pose_format = "tcw"     # "twc" or "tcw"
mesh_path   = "./data/halfDef/5/mesh.obj"             # or a PLY
mesh_unit   = "mm"      # "m" or "mm"

fx, fy = 155.0030, 155.0030
cx, cy = 160.0, 120.0

# ========== UTILS ==========
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
    """Read *_depth16_um.png (uint16). Convert to meters by default or millimeters if requested.
       If a sidecar JSON exists (same stem), it may define the unit: "um"/"mm"/"m"."""
    depth_u16 = imageio.imread(png_path).astype(np.uint16)
    depth = depth_u16.astype(np.float32)

    meta = _load_sidecar_json(png_path)
    if meta:
        unit = str(meta.get("unit", "um")).lower()
        if unit.startswith("um"):
            depth_m = depth * 1e-6
        elif unit.startswith("mm"):
            depth_m = depth * 1e-3
        elif unit in ("m", "meter", "meters"):
            depth_m = depth
        else:
            depth_m = depth * 1e-6  # default µm
    else:
        depth_m = depth * 1e-6     # default µm

    mask = np.isfinite(depth_m) & (depth_m > 0)
    depth_m = np.where(mask, depth_m, np.nan).astype(np.float32)

    if return_unit == "mm":
        return depth_m * 1000.0, mask, meta
    return depth_m, mask, meta

def backproject_depth(depth, fx, fy, cx, cy):
    """Backproject pixels to camera frame; depth unit is preserved (m or mm)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)

def load_pose_txt(path, pose_format="tcw"):
    """Read 3x4 or 4x4 pose. Accept comma or whitespace separators.
       pose_format: "tcw" (T_cw) or "twc" (T_wc). Returns T_wc."""
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

def cam_to_world(pts_cam, T_wc):
    pts = pts_cam.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    return (T_wc @ pts_h.T).T[:, :3]

def ensure_units(pts, mesh_unit="mm", depth_unit="m"):
    mesh_unit = mesh_unit.lower(); depth_unit = depth_unit.lower()
    if depth_unit == mesh_unit:
        return pts
    if depth_unit == "m" and mesh_unit == "mm":
        return pts * 1000.0
    if depth_unit == "mm" and mesh_unit == "m":
        return pts / 1000.0
    return pts

# ========== METRICS ==========
def point_to_mesh_distances(mesh_legacy, pcd_legacy, mesh_unit="mm"):
    """Return dictionary with RMSE/mean/median/p95 for point->mesh surface distance.
       Uses RaycastingScene when available; otherwise falls back to NN to sampled mesh."""
    # Try tensor raycasting
    rmse = mean = median = p95 = np.nan
    try:
        from open3d import t as o3dt
        scene = o3dt.geometry.RaycastingScene()
        tmesh = o3dt.geometry.TriangleMesh.from_legacy(mesh_legacy)
        _ = scene.add_triangles(tmesh)

        pts_np = np.asarray(pcd_legacy.points).astype(np.float32)
        pts_t  = o3dt.core.Tensor(pts_np, dtype=o3dt.core.Dtype.Float32)
        dist   = scene.compute_distance(pts_t).numpy()
        valid  = np.isfinite(dist)
        d = dist[valid]
        if d.size:
            rmse   = float(np.sqrt(np.mean(d**2)))
            mean   = float(np.mean(d))
            median = float(np.median(d))
            p95    = float(np.percentile(d, 95))
        return {"rmse": rmse, "mean": mean, "median": median, "p95": p95, "unit": mesh_unit, "method": "raycasting"}
    except Exception:
        # Fallback: NN to sampled mesh
        num_samples = 200000
        mesh_samples = mesh_legacy.sample_points_uniformly(num_samples=num_samples)
        mesh_pts = np.asarray(mesh_samples.points)
        mesh_tree = o3d.geometry.KDTreeFlann(mesh_samples)

        pts_np = np.asarray(pcd_legacy.points)
        d = np.empty(pts_np.shape[0], dtype=np.float32)
        for i, q in enumerate(pts_np):
            _, idx, _ = mesh_tree.search_knn_vector_3d(q, 1)
            d[i] = np.linalg.norm(q - mesh_pts[idx[0]])
        if d.size:
            rmse   = float(np.sqrt(np.mean(d**2)))
            mean   = float(np.mean(d))
            median = float(np.median(d))
            p95    = float(np.percentile(d, 95))
        return {"rmse": rmse, "mean": mean, "median": median, "p95": p95, "unit": mesh_unit, "method": "nn_fallback"}

def chamfer_symmetric(mesh_legacy, pcd_legacy, mesh_unit="mm", num_samples=200000):
    """Return dictionary for symmetric Chamfer (mean/RMSE/p95-sum)."""
    mesh_samples = mesh_legacy.sample_points_uniformly(num_samples=num_samples)
    mesh_pts = np.asarray(mesh_samples.points)

    pcd_tree  = o3d.geometry.KDTreeFlann(pcd_legacy)
    mesh_tree = o3d.geometry.KDTreeFlann(mesh_samples)

    def nn_dist(query_pts, ref_tree, ref_pts):
        out = np.empty(query_pts.shape[0], dtype=np.float32)
        for i, q in enumerate(query_pts):
            _, idx, _ = ref_tree.search_knn_vector_3d(q, 1)
            out[i] = np.linalg.norm(q - ref_pts[idx[0]])
        return out

    p_pts = np.asarray(pcd_legacy.points)
    d_pm = nn_dist(p_pts, mesh_tree, mesh_pts)       # point -> mesh_samples
    d_mp = nn_dist(mesh_pts, pcd_tree, p_pts)       # mesh_samples -> point

    chamfer_mean = float(np.mean(d_pm) + np.mean(d_mp))
    chamfer_rmse = float(np.sqrt(np.mean(d_pm**2) + np.mean(d_mp**2)))
    chamfer_p95  = float(np.percentile(d_pm,95) + np.percentile(d_mp,95))
    return {"mean": chamfer_mean, "rmse": chamfer_rmse, "p95_sum": chamfer_p95, "unit": mesh_unit, "samples": num_samples}

# ========== MAIN ==========
def main():
    # 1) Load depth (meters)
    depth_m, mask, meta = load_depth_um_png(png_path, return_unit="m")
    print(f"[INFO] depth shape={depth_m.shape}, unit=m, meta={meta}")

    # 2) Backproject to camera (meters)
    pts_cam = backproject_depth(depth_m, fx, fy, cx, cy)

    # 3) Load pose -> T_wc
    T_wc = load_pose_txt(pose_txt, pose_format)

    # 4) Camera->World, units -> mesh_unit
    pts_world_m = cam_to_world(pts_cam, T_wc)
    pts_world = ensure_units(pts_world_m, mesh_unit=mesh_unit, depth_unit="m")
    print(f"[INFO] points after unit align: {pts_world.shape}, mesh_unit={mesh_unit}")

    # 5) Build legacy Open3D objects
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {mesh_path}")
    mesh.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world.astype(np.float64))

    # 6) Metrics
    m1 = point_to_mesh_distances(mesh, pcd, mesh_unit=mesh_unit)
    print(f"[METRIC point->mesh] method={m1['method']}  RMSE={m1['rmse']:.6f}  mean={m1['mean']:.6f}  "
          f"median={m1['median']:.6f}  p95={m1['p95']:.6f}  (unit={m1['unit']})")

    m2 = chamfer_symmetric(mesh, pcd, mesh_unit=mesh_unit, num_samples=200000)
    print(f"[METRIC chamfer] mean={m2['mean']:.6f}  rmse={m2['rmse']:.6f}  "
          f"p95(sum)={m2['p95_sum']:.6f}  (unit={m2['unit']}, samples={m2['samples']})")


    # 7) Save a JSON report next to the PNG path
    stem, _ = os.path.splitext(png_path)
    out_json = stem + "_metrics.json"
    report = {
        "png_path": png_path,
        "pose_txt": pose_txt,
        "mesh_path": mesh_path,
        "mesh_unit": mesh_unit,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "sidecar_meta": meta,
        "point_to_mesh": m1,
        "chamfer": m2,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] saved metrics JSON -> {out_json}")

if __name__ == "__main__":
    main()