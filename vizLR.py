#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化：读取「中心相机」位姿 + 左/右目深度，基线=4.5mm，推导左右目 T_wc，
将左右目深度各自回投到相机坐标，再用各自 T_wc 变换到世界坐标，与网格一起可视化。

特点：
- 位姿文件(3x4 或 4x4) 自动适配逗号/空格分隔；pose_format="tcw"/"twc"可选。
- 深度 PNG 支持 µm/mm/m（可选旁车 JSON: {"unit":"um|mm|m", "scale":s}）。
- 网格单位 "mm" 或 "m"，脚本会将点云单位转换以匹配网格。
- 两个点云使用不同配色（左=红，右=蓝），可选体素降采样。
"""

import os
import json
import numpy as np
import imageio.v2 as imageio
import open3d as o3d

# ==== 用户配置区 ====
dataset       = "6"
left_png      = f"./data/halfDef/{dataset}/results/left.png"   # 左目深度（与左目相机对应）
right_png     = f"./data/halfDef/{dataset}/results/right.png"  # 右目深度（与右目相机对应）
pose_txt      = f"./data/halfDef/{dataset}/results/poses_gt.txt" # 中心相机位姿（3x4 或 4x4）
pose_format   = "tcw"   # "twc" 或 "tcw"（文件里是 T_cw 就写 tcw，会自动求逆）
mesh_path     = f"./data/halfDef/{dataset}/results/PLYs/deformed_mesh_f1.ply"  # 或模板 mesh.obj
mesh_unit     = "mm"     # 网格单位: "m" 或 "mm"

# 相机内参（像素）——左右目相同（如不同，可拆成 fxL/fyL/cxL/cyL 与 fxR/...）
fx, fy = 155.0030, 155.0030
cx, cy = 160.0, 120.0

# rig 基线（毫米）。约定：左=−b/2，右=+b/2（左负右正）
baseline_mm  = 4.5

# 可视化/降采样参数
voxel_size_mm = 0.6   # 体素大小（与网格单位一致：mm）。设为 0 关闭降采样。
max_points    = 2_000_000  # 上限，超过则自动均匀采样

# ===================== 工具函数 =====================

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
    """读取 *_depth16_um.png（uint16, 深度单位通常为 µm），并转为 m 或 mm。支持旁车 JSON。"""
    depth_u16 = imageio.imread(png_path).astype(np.uint16)
    depth = depth_u16.astype(np.float32)

    meta = _load_sidecar_json(png_path)
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
            depth_m = depth * 1e-6
    else:
        depth_m = depth * 1e-6

    mask = np.isfinite(depth_m) & (depth_m > 0)
    depth_m = np.where(mask, depth_m, np.nan).astype(np.float32)
    if return_unit == "mm":
        return depth_m * 1000.0, mask, meta
    return depth_m, mask, meta


def backproject_depth(depth, fx, fy, cx, cy):
    """像素坐标回投到相机坐标系; depth 单位保持不变(m 或 mm)。"""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def load_pose_txt(path, pose_format="tcw"):
    """读取单帧位姿矩阵（中心相机），返回 T_wc。
    - 支持 3x4 或 4x4；逗号/空格分隔自适应。
    - pose_format="tcw" 时会取逆得到 T_wc。
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


def derive_lr_Twcs_from_center(Twc_center, baseline_mm):
    """由中心相机 T_wc 推导左右目 T_wc（平行双目）。左=−b/2，右=+b/2。"""
    half_b_m = (baseline_mm * 0.5) * 1e-3
    tC_L = np.array([-half_b_m, 0.0, 0.0], dtype=np.float64)  # 左负右正
    tC_R = np.array([+half_b_m, 0.0, 0.0], dtype=np.float64)

    Rwc = Twc_center[:3, :3]
    twc = Twc_center[:3, 3]

    TLwc = np.eye(4, dtype=np.float64)
    TRwc = np.eye(4, dtype=np.float64)
    TLwc[:3, :3] = Rwc; TRwc[:3, :3] = Rwc
    TLwc[:3, 3]  = twc + Rwc @ tC_L
    TRwc[:3, 3]  = twc + Rwc @ tC_R
    return TLwc, TRwc


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


def to_o3d_pcd(pts, color_rgb, voxel_size=0.0):
    pcd = o3d.geometry.PointCloud()
    if pts.size == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    colors = np.tile(np.array(color_rgb, dtype=np.float64)[None, :], (pts.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) > max_points:
        idx = np.linspace(0, len(pcd.points)-1, num=max_points, dtype=np.int64)
        pcd = pcd.select_by_index(idx.tolist())
    return pcd

# ===================== 主流程 =====================
if __name__ == "__main__":
    # 1) 读左/右深度（单位：米）
    depthL_m, _, metaL = load_depth_um_png(left_png,  return_unit="m")
    depthR_m, _, metaR = load_depth_um_png(right_png, return_unit="m")
    if depthL_m.shape != depthR_m.shape:
        raise RuntimeError(f"L/R depth size mismatch: L={depthL_m.shape}, R={depthR_m.shape}")
    H, W = depthL_m.shape
    print(f"[INFO] depth(L/R) shape={H}x{W} (m). metaL={metaL}, metaR={metaR}")

    # 2) 回投到相机坐标（各自相机）
    ptsL_cam_m = backproject_depth(depthL_m, fx, fy, cx, cy)
    ptsR_cam_m = backproject_depth(depthR_m, fx, fy, cx, cy)

    # 3) 读中心相机位姿 -> 统一为 T_wc
    Twc_center = load_pose_txt(pose_txt, pose_format)

    # 4) 由中心位姿推导左右目 T_wc（左负右正）
    Twc_L, Twc_R = derive_lr_Twcs_from_center(Twc_center, baseline_mm=baseline_mm)

    # 5) 各自从相机坐标变换到世界坐标（仍是米）
    ptsL_w_m = cam_to_world(ptsL_cam_m, Twc_L)
    ptsR_w_m = cam_to_world(ptsR_cam_m, Twc_R)

    # 6) 单位转换以匹配网格单位（默认网格=mm）
    ptsL_w = ensure_units(ptsL_w_m, mesh_unit=mesh_unit, depth_unit="m")
    ptsR_w = ensure_units(ptsR_w_m, mesh_unit=mesh_unit, depth_unit="m")
    print(f"[INFO] L/R points after unit align: L={ptsL_w.shape}, R={ptsR_w.shape}, mesh_unit={mesh_unit}")

    # 7) 读取网格
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        raise RuntimeError(f"Empty mesh triangles: {mesh_path}")
    mesh.compute_vertex_normals()

    # 8) 体素降采样（与网格单位一致）
    vxl = voxel_size_mm if mesh_unit.lower()=="mm" else voxel_size_mm/1000.0
    pcdL = to_o3d_pcd(ptsL_w, color_rgb=(1.0, 0.0, 0.0), voxel_size=vxl)  # 左=红
    pcdR = to_o3d_pcd(ptsR_w, color_rgb=(0.0, 0.4, 1.0), voxel_size=vxl)  # 右=蓝

    # 9) 可视化
    print("[INFO] Visualizing: mesh + L(red) + R(blue)")
    o3d.visualization.draw_geometries([mesh, pcdL, pcdR])