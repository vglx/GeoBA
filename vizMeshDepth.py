#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版：直接在文件里配置路径和参数，无需传参。
已增强：位姿文件支持逗号/空格分隔（自动适配）。
"""

import os
import json
import numpy as np
import imageio.v2 as imageio
import open3d as o3d

# ==== 用户配置区 ====
dataset     = "6"
png_path   = f"./data/halfDef/{dataset}/results/1.png"
pose_txt   = f"./data/halfDef/{dataset}/results/poses_gt.txt"  # 3x4 或 4x4，支持逗号或空格分隔
pose_format = "tcw"   # "twc" or "tcw"
mesh_path  = f"./data/halfDef/{dataset}/results/PLYs/deformed_mesh_f1.ply"
# mesh_path  = f"./data/halfDef/{dataset}/mesh.obj"
mesh_unit  = "mm"     # 网格单位: "m" 或 "mm"

fx, fy = 155.0030, 155.0030
cx, cy = 160.0, 120.0

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
    """读取 MATLAB 导出的 *_depth16_um.png（uint16, 单位=µm），并转为 m 或 mm。"""
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
            depth_m = depth * 1e-6  # 兜底按 µm
    else:
        depth_m = depth * 1e-6     # 无 sidecar 默认 µm

    # 无效与 NaN 处理
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
    """读取 3x4 或 4x4 位姿矩阵；自动适配逗号/空格分隔。
    - 若文件是 T_cw（相机←世界），传 pose_format="tcw"（默认），内部会求逆得到 T_wc。
    - 若文件已是 T_wc（世界←相机），传 pose_format="twc"，将直接返回。
    """
    # 先尝试逗号分隔
    mat = None
    try:
        mat = np.loadtxt(path, delimiter=",")
    except Exception:
        pass
    if mat is None or (mat.ndim == 1 and mat.size == 0):
        # 回退为空白分隔
        mat = np.loadtxt(path)

    # 规范到 4x4
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

# ===================== 主流程 =====================
if __name__ == "__main__":
    # 1) 读深度（输出为米）
    depth_m, mask, meta = load_depth_um_png(png_path, return_unit="m")
    print(f"[INFO] depth shape={depth_m.shape}, unit=m, meta={meta}")

    # 2) 回投（单位=米）
    pts_cam = backproject_depth(depth_m, fx, fy, cx, cy)

    # 3) 读位姿 -> T_wc
    T_wc = load_pose_txt(pose_txt, pose_format)

    # 4) 相机->世界，并把单位匹配到网格（默认网格=mm）
    pts_world_m = cam_to_world(pts_cam, T_wc)
    pts_world = ensure_units(pts_world_m, mesh_unit=mesh_unit, depth_unit="m")
    print(f"[INFO] points after unit align: {pts_world.shape}, mesh_unit={mesh_unit}")

    # 5) 可视化
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world.astype(np.float64))
    colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (pts_world.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([mesh, pcd])