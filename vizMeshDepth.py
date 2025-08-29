#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone visualization script:
- Reads a deformed mesh (PLY)
- Reads a GT depth image + intrinsics + pose
- Backprojects depth to world point cloud
- Shows mesh and point cloud together with different colors

Dependencies: open3d, numpy, imageio (or opencv)
"""

import numpy as np
import open3d as o3d
import imageio

# ==== USER INPUTS ====
mesh_path   = "./data/halfDef/1/results/deformed_mesh_f1.ply"
depth_path  = "./data/halfDef/1/results/1.png"
pose_txt    = "./data/halfDef/1/results/poses_gt.txt"  # 4x4 or 3x4 matrix
pose_format = "tcw"              # "twc" (T_wc) or "tcw" (T_cw)

fx, fy = 155.0030, 155.0030
cx, cy = 160.0, 120.0

# depth options
depth_scale = 1000.0   # e.g., if PNG uint16 in millimeters
depth_trunc = 4.0      # meters; 0 or negative to disable
subsample   = 2        # take 1 of every N pixels

# colors
pc_color   = [0.0, 1.0, 0.0]  # green for point cloud
mesh_color = [1.0, 0.0, 0.0]  # red for mesh
mesh_alpha = 0.6              # may not be supported in fallback vis

# ==== LOAD MESH ====
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()

# ==== LOAD DEPTH ====
depth_raw = imageio.imread(depth_path)
if depth_raw.dtype == np.uint16:
    depth_m = depth_raw.astype(np.float32) / depth_scale
else:
    depth_m = depth_raw.astype(np.float32)

# ==== LOAD POSE ====
def load_pose_txt(path):
    arr = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 同时支持空格或逗号分隔
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()
            arr.extend([float(x) for x in parts if x])
    arr = np.array(arr, dtype=np.float64)
    if arr.size == 16:
        T = arr.reshape(4,4)
    elif arr.size >= 12:
        T = np.eye(4)
        T[:3,:4] = arr.reshape(3,4)
    else:
        raise RuntimeError("Pose txt must contain 12 or 16 numbers")
    return T

T = load_pose_txt(pose_txt)

# ==== BACKPROJECT DEPTH ====
H, W = depth_m.shape
us = np.arange(0, W, subsample)
vs = np.arange(0, H, subsample)
UU, VV = np.meshgrid(us, vs)
Z = depth_m[VV, UU]

mask = np.isfinite(Z) & (Z > 0)
if depth_trunc > 0:
    mask &= (Z <= depth_trunc)
UU, VV, Z = UU[mask], VV[mask], Z[mask]

X = (UU - cx) / fx * Z
Y = (VV - cy) / fy * Z
xc = np.stack([X, Y, Z], axis=0)  # 3xN

R, t = T[:3,:3], T[:3,3]
if pose_format == "twc":
    xw = (R @ xc) + t.reshape(3,1)
elif pose_format == "tcw":
    xw = R.T @ (xc - t.reshape(3,1))
else:
    raise ValueError("pose_format must be 'twc' or 'tcw'")

pts_w = xw.T

# ==== BUILD POINT CLOUD ====
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_w)
pcd.paint_uniform_color(pc_color)

# ==== VISUALIZE ====
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
    print("[Info] Fallback viewer, alpha not supported.")
    o3d.visualization.draw_geometries([mesh, pcd])