import open3d as o3d
import numpy as np

# === 加载网格 ===
mesh = o3d.io.read_triangle_mesh("mesh.obj")
mesh.compute_vertex_normals()

# === 相机到世界的位姿矩阵（T_cw） ===
camera_pose = np.array([
    [ 0.457691, -0.402822,  0.792624,   66.739804],
    [ 0.450261, -0.663700, -0.597299,   48.320840],
    [ 0.766670,  0.630267, -0.122395, -125.353227],
    [ 0.000000,  0.000000,  0.000000,    1.000000]
])

# === 创建相机坐标系（红X绿Y蓝Z）并应用变换 ===
camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0)
camera_frame.transform(camera_pose)

# === 可视化 ===
o3d.visualization.draw_geometries([mesh, camera_frame])