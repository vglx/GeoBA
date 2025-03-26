import open3d as o3d
import numpy as np

def load_mesh(obj_path):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if not mesh.has_vertices():
        raise ValueError("Mesh has no vertices!")
    mesh.compute_vertex_normals()
    return mesh

def load_poses_txt(txt_path):
    poses = []
    with open(txt_path, 'r') as f:
        for line_num, line in enumerate(f):
            parts = list(map(float, line.strip().split(',')))
            if len(parts) != 16:
                print(f"[WARNING] Line {line_num + 1} skipped: not 16 values")
                continue
            pose = np.array(parts).reshape((4, 4), order='F')  # 列优先
            if not np.isfinite(pose).all():
                print(f"[WARNING] Line {line_num + 1} contains NaN/Inf, skipped")
                continue
            poses.append(pose)
    return poses

def create_camera_frame(pose, size=20.0):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(pose)
    return frame

def main():
    mesh_path = "mesh.obj"
    poses_path = "poses_gt.txt"

    print("[INFO] Loading mesh...")
    mesh = load_mesh(mesh_path)

    print("[INFO] Loading poses...")
    poses = load_poses_txt(poses_path)
    print(f"[INFO] Loaded {len(poses)} poses")

    geometry_list = [mesh]
    for idx, pose in enumerate(poses):
        frame = create_camera_frame(pose)
        geometry_list.append(frame)

    print("[INFO] Visualizing mesh with poses...")
    o3d.visualization.draw_geometries(geometry_list)

if __name__ == "__main__":
    main()
