import numpy as np
from scipy.spatial.transform import Rotation as R

# 读取位姿文件 (支持列优先解析)
def load_poses(file_path):
    poses = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split(',')))
                if len(values) != 16:
                    raise ValueError(f"Unexpected number of elements in line: {line}")
                
                # 按列优先解析数据
                matrix = np.array(values).reshape(4, 4, order='F')  
                poses.append(matrix)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return poses

# 计算相邻两帧的相对位姿
def compute_relative_poses(poses):
    relative_poses = []
    for i in range(len(poses) - 1):
        T1, T2 = poses[i], poses[i + 1]
        R1, t1 = T1[:3, :3], T1[:3, 3]
        R2, t2 = T2[:3, :3], T2[:3, 3]

        R_rel = R1.T @ R2
        t_rel = R1.T @ (t2 - t1)

        rel_pose = np.eye(4)
        rel_pose[:3, :3] = R_rel
        rel_pose[:3, 3] = t_rel

        relative_poses.append(rel_pose)
    
    return relative_poses

# 添加噪声
def add_noise_to_pose(pose, noise_scale=0.01, use_fixed_noise=False):
    noise = np.eye(4)

    if use_fixed_noise:
        axis_angle_noise = np.array([0.01, -0.005, 0.002])  
        translation_noise = np.array([0.01, -0.01, 0.005])
    else:
        axis_angle_noise = noise_scale * np.random.randn(3)  
        translation_noise = noise_scale * np.random.randn(3)  

    # 生成扰动旋转矩阵
    R_noise = R.from_rotvec(axis_angle_noise).as_matrix()
    noise[:3, :3] = R_noise
    noise[:3, 3] = translation_noise

    return pose @ noise

# 计算新的带噪声位姿
def compute_noisy_poses(initial_pose, relative_poses, noise_scale=0.01, use_fixed_noise=False):
    new_poses = [initial_pose]
    current_pose = initial_pose.copy()

    for rel_pose in relative_poses:
        noisy_rel_pose = add_noise_to_pose(rel_pose, noise_scale, use_fixed_noise)
        current_pose = current_pose @ noisy_rel_pose

        # 确保旋转矩阵正交
        U, _, Vt = np.linalg.svd(current_pose[:3, :3])
        current_pose[:3, :3] = U @ Vt
        current_pose[3, :] = [0, 0, 0, 1]  

        new_poses.append(current_pose)

    return new_poses

# 保存带噪声的位姿 (按列优先存储)
def save_poses(file_path, poses):
    try:
        with open(file_path, 'w') as f:
            for pose in poses:
                formatted_values = ','.join(f"{v:.6f}" for v in pose.flatten(order='F'))  
                f.write(formatted_values + '\n')
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")

# 主函数
def main():
    input_file = 'poses_gt.txt'  
    output_file = 'poses_init.txt'  
    noise_scale = 0.01  
    use_fixed_noise = False  

    poses = load_poses(input_file)
    if not poses:
        print("No valid poses found. Exiting.")
        return

    relative_poses = compute_relative_poses(poses)
    noisy_poses = compute_noisy_poses(poses[0], relative_poses, noise_scale, use_fixed_noise)
    save_poses(output_file, noisy_poses)
    print(f'带噪声的新位姿已保存至 {output_file}')

if __name__ == "__main__":
    main()
