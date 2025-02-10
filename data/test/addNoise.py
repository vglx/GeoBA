import numpy as np

# 读取位姿数据
def load_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            data = np.array([float(x) for x in line.strip().split(',')]).reshape(4, 4, order='F')  # 列主序读取
            poses.append(data)
    return poses

# 生成带有极大噪声的新位姿
def add_noise_to_poses(poses, trans_noise=2.0, rot_noise=1.0):
    noisy_poses = []
    prev_pose = None
    for pose in poses:
        noisy_pose = pose.copy()
        
        # 生成旋转矩阵噪声
        angle = np.random.normal(0, rot_noise, 3)  # 绕XYZ轴的大旋转
        Rx = np.array([[1, 0, 0], [0, np.cos(angle[0]), -np.sin(angle[0])], [0, np.sin(angle[0]), np.cos(angle[0])]])
        Ry = np.array([[np.cos(angle[1]), 0, np.sin(angle[1])], [0, 1, 0], [-np.sin(angle[1]), 0, np.cos(angle[1])]])
        Rz = np.array([[np.cos(angle[2]), -np.sin(angle[2]), 0], [np.sin(angle[2]), np.cos(angle[2]), 0], [0, 0, 1]])
        R_noise = Rz @ Ry @ Rx  # 旋转扰动矩阵
        
        # 添加旋转噪声并进行正交化
        noisy_R = pose[:3, :3] @ R_noise
        U, _, Vt = np.linalg.svd(noisy_R)
        noisy_pose[:3, :3] = U @ Vt  # 重新正交化
        
        # 生成极大平移噪声
        translation_noise = np.random.normal(0, trans_noise, 3)
        
        # 确保平移的合理性，防止跳变过大
        if prev_pose is not None:
            prev_translation = prev_pose[:3, 3]
            new_translation = prev_translation + translation_noise
        else:
            new_translation = pose[:3, 3] + translation_noise
        
        noisy_pose[:3, 3] = new_translation
        
        # 确保最后一行仍然是 [0, 0, 0, 1]
        noisy_pose[3, :] = [0, 0, 0, 1]
        
        noisy_poses.append(noisy_pose)
        prev_pose = noisy_pose
    
    return noisy_poses

# 保存带噪声的位姿，按照列主序展开
def save_poses(poses, output_file):
    with open(output_file, 'w') as f:
        for pose in poses:
            line = ','.join(map(lambda x: f"{x:.6f}", pose.flatten(order='F')))
            f.write(line + '\n')

# 示例用法
input_file = 'poses_gt.txt'  # 输入的 ground truth 文件
output_file = 'poses_init.txt'  # 输出带噪声的文件
poses = load_poses(input_file)
noisy_poses = add_noise_to_poses(poses)
save_poses(noisy_poses, output_file)

print("带极大噪声的位姿已保存至", output_file)
