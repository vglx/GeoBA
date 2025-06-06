import numpy as np
from scipy.spatial.transform import Rotation as R

# ==== 参数设置 ====
euler_noise_level = 0.05  # 单位：弧度（约等于3度）
translation_noise_level = 2.0  # 单位：与pose单位一致（如cm或m）
input_file = 'poses_gt.txt'
output_file = 'poses_init.txt'

# ==== 读取原始 pose（列优先）====
def load_poses_col_major(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split(',')))
            if len(values) != 16:
                raise ValueError(f"Invalid pose length: {len(values)} in line: {line}")
            T = np.array(values).reshape((4, 4), order='F')  # 列优先
            poses.append(T)
    return poses

# ==== 保存 pose（列优先）====
def save_poses_col_major(file_path, poses):
    with open(file_path, 'w') as f:
        for T in poses:
            flat = T.flatten(order='F')  # 列优先
            line = ','.join(f"{x:.6f}" for x in flat)
            f.write(line + '\n')

# ==== 主逻辑 ====
def add_noise_to_poses(poses, euler_noise_level, translation_noise_level):
    new_poses = []
    for i, T in enumerate(poses):
        R_mat = T[:3, :3]
        t = T[:3, 3]

        # 转为欧拉角（ZYX顺序，和MATLAB默认一致）
        eul = R.from_matrix(R_mat).as_euler('zyx')

        if i > 0:
            eul_noise = euler_noise_level * (np.random.rand(3) - 0.5)
            t_noise = translation_noise_level * (np.random.rand(3) - 0.5)
            eul += eul_noise
            t += t_noise

        R_noised = R.from_euler('zyx', eul).as_matrix()
        T_noised = np.eye(4)
        T_noised[:3, :3] = R_noised
        T_noised[:3, 3] = t
        new_poses.append(T_noised)

    return new_poses

# ==== 执行 ====
def main():
    poses = load_poses_col_major(input_file)
    poses_noised = add_noise_to_poses(poses, euler_noise_level, translation_noise_level)
    save_poses_col_major(output_file, poses_noised)
    print(f"✅ 带噪 pose（列优先）已保存至 {output_file}")

if __name__ == '__main__':
    main()