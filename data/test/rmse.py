import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm

# 读取位姿文件（支持列优先格式）
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

# 计算总的 RMSE
def compute_total_rmse(gt_poses, est_poses):
    if len(gt_poses) != len(est_poses):
        raise ValueError("Ground truth and estimated poses must have the same length.")

    num_poses = len(gt_poses)
    total_errors = []

    for i in range(num_poses):
        T_gt, T_est = gt_poses[i], est_poses[i]
        R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
        R_est, t_est = T_est[:3, :3], T_est[:3, 3]

        # 计算旋转误差 (logm 计算轴角误差)
        R_err = logm(R_est.T @ R_gt)
        if np.iscomplexobj(R_err):
            R_err = R_err.real  # 避免数值误差导致虚部
        rot_error = np.linalg.norm(R_err, 'fro') / np.sqrt(2)  # 旋转误差（单位：弧度）

        # 计算平移误差
        trans_error = np.linalg.norm(t_est - t_gt)  # 欧式距离（单位：米）

        total_errors.append(rot_error**2 + trans_error**2)

    # 计算总的 RMSE
    total_rmse = np.sqrt(np.mean(total_errors))

    return total_rmse

# 主函数
def main():
    input_file = 'poses_gt.txt'  
    output_file = 'poses_init.txt'  

    gt_poses = load_poses(input_file)
    est_poses = load_poses(output_file)

    if not gt_poses or not est_poses:
        print("Error: Empty pose list.")
        return

    total_rmse = compute_total_rmse(gt_poses, est_poses)
    print(f"Total RMSE: {total_rmse:.6f}")

if __name__ == "__main__":
    main()
