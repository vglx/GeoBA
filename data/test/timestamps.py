import os

# 配置参数
image_folder = "./test/rgb"  # 修改为你的图片目录
fps = 30  # 帧率

# 计算帧间隔
frame_interval = 1.0 / fps

# 获取所有图片文件，并按文件名排序
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

# 生成时间戳
timestamps = []
for i, filename in enumerate(image_files):
    timestamp = i * frame_interval
    timestamps.append(f"{i} {timestamp:.6f}")
    # timestamps.append(f"{timestamp:.6f} {filename}")

# 保存到文件
output_file = os.path.join("./test/", "times.txt")
with open(output_file, "w") as f:
    f.write("\n".join(timestamps))

print(f"时间戳文件已生成: {output_file}")
