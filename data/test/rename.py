import os

# 设置目标文件夹路径
folder_path = "depth/"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".tiff") and "_depth" in filename:
        new_filename = filename.replace("_depth", "")  # 去掉_color
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_filename}")

print("批量重命名完成！")
