import numpy as np

# 你的原始相机标定文件
input_file = "/Users/wangrunze/Desktop/工件数据/内参/ty_color_calib_fs820_207000152299.txt"

# 读取文件
with open(input_file, "r") as f:
    lines = f.readlines()

# 第 2 行是 3×3 内参矩阵（fx 0 cx 0 fy cy 0 0 1）
intrinsics_line = lines[1].strip().split()
intrinsics = list(map(float, intrinsics_line))

# 解析内参
fx, _, cx, _, fy, cy, _, _, _ = intrinsics

# 构建 K 矩阵
K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# 保存为 FoundationPose / BundleSDF 标准 cam_K.txt
np.savetxt("cam_K.txt", K, fmt="%.18e")

print("成功生成 cam_K.txt：")
print(K)
