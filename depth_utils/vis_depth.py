#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
可视化各种形式的深度图（支持 EXR、EXR 伪装 PNG、PNG/TIFF 等常见格式）

请调整脚本末尾的 depth_path 变量为你的深度图路径
'''

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 尝试导入 OpenEXR（用于 EXR / EXR 伪装 PNG）
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

EXR_MAGIC_BYTES = b"\x76\x2f\x31\x01"  # OpenEXR magic number


def is_exr_file_by_magic(path: str) -> bool:
    """通过文件头（magic number）判断是否为 EXR 文件，忽略扩展名。"""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == EXR_MAGIC_BYTES
    except Exception as e:
        print(f"[ERROR] 读取文件头失败: {e}")
        return False


def read_exr_depth(path: str) -> np.ndarray:
    """
    用 OpenEXR 读取深度（Z 通道或其他通道）为 float32 numpy 数组。
    这里不做单位转换，直接返回原始数值。
    """
    if not HAS_OPENEXR:
        raise RuntimeError("未安装 OpenEXR / Imath，无法读取 EXR 文件。")

    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = list(header['channels'].keys())
    print(f"可用通道: {channels}")

    # 优先尝试常见深度通道名
    channel_name = None
    for name in ['Z', 'depth', 'Y', 'R']:
        if name in channels:
            channel_name = name
            break
    if channel_name is None:
        channel_name = channels[0]

    print(f"使用通道: {channel_name}")

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_str = exr_file.channel(channel_name, FLOAT)
    exr_file.close()

    depth = np.frombuffer(channel_str, dtype=np.float32).reshape(height, width)
    print("成功使用 OpenEXR 读取深度图")

    return depth


def read_png_like_depth(path: str) -> np.ndarray:
    """
    按普通图像读取（PNG/TIFF 等），优先用 OpenCV。
    返回单通道 numpy 数组。
    """
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("OpenCV 无法读取该图像。")

    # 如果是多通道（例如渲染出的伪彩深度），取第一通道做可视化
    if depth.ndim == 3 and depth.shape[2] > 1:
        print(f"[WARN] 深度图为多通道: shape={depth.shape}，取第 1 通道用于可视化")
        depth = depth[:, :, 0]

    return depth


def visualize_16bit_depth(depth_path: str):
    """
    读取并可视化深度图（支持：
      - 正常 PNG/TIFF 深度图
      - EXR 深度图
      - EXR 伪装为 PNG 的深度图
    ）
    """
    if not os.path.exists(depth_path):
        print(f"文件不存在: {depth_path}")
        return

    print("=== 可视化深度图 ===")
    print("路径:", depth_path)

    # 先通过文件头判断是否为 EXR
    is_exr = is_exr_file_by_magic(depth_path)

    depth = None
    src_type = None

    try:
        if is_exr:
            print("[INFO] 文件头检测为 EXR 格式（无论扩展名是什么）。")
            depth = read_exr_depth(depth_path).astype(np.float32)
            src_type = "exr"
        else:
            print("[INFO] 文件头不是 EXR，按普通图像读取（PNG/TIFF 等）。")
            depth = read_png_like_depth(depth_path)
            src_type = "png_like"

        if depth is None:
            print(f"[ERROR] 无法读取深度图: {depth_path}")
            return

        print(f"深度图形状: {depth.shape}")
        print(f"深度图数据类型: {depth.dtype}")
        print(f"深度范围: {depth.min():.2f} - {depth.max():.2f}")
        print(f"深度均值: {depth.mean():.2f}")

        # 如果是 int32 / int64，转换为 float32 以便归一化
        if depth.dtype in (np.int32, np.int64):
            depth = depth.astype(np.float32)

        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 原始深度图 (jet 色彩映射)
        im1 = axes[0, 0].imshow(depth, cmap='jet')
        axes[0, 0].set_title(f'原始深度图 (jet) [{src_type}]')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])

        # 原始深度图 (灰度)
        im2 = axes[0, 1].imshow(depth, cmap='gray')
        axes[0, 1].set_title('原始深度图 (灰度)')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])

        # 归一化深度图到 0~255，用于伪彩显示
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        im3 = axes[1, 0].imshow(depth_normalized, cmap='viridis')
        axes[1, 0].set_title('归一化深度图 (0~255)')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])

        # 深度直方图（只统计 >0 的有效深度）
        valid_depth = depth[depth > 0]
        if valid_depth.size > 0:
            axes[1, 1].hist(valid_depth.flatten(), bins=100, color='blue', alpha=0.7)
            axes[1, 1].set_title(f'深度值分布 (有效像素: {valid_depth.size})')
            axes[1, 1].set_xlabel('深度值')
            axes[1, 1].set_ylabel('像素数量')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].set_title('没有 >0 的有效深度值')
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

        return depth

    except Exception as e:
        print(f"读取/可视化深度图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 修改为你要检查的深度图路径
    depth_path = "/Users/wangrunze/Desktop/工件数据/盖板30/强光弱光/3左120度俯角/depth/0067.png"
    _ = visualize_16bit_depth(depth_path)
