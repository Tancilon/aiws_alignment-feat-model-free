#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
检查深度图文件的格式和数值范围，支持 EXR 和常见图

用法示例:
    python check_depth.py /path/to/depth.png
    python check_depth.py /path/to/depth_mm.png
    python check_depth.py /path/to/depth.exr
'''



import os
import sys
import cv2
import numpy as np

# 尝试导入 OpenEXR，用于真正的 EXR 解析（包括“伪装成 png 的 EXR”）
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False


EXR_MAGIC_BYTES = b"\x76\x2f\x31\x01"  # OpenEXR 规范中的 magic number


def is_exr_file_by_magic(path: str) -> bool:
    """通过前 4 个字节判断是否为 EXR 文件（与扩展名无关）"""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == EXR_MAGIC_BYTES
    except Exception as e:
        print(f"[ERROR] 读取文件头失败: {e}")
        return False


def inspect_exr(path: str):
    """读取 EXR 文件的 Z 通道并打印统计信息"""
    if not HAS_OPENEXR:
        print("[ERROR] 未安装 OpenEXR / Imath，无法解析 EXR。")
        print("        请先: pip install OpenEXR Imath")
        return

    print(">>> 作为 EXR 解析（Z 通道）")

    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = list(header["channels"].keys())
    print(f"EXR 通道: {channels}")

    # 优先使用 'Z'，否则用第一个通道
    channel_name = "Z" if "Z" in channels else channels[0]
    print(f"使用通道: {channel_name}")

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    z_str = exr_file.channel(channel_name, pt)
    exr_file.close()

    depth = np.frombuffer(z_str, dtype=np.float32).reshape((height, width))

    print("=== 深度信息 (EXR) ===")
    print("dtype:", depth.dtype)
    print("shape:", depth.shape)
    print("min:", float(depth.min()))
    print("max:", float(depth.max()))
    print("mean:", float(depth.mean()))

    # 简单单位推断（只是提示）
    dmax = float(depth.max())
    if dmax > 0 and dmax < 50:
        print("推测: 浮点深度，数值范围较小，可能单位为米 (m)")
    elif dmax > 0 and dmax < 10000:
        print("推测: 浮点深度，范围在 0~几千，可能已经是毫米 (mm)")
    else:
        print("提示: 数值范围较大，可能需要结合 scale_unit 做换算。")


def inspect_png_like(path: str):
    """按普通图像（PNG/TIFF 等）读取并检查"""
    print(">>> 作为普通图像读取（PNG/TIFF 等）")

    try:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except cv2.error as e:
        print("[ERROR] OpenCV 读取失败:", e)
        return

    if depth is None:
        print("[ERROR] OpenCV 返回 None，无法读取此图像。")
        return

    print("=== 深度信息 (cv2.IMREAD_UNCHANGED) ===")
    print("dtype:", depth.dtype)
    print("shape:", depth.shape)

    # 统计值
    dmin = depth.min()
    dmax = depth.max()
    print("min:", int(dmin) if np.issubdtype(depth.dtype, np.integer) else float(dmin))
    print("max:", int(dmax) if np.issubdtype(depth.dtype, np.integer) else float(dmax))

    # 简单类型判断
    if depth.ndim == 2 or (depth.ndim == 3 and depth.shape[2] == 1):
        # 单通道
        if depth.dtype == np.uint16:
            print("推测: 16-bit 单通道深度图 (uint16)。")
            if dmax > 0 and dmax < 10000:
                print("      数值范围在 0~几千，极大可能是 'uint16 + 毫米 (mm)'。")
            else:
                print("      数值范围较大/较怪，单位需结合相机文档确认。")
        elif depth.dtype in (np.float32, np.float64):
            print("推测: 浮点深度图。")
            if dmax > 0 and dmax < 50:
                print("      数值较小，可能是单位为米 (m) 的深度。")
            elif dmax > 0 and dmax < 10000:
                print("      数值在 0~几千，可能是毫米 (mm) 或附加 scale 的深度。")
        else:
            print(f"注意: 单通道但 dtype={depth.dtype}，可能是某种特殊编码。")
    else:
        # 多通道图
        print("注意: 这是多通道图像 (shape[2] = {}).".format(depth.shape[2]))
        print("      很可能是渲染后的可视化结果，而不是真实 metric 深度。")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # 默认路径，按需修改
        path = "/Users/wangrunze/Desktop/工件数据/test_photo/001/depth_mm.png"

    print("=== 检查文件 ===")
    print("路径:", path)

    if not os.path.isfile(path):
        print("[ERROR] 文件不存在")
        return

    # 先看文件头是不是 EXR
    is_exr = is_exr_file_by_magic(path)

    ext = os.path.splitext(path)[1].lower()
    if is_exr:
        print(f"[INFO] 文件头检测到 OpenEXR magic number (0x76 0x2f 0x31 0x01)。")
        if ext != ".exr":
            print(f"[WARN] 扩展名是 '{ext}'，但文件头是 EXR → 很可能是 EXR 伪装成 {ext}。")
        inspect_exr(path)
    else:
        print(f"[INFO] 文件头不是 EXR magic，按普通图像处理（扩展名: {ext}）。")
        inspect_png_like(path)


if __name__ == "__main__":
    main()
