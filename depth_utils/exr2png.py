#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 EXR 深度图 (float32, 单位 = mm) 转换为 uint16 PNG (单位 = mm)

依赖:
    pip install OpenEXR Imath opencv-python numpy

用法示例:
    # 单张转换，PNG 保存在同一目录
    python exr2png.py /path/to/depth.exr

    # 单张转换，指定输出 PNG 路径
    python exr2png.py /path/to/depth.exr --out /path/to/depth_mm.png

    # 批量转换文件夹中所有 .exr
    python exr2png.py /path/to/exr_folder

    # 批量转换，输出到单独文件夹
    python exr2png.py /path/to/exr_folder --out_dir /path/to/output_dir

    # 如果需要额外缩放 (例如 EXR 是 0.1mm 单位，可用 scale=0.1 → mm)
    python exr2png.py /path/to/depth.exr --scale 1.0
"""

import os
import argparse

import numpy as np
import cv2
import OpenEXR
import Imath


def read_exr_depth_mm(exr_path: str) -> np.ndarray:
    """读取 EXR 文件中的 Z 通道为 float32 numpy 数组 (假设单位已经是 mm)。"""
    if not os.path.isfile(exr_path):
        raise FileNotFoundError(f"EXR 文件不存在: {exr_path}")

    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = header["channels"].keys()
    if "Z" not in channels:
        raise ValueError(f"文件 {exr_path} 中没有 'Z' 通道，实际通道: {list(channels)}")

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    z_str = exr_file.channel("Z", pt)
    exr_file.close()

    depth = np.frombuffer(z_str, dtype=np.float32)
    depth = depth.reshape((height, width))

    return depth  # 单位: mm (由调用者保证)


def exr_to_uint16_png(exr_path: str, png_path: str, scale: float = 1.0) -> None:
    """
    将 EXR 深度图转换为 uint16 PNG (单位仍为 mm)。

    Args:
        exr_path: 输入 EXR 路径
        png_path: 输出 PNG 路径
        scale:   额外缩放系数，最终 depth_mm = depth_exr * scale
                 对你现在的情况，scale=1.0 即可。
    """
    depth_exr = read_exr_depth_mm(exr_path)  # float32, mm (假设)
    print(f"[INFO] 读取 EXR: {exr_path}")
    print(f"       shape={depth_exr.shape}, dtype={depth_exr.dtype}, "
          f"min={depth_exr.min():.2f}, max={depth_exr.max():.2f}")

    # 额外缩放（通常为 1.0，如果 EXR 本来就是 mm）
    depth_mm = depth_exr.astype(np.float32) * float(scale)

    # 负值设为 0
    depth_mm[depth_mm < 0] = 0.0

    # 裁剪到 uint16 范围
    depth_mm_clipped = np.clip(depth_mm, 0, 65535)

    depth_u16 = depth_mm_clipped.astype(np.uint16)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    ok = cv2.imwrite(png_path, depth_u16)
    if not ok:
        raise RuntimeError(f"[ERROR] 写 PNG 失败: {png_path}")

    print(f"[OK] 已保存 uint16(mm) 深度 PNG: {png_path}")
    print(f"     shape={depth_u16.shape}, dtype={depth_u16.dtype}, "
          f"min={depth_u16.min()}, max={depth_u16.max()}\n")


def main():
    parser = argparse.ArgumentParser(
        description="将 EXR 深度图 (float32, mm) 转换为 uint16 PNG (mm)"
    )
    parser.add_argument("input", help="EXR 文件路径或包含 EXR 的文件夹路径")
    parser.add_argument(
        "--out",
        help="当 input 是单个 EXR 文件时，可以指定输出 PNG 路径；"
             "若不指定，则输出到同目录同名 .png",
    )
    parser.add_argument(
        "--out_dir",
        help="当 input 是文件夹时，PNG 输出目录；若不指定，则输出到原 EXR 所在目录",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="深度缩放系数，最终 depth_mm = depth_exr * scale，默认 1.0",
    )

    args = parser.parse_args()
    inp = args.input
    scale = args.scale

    if os.path.isfile(inp):
        # 单个文件
        exr_path = inp
        if args.out:
            png_path = args.out
        else:
            base, _ = os.path.splitext(exr_path)
            png_path = base + ".png"

        exr_to_uint16_png(exr_path, png_path, scale=scale)

    elif os.path.isdir(inp):
        # 文件夹：批量处理 *.exr
        exr_files = [
            os.path.join(inp, f)
            for f in os.listdir(inp)
            if f.lower().endswith(".exr")
        ]
        if not exr_files:
            print(f"[WARN] 目录中没有找到 .exr 文件: {inp}")
            return

        out_dir = args.out_dir if args.out_dir else None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        print(f"[INFO] 在目录中找到 {len(exr_files)} 个 EXR 文件")
        for exr_path in sorted(exr_files):
            fname = os.path.basename(exr_path)
            base, _ = os.path.splitext(fname)
            if out_dir:
                png_path = os.path.join(out_dir, base + ".png")
            else:
                png_path = os.path.join(os.path.dirname(exr_path), base + ".png")

            exr_to_uint16_png(exr_path, png_path, scale=scale)
    else:
        print(f"[ERROR] 输入既不是文件也不是文件夹: {inp}")


if __name__ == "__main__":
    main()
