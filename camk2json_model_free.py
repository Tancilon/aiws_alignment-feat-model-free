#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from glob import glob

'''
usage:
python camk2json.py \
  --data_dir /home/wrz/data/my_scene \
  --cam_k /home/wrz/data/my_scene/cam_K.txt
'''

def read_K_txt(cam_k_path: str):
    """
    读取 3x3 相机内参矩阵（文本里每行3个数）
    返回：list[float]，长度=9，按行优先 flatten
    """
    with open(cam_k_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    rows = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 3:
            raise ValueError(f"cam_K.txt 每行应有3个数，但发现：{ln}")
        rows.append([float(x) for x in parts])

    if len(rows) != 3:
        raise ValueError(f"cam_K.txt 应该是3行，但实际是 {len(rows)} 行")

    # row-major flatten
    K_flat = [rows[r][c] for r in range(3) for c in range(3)]
    return K_flat

def list_rgb_images(rgb_dir: str):
    """
    列出 rgb_dir 下常见图片文件，并按文件名排序
    """
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(rgb_dir, e)))
    files = sorted(files)
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_k", type=str, default="/mnt/data/cam_K.txt",
                        help="cam_K.txt 路径（3x3矩阵）")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据根目录：其下应包含 rgb/ 文件夹")
    parser.add_argument("--rgb_dir", type=str, default=None,
                        help="可选：直接指定 rgb 文件夹路径；默认用 data_dir/rgb")
    parser.add_argument("--out", type=str, default=None,
                        help="输出 json 路径；默认写到 data_dir/scene_camera.json")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                        help='写入 json 的 depth_scale，默认 1.0（参考示例格式）')
    args = parser.parse_args()

    rgb_dir = args.rgb_dir if args.rgb_dir is not None else os.path.join(args.data_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"找不到 rgb 文件夹：{rgb_dir}")

    out_json = args.out if args.out is not None else os.path.join(args.data_dir, "scene_camera.json")

    K_flat = read_K_txt(args.cam_k)
    rgb_files = list_rgb_images(rgb_dir)
    print(f"[INFO] rgb_dir: {rgb_dir}")
    print(f"[INFO] Found {len(rgb_files)} RGB images.")

    scene_camera = {}
    for fp in rgb_files:
        # key 用文件名（去扩展名），比如 000001.png -> "000001"
        key = os.path.splitext(os.path.basename(fp))[0]
        scene_camera[key] = {
            "cam_K": K_flat,
            "depth_scale": float(args.depth_scale),
        }

    # 写出 json（保持可读）
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scene_camera, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Wrote: {out_json}")
    print(f"[INFO] Example entry: {next(iter(scene_camera.items())) if scene_camera else 'EMPTY'}")

if __name__ == "__main__":
    main()
