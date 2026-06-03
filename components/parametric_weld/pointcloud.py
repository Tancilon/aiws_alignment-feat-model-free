from __future__ import annotations

from typing import Any

import numpy as np


def _intrinsic_value(intrinsics: dict[str, Any], key: str) -> float:
    try:
        value = float(intrinsics[key])
    except Exception as exc:
        raise ValueError(f"missing camera intrinsic: {key}") from exc
    if not np.isfinite(value) or value == 0.0:
        raise ValueError(f"invalid camera intrinsic: {key}")
    return value


def depth_mask_to_object_points(
    *,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict[str, Any],
    refined_pose: np.ndarray,
) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float64)
    mask_arr = np.asarray(mask)
    if depth_arr.ndim != 2:
        raise ValueError("depth must be a 2D array")
    if mask_arr.shape != depth_arr.shape:
        raise ValueError("mask shape must match depth shape")

    fx = _intrinsic_value(intrinsics, "fx")
    fy = _intrinsic_value(intrinsics, "fy")
    cx = _intrinsic_value(intrinsics, "cx")
    cy = _intrinsic_value(intrinsics, "cy")

    valid = (mask_arr > 0) & np.isfinite(depth_arr) & (depth_arr > 0.0)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = depth_arr[ys, xs]
    camera_points = np.column_stack(
        [
            (xs.astype(np.float64) - cx) * z / fx,
            (ys.astype(np.float64) - cy) * z / fy,
            z,
        ]
    )

    pose = np.asarray(refined_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("refined_pose must be a finite 4x4 matrix")
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return (camera_points - translation.reshape(1, 3)) @ rotation
