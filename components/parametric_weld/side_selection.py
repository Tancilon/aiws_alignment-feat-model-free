from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


AMBIGUOUS_MARGIN_PX = 25.0


@dataclass(frozen=True)
class ImageVerticalSideSelection:
    side_sign: float
    selected_side: str
    method: str
    preference: str
    candidate_mean_v_px: dict[str, float] | None = None
    score_margin_px: float | None = None
    ambiguous: bool = False
    fallback_reason: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "selected_side": self.selected_side,
            "method": self.method,
            "preference": self.preference,
            "ambiguous": bool(self.ambiguous),
        }
        if self.candidate_mean_v_px is not None:
            payload["candidate_mean_v_px"] = self.candidate_mean_v_px
        if self.score_margin_px is not None:
            payload["score_margin_px"] = self.score_margin_px
        if self.fallback_reason is not None:
            payload["fallback_reason"] = self.fallback_reason
        return payload


def select_image_vertical_side(
    *,
    size_m: np.ndarray,
    refined_pose: np.ndarray,
    intrinsics: dict[str, Any],
    prefer: str,
    default_side_sign: float,
    ambiguous_margin_px: float = AMBIGUOUS_MARGIN_PX,
) -> ImageVerticalSideSelection:
    if prefer not in {"top", "bottom"}:
        raise ValueError("prefer must be top or bottom")

    default_side = _side_name(default_side_sign)
    try:
        size = _validate_size(size_m)
        pose = _validate_pose(refined_pose)
        fy = _intrinsic(intrinsics, "fy")
        cy = _intrinsic(intrinsics, "cy")
        candidate_v = {
            "+Y": _mean_face_v(size=size, pose=pose, fy=fy, cy=cy, side_sign=1.0),
            "-Y": _mean_face_v(size=size, pose=pose, fy=fy, cy=cy, side_sign=-1.0),
        }
    except Exception as exc:
        return ImageVerticalSideSelection(
            side_sign=float(default_side_sign),
            selected_side=default_side,
            method="image_vertical_endpoint_v1",
            preference=f"image_{prefer}",
            ambiguous=True,
            fallback_reason=str(exc),
        )

    if prefer == "top":
        selected_side = min(candidate_v, key=candidate_v.get)
    else:
        selected_side = max(candidate_v, key=candidate_v.get)
    other_side = "-Y" if selected_side == "+Y" else "+Y"
    margin = abs(candidate_v[selected_side] - candidate_v[other_side])
    ambiguous = margin < float(ambiguous_margin_px)
    return ImageVerticalSideSelection(
        side_sign=1.0 if selected_side == "+Y" else -1.0,
        selected_side=selected_side,
        method="image_vertical_endpoint_v1",
        preference=f"image_{prefer}",
        candidate_mean_v_px={key: round(float(value), 3) for key, value in candidate_v.items()},
        score_margin_px=round(float(margin), 3),
        ambiguous=bool(ambiguous),
    )


def _side_name(side_sign: float) -> str:
    return "+Y" if float(side_sign) >= 0.0 else "-Y"


def _validate_size(size_m: np.ndarray) -> np.ndarray:
    size = np.asarray(size_m, dtype=np.float64)
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ValueError("invalid matched workpiece size")
    return size


def _validate_pose(refined_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(refined_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("invalid refined pose")
    return pose


def _intrinsic(intrinsics: dict[str, Any], key: str) -> float:
    value = float(intrinsics[key])
    if not np.isfinite(value) or value == 0.0:
        raise ValueError(f"invalid camera intrinsic: {key}")
    return value


def _mean_face_v(
    *,
    size: np.ndarray,
    pose: np.ndarray,
    fy: float,
    cy: float,
    side_sign: float,
) -> float:
    half_x = 0.5 * size[0]
    half_y = 0.5 * size[1]
    half_z = 0.5 * size[2]
    y = float(side_sign) * half_y
    local_points = np.asarray(
        [
            [0.0, y, 0.0],
            [-half_x, y, -half_z],
            [-half_x, y, half_z],
            [half_x, y, -half_z],
            [half_x, y, half_z],
        ],
        dtype=np.float64,
    )
    camera_points = local_points @ pose[:3, :3].T + pose[:3, 3]
    z = camera_points[:, 2]
    valid = np.isfinite(z) & (z > 1e-8)
    if np.count_nonzero(valid) == 0:
        raise ValueError("candidate face is not projectable")
    v = fy * camera_points[valid, 1] / z[valid] + cy
    if not np.all(np.isfinite(v)):
        raise ValueError("candidate face projection is not finite")
    return float(np.mean(v))
