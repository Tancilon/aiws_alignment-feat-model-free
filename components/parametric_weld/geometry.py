from __future__ import annotations

import numpy as np

from components.parametric_weld.context import ParametricWeldError


def robust_range(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        raise ParametricWeldError("cannot estimate range from empty values")
    a, b = np.percentile(arr, [low, high])
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        raise ParametricWeldError("degenerate observed geometry range")
    return float(a), float(b)


def make_path(
    *,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n: np.ndarray,
    segments: list[dict],
    closed: bool,
) -> dict:
    centerline = []
    for segment in segments:
        for point in segment["points_2d"]:
            centerline.append(np.asarray(point, dtype=np.float64))
    return {
        "centerline_2d": np.asarray(centerline, dtype=np.float64),
        "plane": {
            "origin": np.asarray(origin, dtype=np.float64),
            "u": _unit(u),
            "v": _unit(v),
            "n": _unit(n),
            "planarity": 1.0,
        },
        "fitted": segments,
        "closed": bool(closed),
    }


def line_segment(p0: np.ndarray, p1: np.ndarray, index: int) -> dict:
    return {
        "type": "line",
        "points_2d": [np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)],
        "indices": (index, index + 1),
        "fitting_error_mm": 0.0,
    }


def arc_segment(p0: np.ndarray, pm: np.ndarray, p1: np.ndarray, index: int) -> dict:
    return {
        "type": "arc",
        "points_2d": [
            np.asarray(p0, dtype=np.float64),
            np.asarray(pm, dtype=np.float64),
            np.asarray(p1, dtype=np.float64),
        ],
        "indices": (index, index + 1),
        "fitting_error_mm": 0.0,
    }


def validate_size_m(size_m: np.ndarray) -> np.ndarray:
    size = np.asarray(size_m, dtype=np.float64)
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ParametricWeldError("invalid matched workpiece size")
    return size


def rounded_rectangle_path(
    *,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n: np.ndarray,
    width: float,
    depth: float,
    radius_fraction: float = 0.08,
) -> dict:
    if width <= 0 or depth <= 0:
        raise ParametricWeldError("invalid rounded rectangle extents")
    half_w = 0.5 * float(width)
    half_d = 0.5 * float(depth)
    min_extent = min(float(width), float(depth))
    radius = float(np.clip(min_extent * radius_fraction, min_extent * 0.01, min_extent * 0.35))

    x0, x1 = -half_w, half_w
    z0, z1 = -half_d, half_d
    r = radius

    top_l = np.array([x0 + r, z1])
    top_r = np.array([x1 - r, z1])
    right_t = np.array([x1, z1 - r])
    right_b = np.array([x1, z0 + r])
    bot_r = np.array([x1 - r, z0])
    bot_l = np.array([x0 + r, z0])
    left_b = np.array([x0, z0 + r])
    left_t = np.array([x0, z1 - r])
    diag = r / np.sqrt(2.0)

    return make_path(
        origin=np.asarray(origin, dtype=np.float64),
        u=np.asarray(u, dtype=np.float64),
        v=np.asarray(v, dtype=np.float64),
        n=np.asarray(n, dtype=np.float64),
        segments=[
            line_segment(top_l, top_r, 0),
            arc_segment(top_r, np.array([x1 - r + diag, z1 - r + diag]), right_t, 1),
            line_segment(right_t, right_b, 2),
            arc_segment(right_b, np.array([x1 - r + diag, z0 + r - diag]), bot_r, 3),
            line_segment(bot_r, bot_l, 4),
            arc_segment(bot_l, np.array([x0 + r - diag, z0 + r - diag]), left_b, 5),
            line_segment(left_b, left_t, 6),
            arc_segment(left_t, np.array([x0 + r - diag, z1 - r + diag]), top_l, 7),
        ],
        closed=True,
    )


def rectangle_path(
    *,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n: np.ndarray,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> dict:
    p00 = np.array([u_min, v_min], dtype=np.float64)
    p10 = np.array([u_max, v_min], dtype=np.float64)
    p11 = np.array([u_max, v_max], dtype=np.float64)
    p01 = np.array([u_min, v_max], dtype=np.float64)
    return make_path(
        origin=origin,
        u=u,
        v=v,
        n=n,
        segments=[
            line_segment(p00, p10, 0),
            line_segment(p10, p11, 1),
            line_segment(p11, p01, 2),
            line_segment(p01, p00, 3),
        ],
        closed=True,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ParametricWeldError("invalid plane axis")
    return arr / norm
