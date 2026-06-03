from __future__ import annotations

import numpy as np

from components.parametric_weld.context import ParametricWeldContext, ParametricWeldError
from components.parametric_weld.geometry import arc_segment, line_segment, make_path, robust_range


class HBeamParametricWeldStrategy:
    min_points = 30

    def process_scene(self, context: ParametricWeldContext) -> list[dict]:
        points = context.object_points()
        if len(points) < self.min_points:
            raise ParametricWeldError("insufficient points for H_beam parametric fitting")

        plate_points = self._select_middle_plate_points(points, context.size_m)
        x_min, x_max = robust_range(plate_points[:, 0], low=25.0, high=75.0)
        surface_tol = max((x_max - x_min) * 0.2, 0.002)
        surface_points = plate_points[
            (np.abs(plate_points[:, 0] - x_min) <= surface_tol)
            | (np.abs(plate_points[:, 0] - x_max) <= surface_tol)
        ]
        if len(surface_points) < self.min_points:
            surface_points = plate_points
        y_min, y_max = robust_range(surface_points[:, 1], low=5.0, high=95.0)
        z_min, z_max = robust_range(surface_points[:, 2], low=5.0, high=95.0)
        if (z_max - z_min) < 0.02 or (y_max - y_min) < 0.01:
            raise ParametricWeldError("H_beam observed geometry is too small")

        x_offsets = self._estimate_side_offsets(plate_points[:, 0], x_min, x_max, context.size_m)
        return [
            self._open_u_path(x_offset, y_min, y_max, z_min, z_max, flip=idx == 1)
            for idx, x_offset in enumerate(x_offsets)
        ]

    @staticmethod
    def _select_middle_plate_points(points: np.ndarray, size_m: np.ndarray) -> np.ndarray:
        x_low, x_high = robust_range(points[:, 0])
        x_center = float(np.median(points[:, 0]))
        size = np.asarray(size_m, dtype=np.float64).reshape(3)
        half_band = max((x_high - x_low) * 0.08, float(size[0]) * 0.05, 0.01)
        central = points[np.abs(points[:, 0] - x_center) <= half_band]
        if len(central) < HBeamParametricWeldStrategy.min_points:
            central = points[np.argsort(np.abs(points[:, 0] - x_center))[: HBeamParametricWeldStrategy.min_points]]
        return central

    @staticmethod
    def _estimate_side_offsets(
        x_values: np.ndarray,
        x_min: float,
        x_max: float,
        size_m: np.ndarray,
    ) -> list[float]:
        if x_max - x_min >= 0.004:
            return [float(x_min), float(x_max)]

        center = float(np.median(x_values))
        size = np.asarray(size_m, dtype=np.float64).reshape(3)
        half_gap = max(float(size[0]) * 0.03, 0.003)
        return [center - half_gap, center + half_gap]

    @staticmethod
    def _open_u_path(
        x: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        *,
        flip: bool,
    ) -> dict:
        cy = 0.5 * (y_min + y_max)
        cz = 0.5 * (z_min + z_max)
        z0, z1 = z_min - cz, z_max - cz
        lower = -(y_max - cy)
        upper = -(y_min - cy)
        radius = min((z1 - z0), (upper - lower)) * 0.18
        radius = float(np.clip(radius, 0.002, max(0.002, (upper - lower) * 0.4)))
        if z1 - z0 <= 2.0 * radius:
            radius = max((z1 - z0) * 0.25, 0.001)

        p0 = np.array([z0, lower])
        p1 = np.array([z0, upper - radius])
        p2 = np.array([z0 + radius, upper])
        p3 = np.array([z1 - radius, upper])
        p4 = np.array([z1, upper - radius])
        p5 = np.array([z1, lower])
        segments = [
            line_segment(p0, p1, 0),
            arc_segment(p1, np.array([z0, upper]), p2, 1),
            line_segment(p2, p3, 2),
            arc_segment(p3, np.array([z1, upper]), p4, 3),
            line_segment(p4, p5, 4),
        ]
        if flip:
            segments = [
                dict(segment, points_2d=list(reversed(segment["points_2d"])))
                for segment in reversed(segments)
            ]
        return make_path(
            origin=np.array([x, cy, cz], dtype=np.float64),
            u=np.array([0.0, 0.0, 1.0]),
            v=np.array([0.0, -1.0, 0.0]),
            n=np.array([1.0, 0.0, 0.0]),
            segments=segments,
            closed=False,
        )
