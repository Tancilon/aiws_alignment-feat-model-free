from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np

from aiws_alignment_feat_model_free.components.parametric_weld.pointcloud import (
    depth_mask_to_object_points,
)


class ParametricWeldError(RuntimeError):
    """Raised when scene geometry is insufficient for parametric weld fitting."""


@dataclass
class ParametricWeldContext:
    class_name: str
    size_m: np.ndarray
    refined_pose: np.ndarray
    intrinsics: dict[str, Any]
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    mask: np.ndarray | None = None
    object_mask: np.ndarray | None = None
    object_points_m: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def object_points(self) -> np.ndarray:
        if self.object_points_m is not None:
            points = np.asarray(self.object_points_m, dtype=np.float64)
        else:
            if self.depth is None or self.mask is None:
                raise ParametricWeldError("depth and mask are required for parametric weld fitting")
            points = depth_mask_to_object_points(
                depth=self.depth,
                mask=self.mask,
                intrinsics=self.intrinsics,
                refined_pose=self.refined_pose,
            )

        if points.ndim != 2 or points.shape[1] != 3:
            raise ParametricWeldError("object points must have shape Nx3")
        finite = np.all(np.isfinite(points), axis=1)
        points = points[finite]
        if len(points) == 0:
            raise ParametricWeldError("no valid object points for parametric weld fitting")
        return points
