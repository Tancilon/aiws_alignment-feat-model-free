from __future__ import annotations

import numpy as np

from aiws_alignment_feat_model_free.components.parametric_weld.context import (
    ParametricWeldContext,
)
from aiws_alignment_feat_model_free.components.parametric_weld.geometry import (
    rounded_rectangle_path,
    validate_size_m,
)
from aiws_alignment_feat_model_free.components.parametric_weld.side_selection import (
    select_image_vertical_side,
)


class SquareTubeParametricWeldStrategy:
    def process_scene(self, context: ParametricWeldContext) -> list[dict]:
        size = validate_size_m(context.size_m)
        selection = select_image_vertical_side(
            size_m=size,
            refined_pose=context.refined_pose,
            intrinsics=context.intrinsics,
            prefer="bottom",
            default_side_sign=-1.0,
        )
        side = float(selection.side_sign)
        context.metadata["weld_side_selection"] = selection.as_metadata()
        return [
            rounded_rectangle_path(
                origin=np.array([0.0, side * 0.5 * size[1], 0.0], dtype=np.float64),
                width=float(size[0]),
                depth=float(size[2]),
                u=np.array([1.0, 0.0, 0.0]),
                v=np.array([0.0, 0.0, 1.0]),
                n=np.array([0.0, side, 0.0]),
            )
        ]
