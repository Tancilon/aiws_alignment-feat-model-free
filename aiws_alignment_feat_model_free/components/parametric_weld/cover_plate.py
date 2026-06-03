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


CAD_OUTER_HALF_EXTENT = 0.5
CAD_INNER_TOP_HALF_EXTENT = 0.455645
CAD_WALL_MIDLINE_SCALE = 0.955645


class CoverPlateParametricWeldStrategy:
    def process_scene(self, context: ParametricWeldContext) -> list[dict]:
        size = validate_size_m(context.size_m)
        selection = select_image_vertical_side(
            size_m=size,
            refined_pose=context.refined_pose,
            intrinsics=context.intrinsics,
            prefer="top",
            default_side_sign=1.0,
        )
        side = float(selection.side_sign)
        context.metadata["weld_side_selection"] = selection.as_metadata()
        context.metadata["cover_plate_weld_offset"] = {
            "method": "cad_wall_midline_v1",
            "outer_half_extent": CAD_OUTER_HALF_EXTENT,
            "inner_half_extent": CAD_INNER_TOP_HALF_EXTENT,
            "scale": CAD_WALL_MIDLINE_SCALE,
        }
        return [
            rounded_rectangle_path(
                origin=np.array([0.0, side * 0.5 * size[1], 0.0], dtype=np.float64),
                width=float(size[0]) * CAD_WALL_MIDLINE_SCALE,
                depth=float(size[2]) * CAD_WALL_MIDLINE_SCALE,
                u=np.array([1.0, 0.0, 0.0]),
                v=np.array([0.0, 0.0, 1.0]),
                n=np.array([0.0, side, 0.0]),
            )
        ]
