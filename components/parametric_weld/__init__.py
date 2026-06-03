from __future__ import annotations

from components.parametric_weld.bellmouth import BellmouthParametricWeldStrategy
from components.parametric_weld.context import ParametricWeldContext, ParametricWeldError
from components.parametric_weld.cover_plate import CoverPlateParametricWeldStrategy
from components.parametric_weld.pointcloud import depth_mask_to_object_points
from components.parametric_weld.square_tube import SquareTubeParametricWeldStrategy


def get_parametric_strategy(model_name: str):
    normalized = str(model_name)
    strategies = {
        "cover_plate": CoverPlateParametricWeldStrategy,
        "square_tube": SquareTubeParametricWeldStrategy,
        "bellmouth": BellmouthParametricWeldStrategy,
    }
    cls = strategies.get(normalized)
    if cls is None:
        return None
    return cls()


__all__ = [
    "BellmouthParametricWeldStrategy",
    "CoverPlateParametricWeldStrategy",
    "ParametricWeldContext",
    "ParametricWeldError",
    "SquareTubeParametricWeldStrategy",
    "depth_mask_to_object_points",
    "get_parametric_strategy",
]
