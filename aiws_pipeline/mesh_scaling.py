from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh


def scale_mesh_to_size_mm(
    template_path: str | Path,
    output_path: str | Path,
    target_size_xyz_mm: Iterable[float],
) -> Path:
    template_path = Path(template_path)
    output_path = Path(output_path)
    target_m = np.asarray(
        [float(value) for value in target_size_xyz_mm], dtype=np.float64
    ) / 1000.0
    if target_m.shape != (3,) or np.any(target_m <= 0):
        raise ValueError("target_size_xyz_mm must contain 3 positive values")

    mesh = trimesh.load(template_path, force="mesh", process=False)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if extents.shape != (3,) or np.any(extents <= 1e-12):
        raise ValueError(f"degenerate template extents for {template_path}: {extents.tolist()}")

    scale = target_m / extents
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = scale[0]
    transform[1, 1] = scale[1]
    transform[2, 2] = scale[2]

    scaled = mesh.copy()
    scaled.apply_transform(transform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scaled.export(output_path)
    return output_path
