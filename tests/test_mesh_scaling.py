from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiws_pipeline.mesh_scaling import scale_mesh_to_size_mm


def test_scale_mesh_to_size_mm_writes_metric_obj(tmp_path):
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 4.0))
    template = tmp_path / "template.obj"
    mesh.export(template)
    output = tmp_path / "scaled.obj"

    scale_mesh_to_size_mm(template, output, [100.0, 200.0, 400.0])

    scaled = trimesh.load(output, force="mesh", process=False)
    np.testing.assert_allclose(scaled.extents, [0.1, 0.2, 0.4], atol=1e-6)


def test_scale_mesh_to_size_mm_rejects_degenerate_template(tmp_path):
    template = tmp_path / "bad.obj"
    template.write_text("v 0 0 0\nv 0 0 0\nv 0 0 0\nf 1 2 3\n", encoding="utf-8")

    try:
        scale_mesh_to_size_mm(template, tmp_path / "scaled.obj", [100.0, 200.0, 300.0])
    except ValueError as exc:
        assert "degenerate" in str(exc)
    else:
        raise AssertionError("expected ValueError")
