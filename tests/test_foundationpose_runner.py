from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from aiws_pipeline.foundationpose_runner import run_foundationpose_part


class FakeEstimator:
    received = {}

    def __init__(self, **kwargs):
        FakeEstimator.received["init"] = kwargs

    def register(self, **kwargs):
        FakeEstimator.received["register"] = kwargs
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        return pose


def test_run_foundationpose_part_reads_contract_files(monkeypatch, tmp_path):
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "tube.png"
    depth_path = tmp_path / "depth.npy"
    camera_path = tmp_path / "camera.json"
    cad_path = tmp_path / "tube.obj"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.array([[0, 255, 0], [0, 255, 0]], dtype=np.uint8)).save(mask_path)
    np.save(depth_path, np.ones((2, 3), dtype=np.float32) * 0.5)
    cad_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    camera_path.write_text(
        json.dumps(
            {
                "intrinsics": {"fx": 1.0, "fy": 2.0, "cx": 3.0, "cy": 4.0},
                "width": 3,
                "height": 2,
            }
        ),
        encoding="utf-8",
    )

    fake_runtime = types.SimpleNamespace(
        FoundationPose=FakeEstimator,
        ScorePredictor=lambda: object(),
        PoseRefinePredictor=lambda: object(),
        RasterizeCudaContext=lambda: object(),
        load_mesh=lambda path: types.SimpleNamespace(
            vertices=np.zeros((3, 3), dtype=np.float32),
            vertex_normals=np.ones((3, 3), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        "aiws_pipeline.foundationpose_runner._load_foundationpose_runtime",
        lambda: fake_runtime,
    )
    monkeypatch.setattr(
        "aiws_pipeline.foundationpose_runner.load_depth",
        lambda path: np.load(path),
    )

    pose = run_foundationpose_part(
        cad_path=cad_path,
        scene_dir=tmp_path,
        debug_dir=tmp_path / "debug",
        mask_path=mask_path,
        rgb_path=rgb_path,
        depth_path=depth_path,
        camera_path=camera_path,
    )

    np.testing.assert_allclose(pose[:3, 3], [0.1, 0.2, 0.3])
    register = FakeEstimator.received["register"]
    assert register["rgb"].shape == (2, 3, 3)
    assert register["depth"].shape == (2, 3)
    assert register["ob_mask"].sum() == 2
    np.testing.assert_allclose(
        register["K"],
        [[1.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]],
    )
