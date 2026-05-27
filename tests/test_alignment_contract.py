from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from aiws_pipeline.alignment_contract import run_alignment_from_region_proposal
from components.aiws_pipeline_contracts import validate_alignment_result


def _write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(path)


def _write_square_tube_info(tmp_path: Path) -> Path:
    info_path = tmp_path / "workpiece_priors/workpiece_info.yaml"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(
        "\n".join(
            [
                "square_tube:",
                "  component_assembly:",
                "    tube: workpiece_priors/component_assembly/tube.obj",
                "  weld_focus:",
                "    - tube",
                "  parts:",
                "    tube:",
                "      size:",
                "        - [120, 201, 120]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return info_path


def _write_region(
    sample_dir: Path,
    camera_path: Path,
    include_coarse_pose: bool = True,
) -> Path:
    region = {
        "schema_version": 1,
        "sample_id": "0034",
        "workpiece_type": "square_tube",
        "camera_path": str(camera_path),
        "rgb_path": str(sample_dir / "rgb.png"),
        "depth_path": str(sample_dir / "depth.exr"),
        "object_mask_path": str(sample_dir / "object_mask.png"),
        "focused_parts": {
            "tube": {
                "mask_path": str(sample_dir / "part_masks/tube.png"),
                "raw_size_xyz_m": [0.121, 0.201, 0.119],
                "raw_size_xyz_mm": [121.0, 201.0, 119.0],
                "size_source": "scale_net",
                "matched_size_xyz_mm": [120.0, 201.0, 120.0],
                "size_match_error": 0.01,
                "match_confidence": 0.8,
            }
        },
    }
    if include_coarse_pose:
        region["focused_parts"]["tube"]["coarse_pose_cam_4x4"] = np.eye(4).tolist()
        region["focused_parts"]["tube"]["coarse_pose_source"] = "genpose2"
    region_path = sample_dir / "region_proposal.json"
    region_path.write_text(json.dumps(region), encoding="utf-8")
    return region_path


def test_run_alignment_from_region_proposal_writes_result(tmp_path):
    template = tmp_path / "workpiece_priors/component_assembly/tube.obj"
    _write_template(template)
    info_path = _write_square_tube_info(tmp_path)
    sample_dir = tmp_path / "output/0034"
    sample_dir.mkdir(parents=True)
    camera_path = tmp_path / "workpiece_priors/camera.json"
    camera_path.write_text(
        json.dumps(
            {
                "width": 3,
                "height": 2,
                "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0},
            }
        ),
        encoding="utf-8",
    )
    region_path = _write_region(sample_dir, camera_path)

    received = {}

    def fake_runner(**kwargs):
        received.update(kwargs)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        return pose

    result_path = run_alignment_from_region_proposal(
        region_path=region_path,
        workpiece_info_path=info_path,
        repo_root=tmp_path,
        foundationpose_runner=fake_runner,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validate_alignment_result(payload)
    tube = payload["focused_parts"]["tube"]
    assert tube["pose_source"] == "foundationpose"
    assert tube["status"] == "aligned"
    assert Path(tube["scaled_cad_path"]).exists()
    assert received["rgb_path"] == sample_dir / "rgb.png"
    assert received["depth_path"] == sample_dir / "depth.exr"
    assert received["camera_path"] == camera_path
    assert received["mask_path"] == sample_dir / "part_masks/tube.png"


def test_run_alignment_records_failed_part(tmp_path):
    template = tmp_path / "workpiece_priors/component_assembly/tube.obj"
    _write_template(template)
    info_path = _write_square_tube_info(tmp_path)
    sample_dir = tmp_path / "output/0034"
    sample_dir.mkdir(parents=True)
    camera_path = tmp_path / "workpiece_priors/camera.json"
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    camera_path.write_text(
        json.dumps(
            {
                "width": 3,
                "height": 2,
                "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0},
            }
        ),
        encoding="utf-8",
    )
    region_path = _write_region(sample_dir, camera_path, include_coarse_pose=False)

    def failing_runner(**kwargs):
        raise RuntimeError("FoundationPose failed")

    result_path = run_alignment_from_region_proposal(
        region_path=region_path,
        workpiece_info_path=info_path,
        repo_root=tmp_path,
        foundationpose_runner=failing_runner,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validate_alignment_result(payload)
    assert payload["focused_parts"]["tube"]["status"] == "failed"
    assert "FoundationPose failed" in payload["focused_parts"]["tube"]["error"]
