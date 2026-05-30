from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

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


def _write_bellmouth_info(tmp_path: Path) -> Path:
    info_path = tmp_path / "workpiece_priors/workpiece_info.yaml"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(
        "\n".join(
            [
                "bellmouth:",
                "  component_assembly:",
                "    tube: workpiece_priors/component_assembly/tube.obj",
                "  weld_focus:",
                "    - tube",
                "  parts:",
                "    tube:",
                "      size:",
                "        - [75, 150, 75]",
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
    weld_result = payload["weld_result"]
    assert weld_result["status"] == "extracted"
    weld_json_path = Path(weld_result["weld_json_path"])
    assert weld_json_path.exists()
    weld_json = json.loads(weld_json_path.read_text(encoding="utf-8"))
    assert weld_json["model"] == "square_tube"
    assert weld_json["extraction_method"] == "parametric_cad"
    assert weld_json["weld_side_selection"]["selected_side"] == "+Y"
    assert weld_result["weld_side_selection"] == weld_json["weld_side_selection"]
    assert len(weld_json["weld_paths"]) == 1
    assert len(weld_json["weld_paths"][0]["segments"]) == 8
    assert received["rgb_path"] == sample_dir / "rgb.png"
    assert received["depth_path"] == sample_dir / "depth.exr"
    assert received["camera_path"] == camera_path
    assert received["mask_path"] == sample_dir / "part_masks/tube.png"


def test_run_alignment_passes_scene_inputs_to_weld_extractor(tmp_path, monkeypatch):
    template = tmp_path / "workpiece_priors/component_assembly/tube.obj"
    _write_template(template)
    info_path = _write_bellmouth_info(tmp_path)
    sample_dir = tmp_path / "output/l75"
    (sample_dir / "part_masks").mkdir(parents=True)
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
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(sample_dir / "rgb.png")
    Image.fromarray(np.full((2, 3), 1000, dtype=np.uint16)).save(sample_dir / "depth.png")
    Image.fromarray(np.array([[0, 255, 0], [0, 255, 0]], dtype=np.uint8)).save(
        sample_dir / "part_masks/tube.png"
    )
    Image.fromarray(np.full((2, 3), 255, dtype=np.uint8)).save(sample_dir / "object_mask.png")
    region = {
        "schema_version": 1,
        "sample_id": "l75",
        "workpiece_type": "bellmouth",
        "camera_path": str(camera_path),
        "rgb_path": str(sample_dir / "rgb.png"),
        "depth_path": str(sample_dir / "depth.png"),
        "object_mask_path": str(sample_dir / "object_mask.png"),
        "focused_parts": {
            "tube": {
                "mask_path": str(sample_dir / "part_masks/tube.png"),
                "raw_size_xyz_m": [0.075, 0.150, 0.075],
                "raw_size_xyz_mm": [75.0, 150.0, 75.0],
                "size_source": "scale_net",
                "matched_size_xyz_mm": [75.0, 150.0, 75.0],
                "size_match_error": 0.01,
                "match_confidence": 0.8,
                "coarse_pose_cam_4x4": np.eye(4).tolist(),
                "coarse_pose_source": "genpose2",
            }
        },
    }
    region_path = sample_dir / "region_proposal.json"
    region_path.write_text(json.dumps(region), encoding="utf-8")

    def fake_runner(**kwargs):
        return np.eye(4, dtype=np.float64)

    seen = {}

    class FakeWeldPoseExtractor:
        def __init__(self, **kwargs):
            seen["output_dir"] = kwargs["output_dir"]

        def extract(self, **kwargs):
            seen.update(kwargs)
            weld_path = seen["output_dir"] / "rgb_weld_paths.json"
            weld_path.parent.mkdir(parents=True, exist_ok=True)
            weld_path.write_text("{}", encoding="utf-8")
            return {
                "weld_json_path": str(weld_path),
                "bellmouth_weld_contact_selection": {
                    "selected_face": "+X",
                    "method": "rgbd_contact_support_v1",
                    "ambiguous": False,
                    "score_margin": 0.2,
                    "candidate_scores": [],
                    "span_y_m": [-0.03, 0.04],
                },
            }

    monkeypatch.setattr("aiws_pipeline.alignment_contract.WeldPoseExtractor", FakeWeldPoseExtractor)

    result_path = run_alignment_from_region_proposal(
        region_path=region_path,
        workpiece_info_path=info_path,
        repo_root=tmp_path,
        foundationpose_runner=fake_runner,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert seen["rgb"].shape == (2, 3, 3)
    assert seen["depth"].shape == (2, 3)
    assert seen["mask"].shape == (2, 3)
    assert seen["mask"].dtype == np.bool_
    assert seen["object_mask"].shape == (2, 3)
    assert seen["object_mask"].dtype == np.bool_
    assert payload["weld_result"]["bellmouth_weld_contact_selection"]["selected_face"] == "+X"


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
