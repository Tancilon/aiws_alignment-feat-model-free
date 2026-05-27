from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aiws_pipeline.foundationpose_runner import run_foundationpose_part
from aiws_pipeline.mesh_scaling import scale_mesh_to_size_mm
from components.aiws_pipeline_contracts import (
    validate_alignment_result,
    validate_region_proposal,
)
from components.workpiece_priors import WorkpiecePriorRegistry


def _resolve(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def run_alignment_from_region_proposal(
    region_path: str | Path,
    workpiece_info_path: str | Path,
    repo_root: str | Path,
    foundationpose_runner: Any = run_foundationpose_part,
) -> Path:
    repo_root = Path(repo_root).resolve()
    region_path = Path(region_path).resolve()
    region_payload = json.loads(region_path.read_text(encoding="utf-8"))
    validate_region_proposal(region_payload)

    registry = WorkpiecePriorRegistry(workpiece_info_path, repo_root=repo_root)
    sample_dir = region_path.parent
    scaled_cad_dir = sample_dir / "scaled_cad"
    debug_dir = sample_dir / "debug"
    category = region_payload["workpiece_type"]

    focused_results: dict[str, dict[str, Any]] = {}
    for part_name, part_payload in region_payload["focused_parts"].items():
        try:
            cad_template = registry.component_template(category, part_name)
            scaled_cad_path = scaled_cad_dir / f"{part_name}.obj"
            scale_mesh_to_size_mm(
                template_path=cad_template,
                output_path=scaled_cad_path,
                target_size_xyz_mm=part_payload["matched_size_xyz_mm"],
            )
            pose = foundationpose_runner(
                cad_path=scaled_cad_path,
                scene_dir=sample_dir,
                debug_dir=debug_dir / part_name,
                mask_path=_resolve(part_payload["mask_path"], repo_root),
                rgb_path=_resolve(region_payload["rgb_path"], repo_root),
                depth_path=_resolve(region_payload["depth_path"], repo_root),
                camera_path=_resolve(region_payload["camera_path"], repo_root),
                coarse_pose_cam_4x4=part_payload.get("coarse_pose_cam_4x4"),
            )
            focused_results[part_name] = {
                "cad_template": str(cad_template),
                "scaled_cad_path": str(scaled_cad_path),
                "matched_size_xyz_mm": [
                    float(value) for value in part_payload["matched_size_xyz_mm"]
                ],
                "coarse_pose_cam_4x4": part_payload.get("coarse_pose_cam_4x4"),
                "pose_cam_4x4": np.asarray(pose, dtype=np.float64).reshape(4, 4).tolist(),
                "pose_source": "foundationpose",
                "status": "aligned",
            }
        except Exception as exc:
            focused_results[part_name] = {
                "status": "failed",
                "error": str(exc),
            }

    result_payload = {
        "schema_version": 1,
        "sample_id": region_payload["sample_id"],
        "workpiece_type": category,
        "focused_parts": focused_results,
        "weld_result": {"status": "not_implemented"},
    }
    validate_alignment_result(result_payload)
    result_path = sample_dir / "alignment_result.json"
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path
