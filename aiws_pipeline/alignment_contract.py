from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from aiws_pipeline.foundationpose_runner import run_foundationpose_part
from aiws_pipeline.mesh_scaling import scale_mesh_to_size_mm
from components.aiws_pipeline_contracts import (
    validate_alignment_result,
    validate_region_proposal,
)
from components.workpiece_priors import WorkpiecePriorRegistry
from components.weld_pose_extractor import WeldPoseExtractor
from utils.depth_compat import load_depth


def _resolve(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _load_intrinsics(camera_path: str | Path, repo_root: Path) -> dict[str, float]:
    payload = json.loads(_resolve(camera_path, repo_root).read_text(encoding="utf-8"))
    intrinsics = payload.get("intrinsics", payload)
    return {
        key: float(intrinsics[key])
        for key in ("fx", "fy", "cx", "cy")
    }


class _StaticWeldRegistry:
    def get_entry(self, class_name: str):
        return SimpleNamespace(class_name=class_name)


def _load_weld_scene_inputs(
    *,
    region_payload: dict[str, Any],
    aligned_part_name: str,
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    depth = None
    mask = None
    object_mask = None
    try:
        rgb = np.asarray(
            Image.open(_resolve(region_payload["rgb_path"], repo_root)).convert("RGB")
        )
    except Exception:
        pass

    try:
        depth = load_depth(_resolve(region_payload["depth_path"], repo_root))
    except Exception:
        depth = None

    try:
        part_payload = region_payload.get("focused_parts", {}).get(aligned_part_name, {})
        mask_path = _resolve(part_payload["mask_path"], repo_root)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    except Exception:
        mask = None

    try:
        object_mask = np.asarray(
            Image.open(_resolve(region_payload["object_mask_path"], repo_root)).convert("L")
        ) > 0
    except Exception:
        object_mask = None

    return rgb, depth, mask, object_mask


def _extract_weld_result(
    *,
    category: str,
    region_payload: dict[str, Any],
    focused_results: dict[str, dict[str, Any]],
    sample_dir: Path,
    source_rgb_path: str | Path,
    intrinsics: dict[str, float],
    repo_root: Path,
) -> dict[str, Any]:
    aligned_part_name = "tube"
    aligned = focused_results.get("tube")
    if not isinstance(aligned, dict) or aligned.get("status") != "aligned":
        aligned_part_name = ""
        aligned = None
        for part_name, part_payload in focused_results.items():
            if isinstance(part_payload, dict) and part_payload.get("status") == "aligned":
                aligned_part_name = part_name
                aligned = part_payload
                break
    if aligned is None:
        return {"status": "failed", "error": "no aligned part available for weld extraction"}

    try:
        rgb, depth, mask, object_mask = _load_weld_scene_inputs(
            region_payload=region_payload,
            aligned_part_name=aligned_part_name,
            repo_root=repo_root,
        )
        extractor = WeldPoseExtractor(
            registry=_StaticWeldRegistry(),
            output_dir=sample_dir / "weld_pose",
            enable=True,
        )
        result = extractor.extract(
            rgb=rgb,
            class_name=category,
            size_m=np.asarray(aligned["matched_size_xyz_mm"], dtype=np.float64) / 1000.0,
            refined_pose=np.asarray(aligned["pose_cam_4x4"], dtype=np.float64),
            intrinsics=intrinsics,
            visualizer=None,
            source_rgb_path=source_rgb_path,
            depth=depth,
            mask=mask,
            object_mask=object_mask,
        )
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}

    return {
        "status": "extracted",
        "weld_json_path": result["weld_json_path"],
        **{
            metadata_key: result[metadata_key]
            for metadata_key in ("weld_side_selection", "bellmouth_weld_contact_selection")
            if metadata_key in result
        },
    }


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
    intrinsics = _load_intrinsics(region_payload["camera_path"], repo_root)

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

    weld_result = _extract_weld_result(
        category=category,
        region_payload=region_payload,
        focused_results=focused_results,
        sample_dir=sample_dir,
        source_rgb_path=region_payload["rgb_path"],
        intrinsics=intrinsics,
        repo_root=repo_root,
    )

    result_payload = {
        "schema_version": 1,
        "sample_id": region_payload["sample_id"],
        "workpiece_type": category,
        "focused_parts": focused_results,
        "weld_result": weld_result,
    }
    validate_alignment_result(result_payload)
    result_path = sample_dir / "alignment_result.json"
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path
