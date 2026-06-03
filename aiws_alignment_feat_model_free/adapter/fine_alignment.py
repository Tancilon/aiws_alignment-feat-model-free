from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from aiws_alignment_feat_model_free.adapter.types import WeldAlignmentResult

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = ALIGNMENT_ROOT.parent
for path in (ALIGNMENT_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from aiws_alignment_feat_model_free.aiws_pipeline.alignment_contract import (
    run_alignment_from_region_proposal,
)


DEFAULT_WORKPIECE_INFO_PATH = PROJECT_ROOT / "workpiece_priors/workpiece_info.yaml"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_payload_path(path_value: str | None, *, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _collect_visualization_paths(
    *,
    alignment_payload: dict[str, Any],
    weld_result: dict[str, Any],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for part_payload in alignment_payload.get("focused_parts", {}).values():
        alignment_path = part_payload.get("alignment_visualization_path")
        if alignment_path:
            paths["alignment"] = str(alignment_path)
            break
    weld_path = weld_result.get("weld_visualization_path")
    if weld_path:
        paths["weld"] = str(weld_path)
    return paths


def refine_align_and_extract_weld(
    region_proposal_path: str | Path,
    *,
    visualize: bool = False,
    verbose: bool = False,
    workpiece_info_path: str | Path = DEFAULT_WORKPIECE_INFO_PATH,
    repo_root: str | Path = PROJECT_ROOT,
) -> WeldAlignmentResult:
    region_path = Path(region_proposal_path).resolve()
    try:
        alignment_path = run_alignment_from_region_proposal(
            region_path=region_path,
            workpiece_info_path=workpiece_info_path,
            repo_root=repo_root,
            visualize=visualize,
        )
        alignment_path = Path(alignment_path).resolve()
        alignment_payload = _load_json(alignment_path)
        weld_result = alignment_payload.get("weld_result", {})
        if weld_result.get("status") != "extracted":
            return WeldAlignmentResult(
                status="failed",
                weld_paths=[],
                alignment_result_path=str(alignment_path),
                workpiece_type=alignment_payload.get("workpiece_type"),
                focused_parts=alignment_payload.get("focused_parts"),
                diagnostics={"weld_result": weld_result},
            )

        weld_json_path = _resolve_payload_path(
            weld_result.get("weld_json_path"),
            base_dir=alignment_path.parent,
        )
        if weld_json_path is None:
            return WeldAlignmentResult(
                status="failed",
                weld_paths=[],
                alignment_result_path=str(alignment_path),
                workpiece_type=alignment_payload.get("workpiece_type"),
                focused_parts=alignment_payload.get("focused_parts"),
                diagnostics={"error": "alignment result does not include weld_json_path"},
            )

        weld_payload = _load_json(weld_json_path)
        return WeldAlignmentResult(
            status="ok",
            weld_paths=list(weld_payload.get("weld_paths", [])),
            alignment_result_path=str(alignment_path),
            weld_json_path=str(weld_json_path),
            coord_system=weld_payload.get("coord_system"),
            workpiece_type=alignment_payload.get("workpiece_type"),
            focused_parts=alignment_payload.get("focused_parts"),
            visualization_paths=_collect_visualization_paths(
                alignment_payload=alignment_payload,
                weld_result=weld_result,
            )
            if verbose
            else None,
            diagnostics={"weld_result": weld_result} if verbose else None,
        )
    except Exception as exc:
        return WeldAlignmentResult(
            status="failed",
            weld_paths=[],
            diagnostics={"error": str(exc)},
        )
