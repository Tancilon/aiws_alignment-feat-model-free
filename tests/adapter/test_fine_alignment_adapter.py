from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))
for name in [
    module_name
    for module_name in sys.modules
    if module_name == "adapter"
    or module_name.startswith("adapter.")
    or module_name == "components"
    or module_name.startswith("components.")
    or module_name == "aiws_pipeline"
    or module_name.startswith("aiws_pipeline.")
]:
    sys.modules.pop(name, None)

import adapter.fine_alignment as fine_alignment
from adapter.fine_alignment import refine_align_and_extract_weld


def _write_successful_alignment_payload(tmp_path: Path) -> Path:
    weld_json_path = tmp_path / "weld_pose/rgb_weld_paths.json"
    weld_json_path.parent.mkdir(parents=True, exist_ok=True)
    weld_json_path.write_text(
        json.dumps({"coord_system": "camera_mm", "weld_paths": []}),
        encoding="utf-8",
    )
    alignment_path = tmp_path / "alignment_result.json"
    alignment_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_id": "sample",
                "workpiece_type": "square_tube",
                "focused_parts": {"tube": {"status": "aligned"}},
                "weld_result": {
                    "status": "extracted",
                    "weld_json_path": str(weld_json_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return alignment_path


def test_refine_align_and_extract_weld_returns_minimal_weld_paths(tmp_path, monkeypatch):
    region_path = tmp_path / "region_proposal.json"
    region_path.write_text("{}", encoding="utf-8")

    def fake_alignment(**kwargs):
        assert kwargs["region_path"] == region_path.resolve()
        assert kwargs["visualize"] is False
        weld_json_path = tmp_path / "weld_pose/rgb_weld_paths.json"
        weld_json_path.parent.mkdir(parents=True, exist_ok=True)
        weld_json_path.write_text(
            json.dumps(
                {
                    "coord_system": "camera_mm",
                    "weld_paths": [
                        {
                            "closed": False,
                            "segments": [
                                {
                                    "type": "line",
                                    "points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        alignment_path = tmp_path / "alignment_result.json"
        alignment_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sample_id": "sample",
                    "workpiece_type": "bellmouth",
                    "focused_parts": {"tube": {"status": "aligned"}},
                    "weld_result": {
                        "status": "extracted",
                        "weld_json_path": str(weld_json_path),
                    },
                }
            ),
            encoding="utf-8",
        )
        return alignment_path

    monkeypatch.setattr(fine_alignment, "run_alignment_from_region_proposal", fake_alignment)

    result = refine_align_and_extract_weld(str(region_path))

    assert result.to_dict() == {
        "status": "ok",
        "weld_paths": [
            {
                "closed": False,
                "segments": [
                    {"type": "line", "points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
                ],
            }
        ],
    }


def test_refine_align_and_extract_weld_suppresses_runtime_noise_by_default(
    tmp_path, monkeypatch, capsys
):
    logging.getLogger().setLevel(logging.INFO)
    region_path = tmp_path / "region_proposal.json"
    region_path.write_text("{}", encoding="utf-8")

    def noisy_alignment(**_kwargs):
        print("runtime stdout noise")
        print("runtime stderr noise", file=sys.stderr)
        logging.info("runtime logging noise")
        return _write_successful_alignment_payload(tmp_path)

    monkeypatch.setattr(
        fine_alignment,
        "run_alignment_from_region_proposal",
        noisy_alignment,
    )

    result = refine_align_and_extract_weld(str(region_path), verbose=False)

    captured = capsys.readouterr()
    assert result.status == "ok"
    assert "runtime stdout noise" not in captured.out
    assert "runtime stderr noise" not in captured.err
    assert "runtime logging noise" not in captured.err


def test_refine_align_and_extract_weld_keeps_runtime_noise_when_verbose(
    tmp_path, monkeypatch, capsys
):
    region_path = tmp_path / "region_proposal.json"
    region_path.write_text("{}", encoding="utf-8")

    def noisy_alignment(**_kwargs):
        print("runtime stdout noise")
        print("runtime stderr noise", file=sys.stderr)
        return _write_successful_alignment_payload(tmp_path)

    monkeypatch.setattr(
        fine_alignment,
        "run_alignment_from_region_proposal",
        noisy_alignment,
    )

    result = refine_align_and_extract_weld(str(region_path), verbose=True)

    captured = capsys.readouterr()
    assert result.status == "ok"
    assert "runtime stdout noise" in captured.out
    assert "runtime stderr noise" in captured.err


def test_refine_align_and_extract_weld_verbose_includes_paths_and_visualizations(
    tmp_path, monkeypatch
):
    region_path = tmp_path / "region_proposal.json"
    region_path.write_text("{}", encoding="utf-8")

    def fake_alignment(**kwargs):
        assert kwargs["visualize"] is True
        weld_json_path = tmp_path / "weld_pose/rgb_weld_paths.json"
        weld_json_path.parent.mkdir(parents=True, exist_ok=True)
        weld_json_path.write_text(
            json.dumps({"coord_system": "camera_mm", "weld_paths": []}),
            encoding="utf-8",
        )
        alignment_path = tmp_path / "alignment_result.json"
        alignment_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sample_id": "sample",
                    "workpiece_type": "cover_plate",
                    "focused_parts": {
                        "tube": {
                            "status": "aligned",
                            "alignment_visualization_path": str(
                                tmp_path / "alignment_visualization/rgb_refined_pose.png"
                            ),
                        }
                    },
                    "weld_result": {
                        "status": "extracted",
                        "weld_json_path": str(weld_json_path),
                        "weld_visualization_path": str(
                            tmp_path / "weld_pose/rgb_weld_pose.png"
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
        return alignment_path

    monkeypatch.setattr(fine_alignment, "run_alignment_from_region_proposal", fake_alignment)

    result = refine_align_and_extract_weld(str(region_path), visualize=True, verbose=True)
    payload = result.to_dict(verbose=True)

    assert payload["status"] == "ok"
    assert payload["coord_system"] == "camera_mm"
    assert payload["workpiece_type"] == "cover_plate"
    assert payload["alignment_result_path"].endswith("alignment_result.json")
    assert payload["weld_json_path"].endswith("rgb_weld_paths.json")
    assert payload["visualization_paths"]["alignment"].endswith("rgb_refined_pose.png")
    assert payload["visualization_paths"]["weld"].endswith("rgb_weld_pose.png")
