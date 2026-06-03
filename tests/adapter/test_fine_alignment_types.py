from __future__ import annotations

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

from adapter.types import WeldAlignmentResult


def test_weld_alignment_result_to_dict_hides_verbose_fields_by_default():
    result = WeldAlignmentResult(
        status="ok",
        weld_paths=[
            {
                "closed": False,
                "segments": [{"type": "line", "points": [[1, 2, 3], [4, 5, 6]]}],
            }
        ],
        alignment_result_path="out/alignment_result.json",
        weld_json_path="out/weld_pose/rgb_weld_paths.json",
        coord_system="camera_mm",
        workpiece_type="bellmouth",
        focused_parts={"tube": {"status": "aligned"}},
        visualization_paths={"weld": "out/weld_pose/rgb_weld_pose.png"},
        diagnostics={"score": 1.0},
    )

    assert result.to_dict() == {
        "status": "ok",
        "weld_paths": [
            {
                "closed": False,
                "segments": [{"type": "line", "points": [[1, 2, 3], [4, 5, 6]]}],
            }
        ],
    }


def test_weld_alignment_result_to_dict_can_include_verbose_fields():
    result = WeldAlignmentResult(
        status="ok",
        weld_paths=[],
        alignment_result_path="out/alignment_result.json",
        weld_json_path="out/weld_pose/rgb_weld_paths.json",
        coord_system="camera_mm",
        workpiece_type="cover_plate",
        focused_parts={"tube": {"status": "aligned"}},
        visualization_paths={"alignment": "out/alignment_visualization/rgb_refined_pose.png"},
        diagnostics={"source": "test"},
    )

    payload = result.to_dict(verbose=True)

    assert payload["alignment_result_path"] == "out/alignment_result.json"
    assert payload["weld_json_path"] == "out/weld_pose/rgb_weld_paths.json"
    assert payload["coord_system"] == "camera_mm"
    assert payload["workpiece_type"] == "cover_plate"
    assert payload["focused_parts"] == {"tube": {"status": "aligned"}}
    assert payload["visualization_paths"] == {
        "alignment": "out/alignment_visualization/rgb_refined_pose.png"
    }
    assert payload["diagnostics"] == {"source": "test"}
