from __future__ import annotations

from adapter.types import WeldAlignmentResult


def refine_align_and_extract_weld(*_args, **_kwargs) -> WeldAlignmentResult:
    return WeldAlignmentResult(
        status="failed",
        weld_paths=[],
        diagnostics={"error": "refine_align_and_extract_weld is not implemented"},
    )
