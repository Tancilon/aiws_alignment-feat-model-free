from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WeldAlignmentResult:
    status: str
    weld_paths: list[dict[str, Any]]
    alignment_result_path: str | None = None
    weld_json_path: str | None = None
    coord_system: str | None = None
    workpiece_type: str | None = None
    focused_parts: dict[str, Any] | None = None
    visualization_paths: dict[str, str] | None = None
    diagnostics: dict[str, Any] | None = None

    def to_dict(self, *, verbose: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "weld_paths": list(self.weld_paths),
        }
        if not verbose:
            return payload
        payload.update(
            {
                "alignment_result_path": self.alignment_result_path,
                "weld_json_path": self.weld_json_path,
                "coord_system": self.coord_system,
                "workpiece_type": self.workpiece_type,
                "focused_parts": self.focused_parts,
                "visualization_paths": self.visualization_paths,
                "diagnostics": self.diagnostics,
            }
        )
        return payload
