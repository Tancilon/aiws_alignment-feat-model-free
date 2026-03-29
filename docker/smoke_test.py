import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from foundationpose_aiws import foundationpose_aiws


def main() -> None:
    scene_dir = Path(os.environ.get("FOUNDATIONPOSE_SCENE_DIR", "/workspace/scene"))
    mesh_file = Path(
        os.environ.get("FOUNDATIONPOSE_MESH_FILE", str(scene_dir / "mesh" / "gaiban.obj"))
    )
    debug_dir = Path(os.environ.get("FOUNDATIONPOSE_DEBUG_DIR", "/workspace/debug"))

    pose = foundationpose_aiws(
        mesh_file=mesh_file,
        test_scene_dir=scene_dir,
        debug_dir=debug_dir,
        debug=0,
    )

    payload = {
        "cuda_available": torch.cuda.is_available(),
        "scene_dir": str(scene_dir),
        "mesh_file": str(mesh_file),
        "debug_dir": str(debug_dir),
        "translation": pose[:3, 3].tolist(),
        "rotation_det": float(np.linalg.det(pose[:3, :3])),
        "pose": pose.tolist(),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
