from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from utils.depth_compat import load_depth


def _camera_matrix(camera_path: str | Path) -> np.ndarray:
    payload = json.loads(Path(camera_path).read_text(encoding="utf-8"))
    intrinsics = payload.get("intrinsics", payload)
    return np.asarray(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _load_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"RGB image not found: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mask(path: str | Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"))
    return mask > 0


def _load_foundationpose_runtime() -> Any:
    import nvdiffrast.torch as dr
    import trimesh
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor

    return type(
        "FoundationPoseRuntime",
        (),
        {
            "FoundationPose": FoundationPose,
            "ScorePredictor": ScorePredictor,
            "PoseRefinePredictor": PoseRefinePredictor,
            "RasterizeCudaContext": dr.RasterizeCudaContext,
            "load_mesh": trimesh.load,
        },
    )


def run_foundationpose_part(
    cad_path: str | Path,
    scene_dir: str | Path,
    debug_dir: str | Path,
    mask_path: str | Path,
    rgb_path: str | Path,
    depth_path: str | Path,
    camera_path: str | Path,
    coarse_pose_cam_4x4: Any | None = None,
    iteration: int = 5,
) -> np.ndarray:
    _ = scene_dir
    _ = coarse_pose_cam_4x4
    runtime = _load_foundationpose_runtime()
    mesh = runtime.load_mesh(Path(cad_path))
    estimator = runtime.FoundationPose(
        model_pts=np.asarray(mesh.vertices),
        model_normals=np.asarray(mesh.vertex_normals),
        mesh=mesh,
        scorer=runtime.ScorePredictor(),
        refiner=runtime.PoseRefinePredictor(),
        glctx=runtime.RasterizeCudaContext(),
        debug=0,
        debug_dir=str(debug_dir),
    )
    pose = estimator.register(
        K=_camera_matrix(camera_path),
        rgb=_load_rgb(rgb_path),
        depth=load_depth(depth_path),
        ob_mask=np.asarray(_load_mask(mask_path), dtype=np.bool_),
        iteration=iteration,
    )
    if pose is None:
        raise RuntimeError("FoundationPose returned no pose")
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)
