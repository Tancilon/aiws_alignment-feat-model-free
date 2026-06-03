from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import numpy as np

from aiws_alignment_feat_model_free.components.repo_paths import (
    ALIGNMENT_ROOT,
    prepend_sys_path,
)


class _LazyModuleProxy:
    def __init__(self, loader: Callable[[], ModuleType]):
        object.__setattr__(self, "_loader", loader)
        object.__setattr__(self, "_module", None)

    def _resolve(self) -> ModuleType:
        module = object.__getattribute__(self, "_module")
        if module is None:
            module = object.__getattribute__(self, "_loader")()
            object.__setattr__(self, "_module", module)
        return module

    def __getattribute__(self, name: str) -> Any:
        if name in {"_loader", "_module", "_resolve", "__class__", "__dict__", "__setattr__", "__getattribute__", "__delattr__", "__repr__"}:
            return object.__getattribute__(self, name)
        if name in object.__getattribute__(self, "__dict__"):
            return object.__getattribute__(self, "__dict__")[name]
        return getattr(self._resolve(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_loader", "_module"}:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value

    def __delattr__(self, name: str) -> None:
        if name in self.__dict__:
            del self.__dict__[name]
            return
        delattr(self._resolve(), name)

    def __repr__(self) -> str:
        module = object.__getattribute__(self, "_module")
        if module is None:
            return f"<{self.__class__.__name__} unresolved>"
        return repr(module)


class _LazyCallableProxy(_LazyModuleProxy):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)


def _load_cv2() -> ModuleType:
    import cv2 as _cv2

    return _cv2


def _load_trimesh() -> ModuleType:
    import trimesh as _trimesh

    return _trimesh


def _load_draw_posed_3d_box():
    prepend_sys_path(ALIGNMENT_ROOT)
    from Utils import draw_posed_3d_box as _draw_posed_3d_box

    return _draw_posed_3d_box


def _load_draw_xyz_axis():
    prepend_sys_path(ALIGNMENT_ROOT)
    from Utils import draw_xyz_axis as _draw_xyz_axis

    return _draw_xyz_axis


cv2 = _LazyModuleProxy(_load_cv2)
trimesh = _LazyModuleProxy(_load_trimesh)
draw_posed_3d_box = _LazyCallableProxy(_load_draw_posed_3d_box)
draw_xyz_axis = _LazyCallableProxy(_load_draw_xyz_axis)

MASK_OVERLAY_GREEN_RGB = np.asarray([0.0, 255.0, 0.0], dtype=np.float32)
MASK_OVERLAY_ALPHA = 0.4
WELD_LINE_COLOR_BGR = (255, 80, 0)
WELD_ARC_COLOR_BGR = (0, 80, 255)
WELD_PATH_THICKNESS = 3


class RefinedPoseVisualizer:
    def __init__(
        self,
        registry,
        output_dir: str | Path,
        axis_scale: float = 0.1,
        box_thickness: int = 2,
        axis_thickness: int = 3,
        enable: bool = False,
    ):
        self.registry = registry
        self.output_dir = Path(output_dir)
        self.axis_scale = float(axis_scale)
        self.box_thickness = int(box_thickness)
        self.axis_thickness = int(axis_thickness)
        self.enable = bool(enable)
        self._runtime_bundle = None

    def _load_runtime_bundle(self):
        if self._runtime_bundle is not None:
            return self._runtime_bundle

        prepend_sys_path(ALIGNMENT_ROOT)
        self._runtime_bundle = SimpleNamespace(
            cv2=cv2,
            trimesh=trimesh,
            draw_posed_3d_box=draw_posed_3d_box,
            draw_xyz_axis=draw_xyz_axis,
        )
        return self._runtime_bundle

    @staticmethod
    def _intrinsics_to_matrix(intrinsics: dict[str, float]) -> np.ndarray:
        missing = [key for key in ("fx", "fy", "cx", "cy") if key not in intrinsics]
        if missing:
            raise KeyError(f"Missing intrinsics keys: {missing}")
        return np.asarray(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _scale_mesh(mesh, size_m: np.ndarray):
        scaled = mesh.copy()
        vertices = np.asarray(scaled.vertices, dtype=np.float32)
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        safe_extents = np.where(extents > 1e-8, extents, 1.0)
        scale = np.asarray(size_m, dtype=np.float32) / safe_extents
        scaled.vertices = vertices * scale.reshape(1, 3)
        return scaled

    def _load_scaled_mesh(self, class_name: str, size_m: np.ndarray):
        entry = self.registry.require_complete_entry(class_name)
        runtime = self._load_runtime_bundle()
        mesh = runtime.trimesh.load(entry.obj_path)
        return self._scale_mesh(mesh, size_m)

    @staticmethod
    def _derive_output_path(output_dir: Path, source_rgb_path: str | Path, suffix: str) -> Path:
        source_path = Path(source_rgb_path)
        stem = source_path.stem
        if stem.endswith("_color"):
            stem = stem[:-6]
        return output_dir / f"{stem}_{suffix}.png"

    @staticmethod
    def _project_camera_points(
        points_m: np.ndarray,
        intrinsics: dict[str, float],
        image_shape: tuple[int, int],
    ):
        points = np.asarray(points_m, dtype=np.float64)
        z = points[:, 2]
        valid = z > 1e-8
        u = np.zeros(points.shape[0], dtype=np.float64)
        v = np.zeros(points.shape[0], dtype=np.float64)
        u[valid] = float(intrinsics["fx"]) * points[valid, 0] / z[valid] + float(intrinsics["cx"])
        v[valid] = float(intrinsics["fy"]) * points[valid, 1] / z[valid] + float(intrinsics["cy"])
        height, width = image_shape
        finite = np.isfinite(u) & np.isfinite(v)
        in_bounds = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        safe_u = np.where(finite, u, 0.0)
        safe_v = np.where(finite, v, 0.0)
        pixels = np.column_stack([np.rint(safe_u), np.rint(safe_v)]).astype(np.int32)
        pixel_in_bounds = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        valid = valid & finite & in_bounds & pixel_in_bounds
        return pixels, valid

    @staticmethod
    def _valid_pixel_runs(pixels: np.ndarray, valid: np.ndarray):
        valid_array = np.asarray(valid, dtype=bool)
        start = None
        for index, is_valid in enumerate(valid_array):
            if is_valid:
                if start is None:
                    start = index
            elif start is not None:
                if index - start >= 2:
                    yield pixels[start:index]
                start = None

        if start is not None and valid_array.shape[0] - start >= 2:
            yield pixels[start:]

    @staticmethod
    def _interpolate_arc_camera_mm(points_mm: np.ndarray, samples: int = 32) -> np.ndarray:
        p0, pm, p1 = np.asarray(points_mm, dtype=np.float64)
        p0_to_mid = pm - p0
        p0_to_end = p1 - p0
        normal = np.cross(p0_to_mid, p0_to_end)
        mid_distance = np.linalg.norm(p0_to_mid)
        normal_distance = np.linalg.norm(normal)
        if mid_distance < 1e-8 or normal_distance < 1e-8:
            return np.linspace(p0, p1, samples)

        basis_x = p0_to_mid / mid_distance
        basis_z = normal / normal_distance
        basis_y = np.cross(basis_z, basis_x)

        mid_2d = np.array([np.dot(p0_to_mid, basis_x), np.dot(p0_to_mid, basis_y)])
        end_2d = np.array([np.dot(p0_to_end, basis_x), np.dot(p0_to_end, basis_y)])
        system = np.array(
            [
                [2.0 * mid_2d[0], 2.0 * mid_2d[1]],
                [2.0 * end_2d[0], 2.0 * end_2d[1]],
            ],
            dtype=np.float64,
        )
        rhs = np.array(
            [
                np.dot(mid_2d, mid_2d),
                np.dot(end_2d, end_2d),
            ],
            dtype=np.float64,
        )

        try:
            center = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            return np.linspace(p0, p1, samples)

        radius = np.linalg.norm(center)
        if radius < 1e-8:
            return np.linspace(p0, p1, samples)

        theta0 = np.arctan2(-center[1], -center[0])
        thetam = np.arctan2(mid_2d[1] - center[1], mid_2d[0] - center[0])
        theta1 = np.arctan2(end_2d[1] - center[1], end_2d[0] - center[0])
        full_turn = 2.0 * np.pi

        ccw_0_to_mid = (thetam - theta0) % full_turn
        ccw_0_to_end = (theta1 - theta0) % full_turn
        if ccw_0_to_mid <= ccw_0_to_end:
            delta_0_to_mid = ccw_0_to_mid
            delta_mid_to_end = (theta1 - thetam) % full_turn
        else:
            delta_0_to_mid = -((theta0 - thetam) % full_turn)
            delta_mid_to_end = -((thetam - theta1) % full_turn)

        sample_count = max(3, int(samples))
        total_delta = abs(delta_0_to_mid) + abs(delta_mid_to_end)
        first_intervals = int(round((sample_count - 1) * abs(delta_0_to_mid) / total_delta))
        first_intervals = min(max(first_intervals, 1), sample_count - 2)
        second_intervals = sample_count - 1 - first_intervals

        first_angles = theta0 + np.linspace(0.0, delta_0_to_mid, first_intervals + 1)
        second_angles = thetam + np.linspace(0.0, delta_mid_to_end, second_intervals + 1)
        angles = np.concatenate([first_angles, second_angles[1:]])
        xy = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        return p0 + xy[:, 0:1] * basis_x + xy[:, 1:2] * basis_y

    @staticmethod
    def _segment_points_camera_mm(segment: dict[str, Any]) -> np.ndarray | None:
        try:
            points_mm = np.asarray(segment.get("points", []), dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if points_mm.ndim != 2 or points_mm.shape[1] != 3:
            return None
        if not np.all(np.isfinite(points_mm)):
            return None
        return points_mm

    def save(
        self,
        rgb: np.ndarray,
        class_name: str,
        size_m: np.ndarray,
        refined_pose: np.ndarray,
        intrinsics: dict[str, float],
        source_rgb_path: str | Path,
    ) -> Path | None:
        if not self.enable:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        runtime = self._load_runtime_bundle()
        mesh = self._load_scaled_mesh(class_name, size_m)
        bbox_min = np.asarray(mesh.vertices, dtype=np.float32).min(axis=0)
        bbox_max = np.asarray(mesh.vertices, dtype=np.float32).max(axis=0)
        K = self._intrinsics_to_matrix(intrinsics)
        overlay = np.asarray(rgb, dtype=np.uint8).copy()

        bbox = np.stack((bbox_min, bbox_max), axis=0)
        runtime.draw_posed_3d_box(
            K,
            overlay,
            np.asarray(refined_pose, dtype=np.float32),
            bbox,
            line_color=(0, 255, 0),
            linewidth=self.box_thickness,
        )
        runtime.draw_xyz_axis(
            overlay,
            np.asarray(refined_pose, dtype=np.float32),
            scale=self.axis_scale,
            K=K,
            thickness=self.axis_thickness,
            transparency=0,
            is_input_rgb=True,
        )

        output_path = self._derive_output_path(self.output_dir, source_rgb_path, "refined_pose")
        bgr = overlay[..., ::-1]
        save_ok = runtime.cv2.imwrite(str(output_path), bgr)
        if save_ok is False:
            raise RuntimeError(f"Failed to write refined pose overlay: {output_path}")
        return output_path

    def save_weld_pose_overlay(
        self,
        rgb: np.ndarray,
        class_name: str,
        size_m: np.ndarray,
        refined_pose: np.ndarray,
        intrinsics: dict[str, float],
        source_rgb_path: str | Path,
        weld_json: dict[str, Any],
        output_dir: str | Path | None = None,
    ) -> Path | None:
        if not self.enable:
            return None

        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        runtime = self._load_runtime_bundle()
        mesh = self._load_scaled_mesh(class_name, size_m)
        bbox_min = np.asarray(mesh.vertices, dtype=np.float32).min(axis=0)
        bbox_max = np.asarray(mesh.vertices, dtype=np.float32).max(axis=0)
        K = self._intrinsics_to_matrix(intrinsics)
        overlay = np.asarray(rgb, dtype=np.uint8).copy()

        runtime.draw_posed_3d_box(
            K,
            overlay,
            np.asarray(refined_pose, dtype=np.float32),
            np.stack((bbox_min, bbox_max), axis=0),
            line_color=(0, 255, 0),
            linewidth=self.box_thickness,
        )
        runtime.draw_xyz_axis(
            overlay,
            np.asarray(refined_pose, dtype=np.float32),
            scale=self.axis_scale,
            K=K,
            thickness=self.axis_thickness,
            transparency=0,
            is_input_rgb=True,
        )

        bgr = overlay[..., ::-1].copy()
        if weld_json.get("coord_system") != "camera_mm":
            raise ValueError("weld_json coord_system must be camera_mm")

        weld_paths = weld_json.get("weld_paths", [])
        if not isinstance(weld_paths, (list, tuple)):
            weld_paths = []
        for path in weld_paths:
            if not isinstance(path, dict):
                continue
            segments = path.get("segments", [])
            if not isinstance(segments, (list, tuple)):
                segments = []
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                points_mm = self._segment_points_camera_mm(segment)
                if points_mm is None:
                    continue
                if segment.get("type") == "arc" and points_mm.shape[0] == 3:
                    draw_points_mm = self._interpolate_arc_camera_mm(points_mm)
                    color = WELD_ARC_COLOR_BGR
                elif segment.get("type") == "line" and points_mm.shape[0] >= 2:
                    draw_points_mm = points_mm
                    color = WELD_LINE_COLOR_BGR
                else:
                    continue

                pixels, valid = self._project_camera_points(
                    draw_points_mm / 1000.0,
                    intrinsics,
                    image_shape=bgr.shape[:2],
                )
                if np.count_nonzero(valid) < 2:
                    continue
                for pixel_run in self._valid_pixel_runs(pixels, valid):
                    runtime.cv2.polylines(
                        bgr,
                        [pixel_run.reshape(-1, 1, 2)],
                        False,
                        color,
                        WELD_PATH_THICKNESS,
                    )

        output_path = self._derive_output_path(target_dir, source_rgb_path, "weld_pose")
        save_ok = runtime.cv2.imwrite(str(output_path), bgr)
        if save_ok is False:
            raise RuntimeError(f"Failed to write weld pose overlay: {output_path}")
        return output_path

    def save_mask_overlay(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        source_rgb_path: str | Path,
    ) -> Path | None:
        if not self.enable:
            return None

        rgb_array = np.asarray(rgb)
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            raise ValueError("rgb must have shape HxWx3")

        mask_array = np.asarray(mask)
        if mask_array.ndim != 2 or mask_array.shape != rgb_array.shape[:2]:
            raise ValueError("mask must match rgb spatial dimensions")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        runtime = self._load_runtime_bundle()

        overlay = np.asarray(rgb_array, dtype=np.float32).copy()
        mask_bool = mask_array.astype(bool)
        overlay[mask_bool] = (
            overlay[mask_bool] * (1.0 - MASK_OVERLAY_ALPHA)
            + MASK_OVERLAY_GREEN_RGB * MASK_OVERLAY_ALPHA
        )
        overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

        output_path = self._derive_output_path(self.output_dir, source_rgb_path, "mask_overlay")
        bgr = overlay[..., ::-1]
        save_ok = runtime.cv2.imwrite(str(output_path), bgr)
        if save_ok is False:
            raise RuntimeError(f"Failed to write mask overlay: {output_path}")
        return output_path
