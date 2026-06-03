from __future__ import annotations

import json
import heapq
import importlib.util
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace
from typing import Any
import warnings

import numpy as np

from components.parametric_weld import ParametricWeldContext, ParametricWeldError, get_parametric_strategy
from components.repo_paths import ALIGNMENT_ROOT, prepend_sys_path


class WeldPoseExtractionError(RuntimeError):
    pass


MIN_COMPONENT_VERTICES = 10


def _parse_obj_objects(obj_path: str) -> dict[str, dict]:
    objects: dict[str, dict] = {}
    all_vertices: list[list[float]] = []
    current_name: str | None = None

    with open(obj_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            token = parts[0]
            if token == "o":
                current_name = " ".join(parts[1:])
                objects.setdefault(current_name, {"vertices": [], "faces": []})
            elif token == "v":
                coords = [float(value) for value in parts[1:4]]
                all_vertices.append(coords)
                if current_name is not None:
                    objects[current_name]["vertices"].append(len(all_vertices) - 1)
            elif token == "f" and current_name is not None:
                raw = [int(value.split("/")[0]) - 1 for value in parts[1:]]
                for index in range(1, len(raw) - 1):
                    objects[current_name]["faces"].append((raw[0], raw[index], raw[index + 1]))

    all_vertices_array = np.array(all_vertices, dtype=float) if all_vertices else np.empty((0, 3))
    parsed: dict[str, dict] = {}
    for name, data in objects.items():
        global_indices = data["vertices"]
        if not global_indices:
            continue
        global_to_local = {global_index: local_index for local_index, global_index in enumerate(global_indices)}
        faces = []
        for face in data["faces"]:
            try:
                faces.append([global_to_local[face_index] for face_index in face])
            except KeyError:
                continue
        parsed[name] = {
            "vertices": all_vertices_array[global_indices],
            "faces": np.array(faces, dtype=int) if faces else np.empty((0, 3), dtype=int),
        }
    return parsed


def _load_weld_mesh(obj_path: str, trimesh_module):
    objects = _parse_obj_objects(obj_path)
    if not objects:
        raise ValueError(f"No valid mesh found in {obj_path}")
    best_name = max(objects, key=lambda name: len(objects[name]["vertices"]))
    data = objects[best_name]
    return trimesh_module.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)


def _face_component_indices(faces: np.ndarray) -> list[np.ndarray]:
    vertex_to_faces: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            vertex_to_faces.setdefault(int(vertex_index), []).append(face_index)

    visited = np.zeros(len(faces), dtype=bool)
    components = []
    for start in range(len(faces)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            face_index = stack.pop()
            component.append(face_index)
            for vertex_index in faces[face_index]:
                for neighbor in vertex_to_faces[int(vertex_index)]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
        components.append(np.array(component, dtype=int))
    return components


def _submesh_from_faces(mesh, face_indices: np.ndarray, trimesh_module):
    original_faces = np.asarray(mesh.faces, dtype=int)
    original_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    component_faces = original_faces[face_indices]
    used_vertices = np.unique(component_faces.reshape(-1))
    reindex = np.full(len(original_vertices), -1, dtype=int)
    reindex[used_vertices] = np.arange(len(used_vertices))
    return trimesh_module.Trimesh(
        vertices=original_vertices[used_vertices],
        faces=reindex[component_faces],
        process=False,
    )


def _split_mesh_without_graph(mesh, trimesh_module, only_watertight: bool = False, **_kwargs):
    faces = np.asarray(getattr(mesh, "faces", []), dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        return []
    components = [_submesh_from_faces(mesh, indices, trimesh_module) for indices in _face_component_indices(faces)]
    if only_watertight:
        components = [component for component in components if component.is_watertight]
    return components


def _attach_graphless_split(mesh, trimesh_module):
    def split(bound_mesh, only_watertight: bool = False, **kwargs):
        return _split_mesh_without_graph(
            bound_mesh,
            trimesh_module,
            only_watertight=only_watertight,
            **kwargs,
        )

    mesh.split = MethodType(split, mesh)
    return mesh


def _pca_project(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    origin = vertices.mean(axis=0)
    centered = vertices - origin
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    u, v, normal = vt[0], vt[1], vt[2]
    variance = singular_values**2
    planarity = 1.0 - variance[2] / variance.sum()
    if planarity < 0.95:
        warnings.warn(
            f"Weld seam planarity is {planarity:.1%} (< 95%). "
            "Results may be inaccurate for non-planar weld seams."
        )
    pts_2d = centered @ np.column_stack([u, v])
    return pts_2d, {"origin": origin, "u": u, "v": v, "n": normal, "planarity": planarity}


def _back_project(pts_2d: np.ndarray, plane: dict) -> np.ndarray:
    return plane["origin"] + pts_2d[:, 0:1] * plane["u"] + pts_2d[:, 1:2] * plane["v"]


def _fit_circle_center(pts_2d: np.ndarray) -> tuple[np.ndarray, float]:
    x, y = pts_2d[:, 0], pts_2d[:, 1]
    system = np.column_stack([x, y, np.ones_like(x)])
    target = -(x**2 + y**2)
    result, _, _, _ = np.linalg.lstsq(system, target, rcond=None)
    d_value, e_value, f_value = result
    center_x, center_y = -d_value / 2, -e_value / 2
    radius_squared = center_x**2 + center_y**2 - f_value
    radius = np.sqrt(max(radius_squared, 0.0))
    if radius < 1e-12:
        center = pts_2d.mean(axis=0)
        radius = np.max(np.linalg.norm(pts_2d - center, axis=1))
        return center, radius
    return np.array([center_x, center_y]), radius


def _resample_by_arclength(points: np.ndarray, target_points: int) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    if total_length < 1e-12:
        return points

    target_distances = np.linspace(0, total_length, target_points)
    resampled = np.empty((target_points, 2))
    for index, distance in enumerate(target_distances):
        segment_index = np.searchsorted(cumulative, distance, side="right") - 1
        segment_index = np.clip(segment_index, 0, len(points) - 2)
        segment_length = segment_lengths[segment_index]
        if segment_length < 1e-12:
            resampled[index] = points[segment_index]
        else:
            t_value = (distance - cumulative[segment_index]) / segment_length
            resampled[index] = points[segment_index] * (1 - t_value) + points[segment_index + 1] * t_value
    return resampled


def _extract_centerline_by_angle(pts_2d: np.ndarray, target_points: int = 120) -> np.ndarray:
    center, radius = _fit_circle_center(pts_2d)
    angles = np.arctan2(pts_2d[:, 1] - center[1], pts_2d[:, 0] - center[0])
    sorted_points = pts_2d[np.argsort(angles)]

    window = max(2, len(pts_2d) // target_points)
    groups = len(pts_2d) // window
    if groups < 3:
        raise ValueError(f"Too few groups ({groups}) for centerline extraction")

    centerline = np.array(
        [sorted_points[index * window : (index + 1) * window].mean(axis=0) for index in range(groups)]
    )
    if len(centerline) > 5 and np.isfinite(radius) and radius > 1e-12:
        distances = np.linalg.norm(centerline - center, axis=1)
        keep = np.abs(distances - radius) < radius * 0.3
        if keep.sum() >= 5:
            centerline = centerline[keep]

    for _ in range(3):
        if len(centerline) < 5:
            break
        step_distances = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
        median_step = np.median(step_distances)
        if median_step <= 0:
            break
        max_neighbor_step = np.zeros(len(centerline))
        max_neighbor_step[0] = step_distances[0]
        max_neighbor_step[-1] = step_distances[-1]
        for index in range(1, len(centerline) - 1):
            max_neighbor_step[index] = max(step_distances[index - 1], step_distances[index])
        keep = max_neighbor_step < 5.0 * median_step
        if keep.all():
            break
        centerline = centerline[keep]

    if len(centerline) < 3:
        raise ValueError("Too few centerline points after outlier removal")
    if len(centerline) >= 5:
        smoothed = centerline.copy()
        smoothed[1:-1] = (centerline[:-2] + centerline[1:-1] + centerline[2:]) / 3.0
        centerline = smoothed

    step_sizes = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    if len(step_sizes) > 10:
        p25 = np.percentile(step_sizes, 25)
        p75 = np.percentile(step_sizes, 75)
        if p75 / max(p25, 1e-12) > 3.0:
            centerline = _resample_by_arclength(centerline, max(target_points * 2, len(centerline) * 2))
    return centerline


def _compute_curvature(points: np.ndarray) -> np.ndarray:
    curvature = np.zeros(len(points))
    for index in range(1, len(points) - 1):
        a_point, b_point, c_point = points[index - 1], points[index], points[index + 1]
        ab = np.linalg.norm(b_point - a_point)
        bc = np.linalg.norm(c_point - b_point)
        ac = np.linalg.norm(c_point - a_point)
        cross = abs(
            (b_point[0] - a_point[0]) * (c_point[1] - a_point[1])
            - (b_point[1] - a_point[1]) * (c_point[0] - a_point[0])
        )
        denominator = ab * bc * ac
        curvature[index] = 0.0 if denominator < 1e-12 else 2.0 * cross / denominator
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


def _segment_by_curvature(centerline: np.ndarray) -> list[dict]:
    curvature = _compute_curvature(centerline)
    kernel_size = max(3, min(len(curvature) // 10, 7))
    smoothed = np.convolve(curvature, np.ones(kernel_size) / kernel_size, mode="same")
    nonzero = smoothed[smoothed > 1e-8]
    if len(nonzero) == 0:
        threshold = 1e-6
    else:
        sorted_values = np.sort(nonzero)
        if len(sorted_values) >= 4:
            ratios = sorted_values[1:] / np.maximum(sorted_values[:-1], 1e-12)
            gap_index = np.argmax(ratios)
            threshold = (
                (sorted_values[gap_index] + sorted_values[gap_index + 1]) / 2
                if ratios[gap_index] > 3.0
                else np.median(nonzero) * 0.5
            )
        else:
            threshold = np.median(nonzero) * 0.5

    labels = np.where(smoothed < threshold, 0, 1)
    segments = []
    index = 0
    while index < len(labels):
        label = labels[index]
        end = index
        while end < len(labels) and labels[end] == label:
            end += 1
        segments.append({"label": label, "start": index, "end": end})
        index = end

    min_length = max(3, len(centerline) // 40)
    changed = True
    while changed:
        changed = False
        updated = []
        for segment in segments:
            if segment["end"] - segment["start"] < min_length and updated:
                previous = updated[-1]
                previous["end"] = segment["end"]
                if previous["label"] != segment["label"]:
                    merged = smoothed[previous["start"] : previous["end"]]
                    previous["label"] = 1 if np.median(merged) >= threshold else 0
                changed = True
            else:
                updated.append(segment)
        segments = updated

    merged = [segments[0]]
    for segment in segments[1:]:
        if segment["label"] == merged[-1]["label"]:
            merged[-1]["end"] = segment["end"]
        else:
            merged.append(segment)

    return [
        {
            "type": "arc" if segment["label"] == 1 else "line",
            "indices": (segment["start"], segment["end"]),
            "points_2d": centerline[segment["start"] : segment["end"]],
        }
        for segment in merged
    ]


def _fit_line_error(points: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return float(np.max(np.linalg.norm(points - p0, axis=1)))
    diffs = points - p0
    cross = np.abs(diffs[:, 0] * direction[1] - diffs[:, 1] * direction[0])
    return float(np.max(cross / length))


def _fit_arc_error(points: np.ndarray, p0: np.ndarray, pm: np.ndarray, p1: np.ndarray) -> float:
    ax, ay = p0
    bx, by = pm
    cx, cy = p1
    determinant = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(determinant) < 1e-12:
        return _fit_line_error(points, p0, p1)
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / determinant
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / determinant
    center = np.array([ux, uy])
    radius = np.linalg.norm(p0 - center)
    return float(np.max(np.abs(np.linalg.norm(points - center, axis=1) - radius)))


def _fit_segment(segment: dict) -> dict:
    points = segment["points_2d"]
    if segment["type"] == "line":
        key_points = [points[0], points[-1]]
        error = _fit_line_error(points, points[0], points[-1])
    else:
        middle = len(points) // 2
        key_points = [points[0], points[middle], points[-1]]
        error = _fit_arc_error(points, points[0], points[middle], points[-1])
    return {
        "type": segment["type"],
        "points_2d": key_points,
        "indices": segment["indices"],
        "fitting_error_mm": round(error, 4),
    }


def _detect_closed(centerline: np.ndarray) -> bool:
    diffs = np.diff(centerline, axis=0)
    total_length = np.sum(np.linalg.norm(diffs, axis=1))
    gap = np.linalg.norm(centerline[-1] - centerline[0])
    return bool(gap < total_length * 0.02)


def _make_closing_segment(fitted_segments, centerline):
    last_segment = fitted_segments[-1]
    first_segment = fitted_segments[0]
    return {
        "type": "line",
        "points_2d": [np.array(last_segment["points_2d"][-1]), np.array(first_segment["points_2d"][0])],
        "indices": (len(centerline) - 1, 0),
        "fitting_error_mm": 0.0,
    }


def _process_component(component_mesh, force_close: bool = False) -> dict:
    pts_2d, plane = _pca_project(component_mesh.vertices)
    centerline = _extract_centerline_by_angle(pts_2d)
    fitted = [_fit_segment(segment) for segment in _segment_by_curvature(centerline)]
    is_closed = _detect_closed(centerline)
    if force_close and not is_closed:
        fitted = fitted + [_make_closing_segment(fitted, centerline)]
        is_closed = True
    return {"centerline_2d": centerline, "plane": plane, "fitted": fitted, "closed": is_closed}


def _build_json_output_multi(model_name, paths_data):
    weld_paths = []
    for path in paths_data:
        segments_json = []
        for segment in path["fitted"]:
            points_2d = np.array(segment["points_2d"])
            points_3d = _back_project(points_2d, path["plane"])
            segments_json.append(
                {
                    "type": segment["type"],
                    "points": [[round(c, 6) for c in point] for point in points_3d.tolist()],
                    "fitting_error_mm": segment["fitting_error_mm"],
                }
            )
        weld_paths.append({"closed": bool(path["closed"]), "segments": segments_json})
    return {"model": model_name, "coord_system": "raw", "weld_paths": weld_paths}


def _print_summary(_model_name, _paths_data) -> None:
    return None


def _visualize_multi(*_args, **_kwargs) -> None:
    raise RuntimeError("weld visualizer is unavailable in embedded weld core shim")


def _install_weld_core_shim(trimesh_module) -> None:
    if "weld.core" in sys.modules:
        return
    core = ModuleType("weld.core")
    core.MIN_COMPONENT_VERTICES = MIN_COMPONENT_VERTICES
    core.load_weld_mesh = lambda obj_path: _load_weld_mesh(obj_path, trimesh_module)
    core.pca_project = _pca_project
    core.back_project = _back_project
    core.compute_curvature = _compute_curvature
    core.segment_by_curvature = _segment_by_curvature
    core.fit_line_error = _fit_line_error
    core.fit_arc_error = _fit_arc_error
    core.extract_centerline = lambda mesh, pts_2d: _extract_centerline_by_angle(pts_2d)
    core.detect_closed = _detect_closed
    core.build_json_output_multi = _build_json_output_multi
    core.extract_model_name = lambda workpiece_path: Path(workpiece_path).stem
    core.print_summary = _print_summary
    core.visualize_multi = _visualize_multi
    core._process_component = _process_component
    sys.modules["weld.core"] = core


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class _OptionalImportStubs:
    def __init__(self):
        self._inserted: dict[str, ModuleType] = {}

    def __enter__(self):
        if not _module_available("matplotlib"):
            self._install_matplotlib_stub()
        if not _module_available("networkx"):
            self._install_networkx_stub()
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        for name, module in reversed(self._inserted.items()):
            if sys.modules.get(name) is module:
                del sys.modules[name]

    def _set_module(self, name: str, module: ModuleType) -> None:
        if name not in sys.modules:
            sys.modules[name] = module
            self._inserted[name] = module

    def _install_matplotlib_stub(self) -> None:
        matplotlib = ModuleType("matplotlib")
        matplotlib.use = lambda *_args, **_kwargs: None
        pyplot = ModuleType("matplotlib.pyplot")
        mpl_toolkits = ModuleType("mpl_toolkits")
        mplot3d = ModuleType("mpl_toolkits.mplot3d")
        art3d = ModuleType("mpl_toolkits.mplot3d.art3d")

        class Line3DCollection:
            def __init__(self, *_args, **_kwargs):
                pass

        art3d.Line3DCollection = Line3DCollection
        self._set_module("matplotlib", matplotlib)
        self._set_module("matplotlib.pyplot", pyplot)
        self._set_module("mpl_toolkits", mpl_toolkits)
        self._set_module("mpl_toolkits.mplot3d", mplot3d)
        self._set_module("mpl_toolkits.mplot3d.art3d", art3d)

    def _install_networkx_stub(self) -> None:
        networkx = ModuleType("networkx")

        class Graph:
            def __init__(self):
                self._adjacency: dict[int, dict[int, float]] = {}

            def add_edge(self, source: int, target: int, weight: float = 1.0) -> None:
                self._adjacency.setdefault(source, {})[target] = float(weight)
                self._adjacency.setdefault(target, {})[source] = float(weight)

        def single_source_dijkstra_path_length(graph: Graph, source: int) -> dict[int, float]:
            distances = {source: 0.0}
            queue = [(0.0, source)]
            while queue:
                distance, node = heapq.heappop(queue)
                if distance > distances[node]:
                    continue
                for neighbor, weight in graph._adjacency.get(node, {}).items():
                    candidate = distance + weight
                    if candidate < distances.get(neighbor, float("inf")):
                        distances[neighbor] = candidate
                        heapq.heappush(queue, (candidate, neighbor))
            return distances

        networkx.Graph = Graph
        networkx.single_source_dijkstra_path_length = single_source_dijkstra_path_length
        self._set_module("networkx", networkx)


class WeldPoseExtractor:
    def __init__(self, registry, output_dir: str | Path, enable: bool = True):
        self.registry = registry
        self.output_dir = Path(output_dir)
        self.enable = bool(enable)
        self._runtime_bundle = None

    def _load_runtime_bundle(self):
        if self._runtime_bundle is not None:
            return self._runtime_bundle

        prepend_sys_path(ALIGNMENT_ROOT)
        with _OptionalImportStubs():
            import trimesh
            _install_weld_core_shim(trimesh)
            from weld.core import build_json_output_multi

        self._runtime_bundle = SimpleNamespace(
            trimesh=trimesh,
            build_json_output_multi=build_json_output_multi,
        )
        return self._runtime_bundle

    @staticmethod
    def _derive_output_path(output_dir: Path, source_rgb_path: str | Path, suffix: str) -> Path:
        source_path = Path(source_rgb_path)
        stem = source_path.stem
        if stem.endswith("_color"):
            stem = stem[:-6]
        return output_dir / f"{stem}_{suffix}"

    @staticmethod
    def _as_vertices(mesh: Any) -> np.ndarray:
        vertices = np.asarray(getattr(mesh, "vertices", None), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            raise WeldPoseExtractionError("mesh must contain Nx3 vertices")
        return vertices

    @staticmethod
    def _copy_mesh(mesh: Any):
        if not hasattr(mesh, "copy"):
            raise WeldPoseExtractionError("mesh object must support copy()")
        return mesh.copy()

    @classmethod
    def _scale_weld_mesh(cls, workpiece_mesh: Any, weld_mesh: Any, size_m: np.ndarray):
        workpiece_vertices = cls._as_vertices(workpiece_mesh)
        extents = workpiece_vertices.max(axis=0) - workpiece_vertices.min(axis=0)
        if extents.shape != (3,) or not np.all(np.isfinite(extents)) or np.any(extents <= 1e-8):
            raise WeldPoseExtractionError("invalid workpiece CAD extents")

        size = np.asarray(size_m, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise WeldPoseExtractionError("invalid matched workpiece size")

        scaled = cls._copy_mesh(weld_mesh)
        weld_vertices = cls._as_vertices(scaled)
        scaled.vertices = weld_vertices * (size / extents).reshape(1, 3)
        return scaled

    @classmethod
    def _transform_mesh_to_camera_mm(cls, mesh: Any, refined_pose: np.ndarray):
        pose = np.asarray(refined_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise WeldPoseExtractionError("invalid refined_pose")

        transformed = cls._copy_mesh(mesh)
        vertices = cls._as_vertices(transformed)
        homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
        camera_m = (pose @ homogeneous.T).T[:, :3]
        transformed.vertices = camera_m * 1000.0
        return transformed

    @staticmethod
    def _validated_pose(refined_pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(refined_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise WeldPoseExtractionError("invalid refined_pose")
        return pose

    @classmethod
    def _transform_paths_to_camera_mm(cls, paths_data: list[dict], refined_pose: np.ndarray) -> list[dict]:
        pose = cls._validated_pose(refined_pose)
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        transformed_paths = []

        for path in paths_data:
            plane = path["plane"]
            origin_m = np.asarray(plane["origin"], dtype=np.float64)
            camera_plane = dict(plane)
            camera_plane["origin"] = (rotation @ origin_m + translation) * 1000.0
            camera_plane["u"] = (rotation @ np.asarray(plane["u"], dtype=np.float64)) * 1000.0
            camera_plane["v"] = (rotation @ np.asarray(plane["v"], dtype=np.float64)) * 1000.0
            camera_plane["n"] = rotation @ np.asarray(plane["n"], dtype=np.float64)

            fitted = []
            for segment in path["fitted"]:
                converted = dict(segment)
                converted["fitting_error_mm"] = round(float(segment["fitting_error_mm"]) * 1000.0, 4)
                fitted.append(converted)

            converted_path = dict(path)
            converted_path["plane"] = camera_plane
            converted_path["fitted"] = fitted
            transformed_paths.append(converted_path)

        return transformed_paths

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_registry_entry(self, class_name: str):
        if hasattr(self.registry, "get_entry"):
            return self.registry.get_entry(class_name)
        if hasattr(self.registry, "require_complete_entry"):
            return self.registry.require_complete_entry(class_name)
        if hasattr(self.registry, "normalize_class_name"):
            return SimpleNamespace(class_name=self.registry.normalize_class_name(class_name))
        return SimpleNamespace(class_name=class_name)

    def extract(
        self,
        rgb: np.ndarray,
        class_name: str,
        size_m: np.ndarray,
        refined_pose: np.ndarray,
        intrinsics: dict[str, float],
        visualizer,
        source_rgb_path: str | Path,
        depth: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        object_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if not self.enable:
            return {}

        entry = self._get_registry_entry(class_name)
        model_name = getattr(entry, "class_name", class_name)
        parametric_strategy = get_parametric_strategy(model_name)
        if parametric_strategy is None:
            raise WeldPoseExtractionError(f"unsupported weld extraction category: {model_name}")

        context = ParametricWeldContext(
            class_name=model_name,
            rgb=rgb,
            depth=depth,
            mask=mask,
            object_mask=object_mask,
            size_m=np.asarray(size_m, dtype=np.float64),
            refined_pose=np.asarray(refined_pose, dtype=np.float64),
            intrinsics=dict(intrinsics),
        )
        try:
            paths_data = parametric_strategy.process_scene(context)
        except ParametricWeldError as exc:
            raise WeldPoseExtractionError(str(exc)) from exc

        runtime = self._load_runtime_bundle()
        camera_paths_data = self._transform_paths_to_camera_mm(paths_data, refined_pose)
        weld_json = runtime.build_json_output_multi(model_name, camera_paths_data)
        weld_json["coord_system"] = "camera_mm"
        weld_json["extraction_method"] = "parametric_cad"
        metadata_keys = ("weld_side_selection", "bellmouth_weld_contact_selection")
        for metadata_key in metadata_keys:
            metadata_value = context.metadata.get(metadata_key)
            if metadata_value is not None:
                weld_json[metadata_key] = metadata_value

        json_path = self._derive_output_path(self.output_dir, source_rgb_path, "weld_paths.json")
        self._write_json(json_path, weld_json)

        result = {"weld_json_path": str(json_path)}
        for metadata_key in metadata_keys:
            if metadata_key in weld_json:
                result[metadata_key] = weld_json[metadata_key]
        if visualizer is not None and hasattr(visualizer, "save_weld_pose_overlay"):
            try:
                viz_path = visualizer.save_weld_pose_overlay(
                    rgb=rgb,
                    class_name=model_name,
                    size_m=np.asarray(size_m, dtype=np.float64),
                    refined_pose=np.asarray(refined_pose, dtype=np.float64),
                    intrinsics=intrinsics,
                    source_rgb_path=source_rgb_path,
                    weld_json=weld_json,
                    output_dir=self.output_dir,
                )
                if viz_path is not None:
                    result["weld_visualization_path"] = str(viz_path)
            except Exception as exc:
                result["weld_error"] = f"weld visualization failed: {exc}"
        return result
