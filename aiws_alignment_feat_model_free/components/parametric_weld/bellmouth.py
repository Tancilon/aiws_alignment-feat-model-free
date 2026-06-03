from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aiws_alignment_feat_model_free.components.parametric_weld.context import (
    ParametricWeldContext,
)
from aiws_alignment_feat_model_free.components.parametric_weld.geometry import (
    line_segment,
    make_path,
    validate_size_m,
)
from aiws_alignment_feat_model_free.components.repo_paths import PROJECT_ROOT


DEFAULT_CONTACT_FACE = "-Z"
FALLBACK_SPAN_RATIO = 0.55
MAX_SCENE_SUPPORT_POINTS = 250_000
PLATE_NEAR_MARGIN_M = 0.12
PLANE_DISTANCE_TOL_M = 0.008
MIN_PLATE_INLIERS = 20
MIN_PLANE_INLIER_RATIO = 0.35
AMBIGUOUS_FACE_MARGIN = 0.08
MIN_PLANE_TANGENT_RATIO = 0.08
DEFAULT_TUBE_CAD_PATH = PROJECT_ROOT / "workpiece_priors/component_assembly/tube.obj"
CAD_FACE_BAND_RATIO = 0.005
CAD_PROFILE_PERCENTILE = 95.0
CAD_PROFILE_MIN_RATIO = 0.05
CAD_PROFILE_MAX_RATIO = 0.995


@dataclass(frozen=True)
class _FaceCandidate:
    name: str
    plane_axis: int
    plane_sign: float
    edge_axis: int


@dataclass(frozen=True)
class _PlatePlaneFit:
    status: str
    origin: np.ndarray | None
    normal: np.ndarray | None
    inliers: np.ndarray
    inlier_ratio: float
    residual_median: float | None
    residual_p95: float | None
    reason: str | None = None

    def as_metadata(self) -> dict:
        payload = {
            "status": self.status,
            "inlier_count": int(len(self.inliers)),
            "inlier_ratio": round(float(self.inlier_ratio), 6),
        }
        if self.origin is not None:
            payload["origin"] = [float(value) for value in self.origin.tolist()]
        if self.normal is not None:
            payload["normal"] = [float(value) for value in self.normal.tolist()]
        if self.residual_median is not None:
            payload["residual_median_m"] = round(float(self.residual_median), 6)
        if self.residual_p95 is not None:
            payload["residual_p95_m"] = round(float(self.residual_p95), 6)
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class _CadFlatEdgeProfile:
    method: str
    flat_half_x_m: float
    flat_half_z_m: float
    shrink_x_m: float
    shrink_z_m: float
    fallback_reason: str | None = None

    @property
    def uses_bbox(self) -> bool:
        return self.method == "bbox_fallback"

    def as_metadata(self) -> dict:
        payload = {
            "method": self.method,
            "flat_half_x_m": float(self.flat_half_x_m),
            "flat_half_z_m": float(self.flat_half_z_m),
            "shrink_x_m": float(self.shrink_x_m),
            "shrink_z_m": float(self.shrink_z_m),
        }
        if self.fallback_reason is not None:
            payload["fallback_reason"] = self.fallback_reason
        return payload


class BellmouthParametricWeldStrategy:
    def process_scene(self, context: ParametricWeldContext) -> list[dict]:
        size = validate_size_m(context.size_m)
        cad_edge_profile = _cad_flat_edge_profile(size)
        selection = self._select_contact(context, size)
        selection["metadata"]["cad_edge_profile"] = cad_edge_profile.as_metadata()
        context.metadata["bellmouth_weld_contact_selection"] = selection["metadata"]
        return self._paths_for_face(
            size,
            selection["selected_face"],
            selection["span_y"],
            cad_edge_profile,
        )

    @staticmethod
    def _select_contact(context: ParametricWeldContext, size: np.ndarray) -> dict:
        candidate_points, candidate_error = _plate_candidate_points(context, size)
        fit = _fit_plate_plane(candidate_points)
        if fit.status != "fitted":
            reason = (
                candidate_error
                if len(candidate_points) == 0 and candidate_error is not None
                else fit.reason or candidate_error or "plate_plane_fit_failed"
            )
            ambiguous = reason in {
                "ambiguous_plate_plane_contact_face",
                "degenerate_plate_plane",
            }
            return _fallback_selection(
                size=size,
                reason="ambiguous_plate_plane_contact_face" if ambiguous else reason,
                candidate_scores=[],
                ambiguous=ambiguous,
                plate_plane_fit=fit.as_metadata(),
            )

        scores = [_score_face_from_plane(candidate, fit, size) for candidate in _face_candidates()]
        scores = sorted(scores, key=lambda item: item["score"], reverse=True)
        best = scores[0]
        second = scores[1] if len(scores) > 1 else {"score": 0.0}
        margin = float(best["score"]) - float(second["score"])
        ambiguous = margin < AMBIGUOUS_FACE_MARGIN
        span_y, span_source = _span_from_plate_inliers(fit, size)

        if ambiguous:
            return _fallback_selection(
                size=size,
                reason="ambiguous_plate_plane_contact_face",
                candidate_scores=scores,
                ambiguous=True,
                plate_plane_fit=fit.as_metadata(),
            )
        if span_y is None:
            return _fallback_selection(
                size=size,
                reason=span_source,
                candidate_scores=scores,
                ambiguous=False,
                plate_plane_fit=fit.as_metadata(),
            )

        metadata = {
            "selected_face": best["face"],
            "method": "plate_plane_fit_v1",
            "ambiguous": False,
            "score_margin": round(float(margin), 6),
            "candidate_scores": scores,
            "span_y_m": [float(span_y[0]), float(span_y[1])],
            "plate_plane_fit": fit.as_metadata(),
            "span_source": span_source,
        }
        return {"selected_face": best["face"], "span_y": span_y, "metadata": metadata}

    @staticmethod
    def _paths_for_face(
        size: np.ndarray,
        face_name: str,
        span_y: tuple[float, float],
        cad_edge_profile: _CadFlatEdgeProfile | None = None,
    ) -> list[dict]:
        x_half = 0.5 * size[0]
        z_half = 0.5 * size[2]
        flat_half_x = _profile_flat_half_x(cad_edge_profile, x_half)
        flat_half_z = _profile_flat_half_z(cad_edge_profile, z_half)
        if face_name in {"-Z", "+Z"}:
            z = -z_half if face_name == "-Z" else z_half
            return [
                _line_path(
                    origin=np.array([-flat_half_x, 0.0, z], dtype=np.float64),
                    u=np.array([0.0, 1.0, 0.0]),
                    v=np.array([0.0, 0.0, 1.0]),
                    n=np.array([-1.0, 0.0, 0.0]),
                    span_y=span_y,
                    flip=False,
                ),
                _line_path(
                    origin=np.array([flat_half_x, 0.0, z], dtype=np.float64),
                    u=np.array([0.0, 1.0, 0.0]),
                    v=np.array([0.0, 0.0, 1.0]),
                    n=np.array([1.0, 0.0, 0.0]),
                    span_y=span_y,
                    flip=True,
                ),
            ]

        x = -x_half if face_name == "-X" else x_half
        return [
            _line_path(
                origin=np.array([x, 0.0, -flat_half_z], dtype=np.float64),
                u=np.array([0.0, 1.0, 0.0]),
                v=np.array([1.0, 0.0, 0.0]),
                n=np.array([0.0, 0.0, -1.0]),
                span_y=span_y,
                flip=False,
            ),
            _line_path(
                origin=np.array([x, 0.0, flat_half_z], dtype=np.float64),
                u=np.array([0.0, 1.0, 0.0]),
                v=np.array([1.0, 0.0, 0.0]),
                n=np.array([0.0, 0.0, 1.0]),
                span_y=span_y,
                flip=True,
            ),
        ]


def _profile_flat_half_x(profile: _CadFlatEdgeProfile | None, fallback: float) -> float:
    if profile is None or profile.uses_bbox:
        return float(fallback)
    value = float(profile.flat_half_x_m)
    if not np.isfinite(value) or value <= 0.0:
        return float(fallback)
    return min(value, float(fallback))


def _profile_flat_half_z(profile: _CadFlatEdgeProfile | None, fallback: float) -> float:
    if profile is None or profile.uses_bbox:
        return float(fallback)
    value = float(profile.flat_half_z_m)
    if not np.isfinite(value) or value <= 0.0:
        return float(fallback)
    return min(value, float(fallback))


def _line_path(
    *,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n: np.ndarray,
    span_y: tuple[float, float],
    flip: bool,
) -> dict:
    p0 = np.array([span_y[0], 0.0], dtype=np.float64)
    p1 = np.array([span_y[1], 0.0], dtype=np.float64)
    if flip:
        p0, p1 = p1, p0
    return make_path(
        origin=origin,
        u=u,
        v=v,
        n=n,
        segments=[line_segment(p0, p1, 0)],
        closed=False,
    )


def _face_candidates() -> list[_FaceCandidate]:
    return [
        _FaceCandidate("-Z", plane_axis=2, plane_sign=-1.0, edge_axis=0),
        _FaceCandidate("+Z", plane_axis=2, plane_sign=1.0, edge_axis=0),
        _FaceCandidate("-X", plane_axis=0, plane_sign=-1.0, edge_axis=2),
        _FaceCandidate("+X", plane_axis=0, plane_sign=1.0, edge_axis=2),
    ]


def _support_points(context: ParametricWeldContext) -> tuple[np.ndarray | None, str | None]:
    if context.object_points_m is not None:
        points = np.asarray(context.object_points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            return None, "invalid_object_points"
        points = points[np.all(np.isfinite(points), axis=1)]
        return points, None

    if context.depth is None or context.mask is None:
        return None, "missing_rgbd_support"
    try:
        depth = np.asarray(context.depth, dtype=np.float64)
        mask = np.asarray(context.mask)
        if depth.ndim != 2 or mask.shape != depth.shape:
            return None, "invalid_rgbd_support"
        if context.object_mask is not None:
            object_mask = np.asarray(context.object_mask)
            if object_mask.shape != depth.shape:
                return None, "invalid_object_mask_support"
            support_region = (object_mask > 0) & (~(mask > 0))
        else:
            support_region = ~(mask > 0)
        valid = support_region & np.isfinite(depth) & (depth > 0.0)
        ys, xs = np.where(valid)
        if len(xs) == 0:
            return None, "empty_rgbd_support"
        if len(xs) > MAX_SCENE_SUPPORT_POINTS:
            stride = int(np.ceil(len(xs) / MAX_SCENE_SUPPORT_POINTS))
            ys = ys[::stride]
            xs = xs[::stride]

        fx = _intrinsic(context.intrinsics, "fx")
        fy = _intrinsic(context.intrinsics, "fy")
        cx = _intrinsic(context.intrinsics, "cx")
        cy = _intrinsic(context.intrinsics, "cy")
        z = depth[ys, xs]
        camera_points = np.column_stack(
            [
                (xs.astype(np.float64) - cx) * z / fx,
                (ys.astype(np.float64) - cy) * z / fy,
                z,
            ]
        )
        pose = np.asarray(context.refined_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            return None, "invalid_refined_pose"
        points = (camera_points - pose[:3, 3].reshape(1, 3)) @ pose[:3, :3]
        points = points[np.all(np.isfinite(points), axis=1)]
        return points, None
    except Exception as exc:
        return None, str(exc)


def _plate_candidate_points(
    context: ParametricWeldContext, size: np.ndarray
) -> tuple[np.ndarray, str | None]:
    points, error = _support_points(context)
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float64), error or "missing_plate_candidates"

    half = 0.5 * np.asarray(size, dtype=np.float64)
    margin = max(PLATE_NEAR_MARGIN_M, float(max(size[0], size[2])) * 0.75)
    near_tube = (
        (points[:, 0] >= -half[0] - margin)
        & (points[:, 0] <= half[0] + margin)
        & (points[:, 1] >= -half[1] - margin)
        & (points[:, 1] <= half[1] + margin)
        & (points[:, 2] >= -half[2] - margin)
        & (points[:, 2] <= half[2] + margin)
    )
    filtered = points[near_tube]
    if len(filtered) == 0:
        return filtered, "empty_near_tube_plate_candidates"
    return filtered, None


def _fit_plate_plane(points: np.ndarray) -> _PlatePlaneFit:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        points = np.empty((0, 3), dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < MIN_PLATE_INLIERS:
        return _failed_plane_fit("insufficient_plate_candidates")

    candidates = []
    axis_planes = [
        (0, float(np.percentile(points[:, 0], 5.0))),
        (0, float(np.percentile(points[:, 0], 95.0))),
        (2, float(np.percentile(points[:, 2], 5.0))),
        (2, float(np.percentile(points[:, 2], 95.0))),
    ]
    for axis, value in axis_planes:
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
        origin = np.zeros(3, dtype=np.float64)
        origin[axis] = value
        residuals = np.abs((points - origin.reshape(1, 3)) @ normal)
        inliers = points[residuals <= PLANE_DISTANCE_TOL_M]
        candidates.append((len(inliers), residuals, inliers, origin, normal))

    best_count, best_residuals, best_inliers, best_origin, best_normal = max(
        candidates, key=lambda item: item[0]
    )
    inlier_ratio = float(best_count) / float(len(points))
    if best_count < MIN_PLATE_INLIERS or inlier_ratio < MIN_PLANE_INLIER_RATIO:
        return _PlatePlaneFit(
            status="failed",
            origin=best_origin,
            normal=best_normal,
            inliers=best_inliers,
            inlier_ratio=inlier_ratio,
            residual_median=float(np.median(best_residuals)),
            residual_p95=float(np.percentile(best_residuals, 95.0)),
            reason="weak_plate_plane",
        )

    refined_origin = np.mean(best_inliers, axis=0)
    centered = best_inliers - refined_origin.reshape(1, 3)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[0] <= 0.0:
        return _PlatePlaneFit(
            status="failed",
            origin=refined_origin,
            normal=best_normal,
            inliers=best_inliers,
            inlier_ratio=inlier_ratio,
            residual_median=float(np.median(best_residuals)),
            residual_p95=float(np.percentile(best_residuals, 95.0)),
            reason="degenerate_plate_plane",
        )
    if singular_values[1] / singular_values[0] < MIN_PLANE_TANGENT_RATIO:
        return _PlatePlaneFit(
            status="failed",
            origin=refined_origin,
            normal=best_normal,
            inliers=best_inliers,
            inlier_ratio=inlier_ratio,
            residual_median=float(np.median(best_residuals)),
            residual_p95=float(np.percentile(best_residuals, 95.0)),
            reason="degenerate_plate_plane",
        )

    refined_normal = vh[-1]
    if np.dot(refined_normal, best_normal) < 0.0:
        refined_normal = -refined_normal
    normal_norm = np.linalg.norm(refined_normal)
    if normal_norm <= 0.0:
        return _PlatePlaneFit(
            status="failed",
            origin=refined_origin,
            normal=best_normal,
            inliers=best_inliers,
            inlier_ratio=inlier_ratio,
            residual_median=float(np.median(best_residuals)),
            residual_p95=float(np.percentile(best_residuals, 95.0)),
            reason="degenerate_plate_plane",
        )
    refined_normal = refined_normal / normal_norm
    refined_residuals = np.abs(centered @ refined_normal)
    return _PlatePlaneFit(
        status="fitted",
        origin=refined_origin,
        normal=refined_normal,
        inliers=best_inliers,
        inlier_ratio=inlier_ratio,
        residual_median=float(np.median(refined_residuals)),
        residual_p95=float(np.percentile(refined_residuals, 95.0)),
    )


def _failed_plane_fit(reason: str) -> _PlatePlaneFit:
    return _PlatePlaneFit(
        status="failed",
        origin=None,
        normal=None,
        inliers=np.empty((0, 3), dtype=np.float64),
        inlier_ratio=0.0,
        residual_median=None,
        residual_p95=None,
        reason=reason,
    )


def _bbox_flat_edge_profile(size: np.ndarray, reason: str) -> _CadFlatEdgeProfile:
    x_half = 0.5 * float(size[0])
    z_half = 0.5 * float(size[2])
    return _CadFlatEdgeProfile(
        method="bbox_fallback",
        flat_half_x_m=x_half,
        flat_half_z_m=z_half,
        shrink_x_m=0.0,
        shrink_z_m=0.0,
        fallback_reason=reason,
    )


def _load_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    arr = np.asarray(vertices, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 16:
        raise ValueError("cad_profile_invalid_geometry")
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if len(arr) < 16:
        raise ValueError("cad_profile_invalid_geometry")
    return arr


def _axis_half_extent(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("cad_profile_invalid_geometry")
    half = max(abs(float(np.min(finite))), abs(float(np.max(finite))))
    if half <= 0.0 or not np.isfinite(half):
        raise ValueError("cad_profile_invalid_geometry")
    return half


def _flat_half_ratio_for_face(
    vertices: np.ndarray,
    *,
    face_axis: int,
    tangent_axis: int,
    face_sign: float,
    face_half: float,
    tangent_half: float,
) -> float:
    signed_face = face_sign * vertices[:, face_axis]
    face_max = float(np.max(signed_face))
    band = max(face_half * CAD_FACE_BAND_RATIO, 1e-9)
    face_points = vertices[signed_face >= face_max - band]
    if len(face_points) < 4:
        raise ValueError("cad_profile_insufficient_face_vertices")
    tangent_abs = np.abs(face_points[:, tangent_axis])
    flat_half = float(np.percentile(tangent_abs, CAD_PROFILE_PERCENTILE))
    ratio = flat_half / tangent_half
    if not np.isfinite(ratio) or ratio <= CAD_PROFILE_MIN_RATIO or ratio >= CAD_PROFILE_MAX_RATIO:
        raise ValueError("cad_profile_invalid_flat_extent")
    return ratio


def _derive_cad_flat_edge_profile_from_vertices(
    vertices: np.ndarray, size: np.ndarray
) -> _CadFlatEdgeProfile:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("cad_profile_invalid_geometry")
    vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
    if len(vertices) < 16:
        raise ValueError("cad_profile_invalid_geometry")

    size = validate_size_m(size)
    x_half_m = 0.5 * float(size[0])
    z_half_m = 0.5 * float(size[2])
    template_x_half = _axis_half_extent(vertices[:, 0])
    template_z_half = _axis_half_extent(vertices[:, 2])

    flat_x_ratios = [
        _flat_half_ratio_for_face(
            vertices,
            face_axis=2,
            tangent_axis=0,
            face_sign=1.0,
            face_half=template_z_half,
            tangent_half=template_x_half,
        ),
        _flat_half_ratio_for_face(
            vertices,
            face_axis=2,
            tangent_axis=0,
            face_sign=-1.0,
            face_half=template_z_half,
            tangent_half=template_x_half,
        ),
    ]
    flat_z_ratios = [
        _flat_half_ratio_for_face(
            vertices,
            face_axis=0,
            tangent_axis=2,
            face_sign=1.0,
            face_half=template_x_half,
            tangent_half=template_z_half,
        ),
        _flat_half_ratio_for_face(
            vertices,
            face_axis=0,
            tangent_axis=2,
            face_sign=-1.0,
            face_half=template_x_half,
            tangent_half=template_z_half,
        ),
    ]
    flat_half_x_m = x_half_m * float(np.median(flat_x_ratios))
    flat_half_z_m = z_half_m * float(np.median(flat_z_ratios))
    shrink_x_m = max(0.0, x_half_m - flat_half_x_m)
    shrink_z_m = max(0.0, z_half_m - flat_half_z_m)
    if shrink_x_m <= 0.0 or shrink_z_m <= 0.0:
        raise ValueError("cad_profile_invalid_flat_extent")
    return _CadFlatEdgeProfile(
        method="cad_flat_edge_profile_v1",
        flat_half_x_m=flat_half_x_m,
        flat_half_z_m=flat_half_z_m,
        shrink_x_m=shrink_x_m,
        shrink_z_m=shrink_z_m,
        fallback_reason=None,
    )


def _cad_flat_edge_profile(size: np.ndarray, cad_path: Path | None = None) -> _CadFlatEdgeProfile:
    path = DEFAULT_TUBE_CAD_PATH if cad_path is None else Path(cad_path)
    try:
        vertices = _load_obj_vertices(path)
        return _derive_cad_flat_edge_profile_from_vertices(vertices, size)
    except Exception as exc:
        reason = str(exc) or exc.__class__.__name__
        if not reason.startswith("cad_profile_"):
            reason = "cad_profile_load_failed"
        return _bbox_flat_edge_profile(size, reason)


def _score_face_from_plane(candidate: _FaceCandidate, fit: _PlatePlaneFit, size: np.ndarray) -> dict:
    if fit.origin is None or fit.normal is None:
        return {
            "face": candidate.name,
            "score": 0.0,
            "distance_m": float("inf"),
            "normal_alignment": 0.0,
            "coverage_ratio": 0.0,
        }

    half = 0.5 * np.asarray(size, dtype=np.float64)
    face_normal = np.zeros(3, dtype=np.float64)
    face_normal[candidate.plane_axis] = candidate.plane_sign
    face_center = np.zeros(3, dtype=np.float64)
    face_center[candidate.plane_axis] = candidate.plane_sign * half[candidate.plane_axis]

    distance_m = abs(float((face_center - fit.origin) @ fit.normal))
    normal_alignment = abs(float(np.dot(face_normal, fit.normal)))
    distance_score = max(0.0, 1.0 - distance_m / max(PLANE_DISTANCE_TOL_M * 4.0, 1e-6))

    if len(fit.inliers) == 0:
        coverage_ratio = 0.0
    else:
        face_value = candidate.plane_sign * half[candidate.plane_axis]
        near_face = np.abs(fit.inliers[:, candidate.plane_axis] - face_value) <= (
            PLANE_DISTANCE_TOL_M * 3.0
        )
        coverage_ratio = float(np.count_nonzero(near_face)) / float(len(fit.inliers))

    score = 0.45 * normal_alignment + 0.35 * distance_score + 0.20 * coverage_ratio
    return {
        "face": candidate.name,
        "score": round(float(score), 6),
        "distance_m": round(float(distance_m), 6),
        "normal_alignment": round(float(normal_alignment), 6),
        "coverage_ratio": round(float(coverage_ratio), 6),
    }


def _span_from_plate_inliers(
    fit: _PlatePlaneFit, size: np.ndarray
) -> tuple[tuple[float, float] | None, str]:
    if len(fit.inliers) == 0:
        return None, "empty_plate_plane_inliers"

    half_y = 0.5 * float(size[1])
    y_values = fit.inliers[:, 1]
    y_min, y_max = np.percentile(y_values, [2.0, 98.0])
    y_min = max(-half_y, float(y_min))
    y_max = min(half_y, float(y_max))
    min_span = min(float(size[1]) * 0.15, 0.04)
    if y_max - y_min < min_span:
        return None, "plate_plane_span_too_short"
    return (y_min, y_max), "plate_plane_inliers"


def _fallback_span(size: np.ndarray) -> tuple[float, float]:
    y_half = 0.5 * float(size[1])
    half_span = y_half * FALLBACK_SPAN_RATIO
    return -half_span, half_span


def _fallback_selection(
    *,
    size: np.ndarray,
    reason: str,
    candidate_scores: list[dict],
    ambiguous: bool,
    plate_plane_fit: dict | None = None,
) -> dict:
    span_y = _fallback_span(size)
    metadata = {
        "selected_face": DEFAULT_CONTACT_FACE,
        "method": "fallback_short_span_v1",
        "ambiguous": bool(ambiguous),
        "score_margin": 0.0,
        "candidate_scores": candidate_scores,
        "span_y_m": [float(span_y[0]), float(span_y[1])],
        "span_source": "fallback_short_span",
        "fallback_reason": reason,
    }
    if plate_plane_fit is not None:
        metadata["plate_plane_fit"] = plate_plane_fit
    return {"selected_face": DEFAULT_CONTACT_FACE, "span_y": span_y, "metadata": metadata}


def _intrinsic(intrinsics: dict, key: str) -> float:
    value = float(intrinsics[key])
    if not np.isfinite(value) or value == 0.0:
        raise ValueError(f"invalid camera intrinsic: {key}")
    return value
