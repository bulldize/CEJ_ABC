import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from src.utils.geometry import tooth_surface


def _unit(vec, fallback=None):
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n > 1e-6 and np.all(np.isfinite(v)):
        return (v / n).astype(np.float32)
    if fallback is None:
        return None
    return np.asarray(fallback, dtype=np.float32)


def _project_out(vec, *axes):
    out = np.asarray(vec, dtype=np.float32).copy()
    for axis in axes:
        a = _unit(axis)
        if a is not None:
            out = out - float(np.dot(out, a)) * a
    return out


def _fallback_perpendicular(axis):
    axis = _unit(axis, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    candidates = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]
    best = min(candidates, key=lambda c: abs(float(np.dot(c, axis))))
    return _unit(_project_out(best, axis), fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))


def _pca(points_mm):
    pts = np.asarray(points_mm, dtype=np.float32)
    center = pts.mean(axis=0)
    centered = pts - center
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    return center.astype(np.float32), s.astype(np.float32), vt.astype(np.float32)


def _orient_long_axis(points_mm, axis):
    axis = _unit(axis, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    center = np.asarray(points_mm, dtype=np.float32).mean(axis=0)
    local = np.asarray(points_mm, dtype=np.float32) - center
    t = local @ axis
    if t.size < 8 or float(np.ptp(t)) <= 1e-6:
        return axis, 0.25, {"method": "pca_unoriented"}

    lo = np.percentile(t, 20)
    hi = np.percentile(t, 80)
    low_pts = local[t <= lo]
    high_pts = local[t >= hi]

    def _radial_spread(arr):
        if arr.shape[0] == 0:
            return 0.0
        axial = (arr @ axis)[:, None] * axis[None, :]
        radial = arr - axial
        return float(np.median(np.linalg.norm(radial, axis=1)))

    low_spread = _radial_spread(low_pts)
    high_spread = _radial_spread(high_pts)
    if high_spread < low_spread:
        axis = -axis
        low_spread, high_spread = high_spread, low_spread

    denom = max(high_spread, low_spread, 1e-6)
    confidence = float(np.clip(abs(high_spread - low_spread) / denom, 0.05, 1.0))
    return axis.astype(np.float32), confidence, {
        "method": "pca_end_radial_spread",
        "root_end_radial_spread_mm": float(low_spread),
        "crown_end_radial_spread_mm": float(high_spread),
    }


def _same_quadrant_centroids(tooth_id, centroids):
    tid = int(tooth_id)
    quadrant = tid // 10
    out = {}
    for key, value in (centroids or {}).items():
        k = int(key)
        if k // 10 == quadrant:
            out[k] = np.asarray(value, dtype=np.float32)
    return out


def _estimate_mesial_axis(tooth_id, centroids, long_axis, pca_axes):
    tid = int(tooth_id) if tooth_id is not None else None
    long_axis = _unit(long_axis, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if tid is not None and centroids and tid in {int(k) for k in centroids.keys()}:
        centroids_int = {int(k): np.asarray(v, dtype=np.float32) for k, v in centroids.items()}
        cur = centroids_int[tid]
        pos = tid % 10
        candidates = []
        sources = []
        if pos > 1 and (tid - 1) in centroids_int:
            candidates.append(centroids_int[tid - 1] - cur)
            sources.append(f"mesial_neighbor_{tid - 1}")
        if pos < 8 and (tid + 1) in centroids_int:
            candidates.append(cur - centroids_int[tid + 1])
            sources.append(f"distal_neighbor_{tid + 1}")
        if candidates:
            vec = np.sum([_unit(v, fallback=np.zeros(3, dtype=np.float32)) for v in candidates], axis=0)
            vec = _project_out(vec, long_axis)
            axis = _unit(vec)
            if axis is not None:
                return axis, 0.9, {"method": "adjacent_tooth_centroids", "sources": sources}

        same_q = _same_quadrant_centroids(tid, centroids_int)
        if len(same_q) >= 3:
            ids = sorted(same_q)
            pts = np.stack([same_q[i] for i in ids], axis=0)
            _, _, vt = _pca(pts)
            axis = _unit(_project_out(vt[0], long_axis))
            if axis is not None:
                projections = pts @ axis
                corr = float(np.corrcoef(np.asarray([i % 10 for i in ids], dtype=np.float32), projections)[0, 1])
                if np.isfinite(corr) and corr > 0:
                    axis = -axis
                return axis, 0.65, {"method": "quadrant_arch_tangent", "n_teeth": len(ids)}

    for idx in range(1, min(3, pca_axes.shape[0])):
        axis = _unit(_project_out(pca_axes[idx], long_axis))
        if axis is not None:
            return axis, 0.35, {"method": "tooth_pca_fallback", "pca_axis_index": int(idx)}

    return _fallback_perpendicular(long_axis), 0.2, {"method": "orthogonal_basis_fallback"}


def _estimate_buccal_axis(long_axis, mesial_axis, current_centroid_mm=None, arch_centroid_mm=None):
    long_axis = _unit(long_axis, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    mesial_axis = _unit(_project_out(mesial_axis, long_axis), fallback=_fallback_perpendicular(long_axis))
    buccal_axis = _unit(np.cross(long_axis, mesial_axis), fallback=_fallback_perpendicular(long_axis))
    source = {"method": "cross_long_mesial"}
    confidence = 0.35

    if current_centroid_mm is not None and arch_centroid_mm is not None:
        outward = np.asarray(current_centroid_mm, dtype=np.float32) - np.asarray(arch_centroid_mm, dtype=np.float32)
        outward = _project_out(outward, long_axis, mesial_axis)
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm > 1e-6:
            outward = outward / outward_norm
            if float(np.dot(buccal_axis, outward)) < 0:
                buccal_axis = -buccal_axis
            confidence = 0.65
            source = {"method": "arch_center_outward", "projected_outward_norm_mm": outward_norm}

    return buccal_axis.astype(np.float32), confidence, source


def _surface_points_mm(mask, spacing):
    surface = tooth_surface(np.asarray(mask, dtype=np.uint8))
    pts = np.argwhere(surface > 0).astype(np.float32)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return (pts * np.asarray(spacing, dtype=np.float32)).astype(np.float32)


def build_geometry_prior(
    tooth_mask,
    spacing,
    tooth_id=None,
    case_id=None,
    points_vox=None,
    full_tooth_centroids_mm: Optional[Dict[int, np.ndarray]] = None,
    arch_centroid_mm=None,
    roi_origin_vox=None,
    spacing_full=None,
    constraint_params: Optional[Dict] = None,
):
    spacing_arr = np.asarray(spacing, dtype=np.float32)
    mask = np.asarray(tooth_mask, dtype=np.uint8)
    idx = np.argwhere(mask > 0).astype(np.float32)
    constraint_params = dict(constraint_params or {})

    prior = {
        "version": 1,
        "case_id": case_id,
        "tooth_id": int(tooth_id) if tooth_id is not None else None,
        "enabled": True,
        "fallback": False,
        "fallback_reason": None,
        "direction_labels": {
            "long_positive": "crown",
            "long_negative": "root",
            "mesial_positive": "mesial",
            "mesial_negative": "distal",
            "buccal_positive": "buccal",
            "buccal_negative": "lingual",
        },
        "constraint_params": constraint_params,
        "constraints": {
            "mesial_distal": "crownward_local_peak",
            "buccal_lingual": "rootward_local_valley",
        },
    }

    if idx.shape[0] < 8:
        prior.update(
            {
                "enabled": False,
                "fallback": True,
                "fallback_reason": "empty_or_tiny_tooth_mask",
                "axes": {},
                "confidence": {"overall": 0.0},
                "_runtime": {"tooth_mask": mask, "surface_points_mm": np.zeros((0, 3), dtype=np.float32)},
            }
        )
        return prior

    pts_mm = idx * spacing_arr
    center_mm, s, vt = _pca(pts_mm)
    long_axis, long_conf, long_source = _orient_long_axis(pts_mm, vt[0])

    centroids = {int(k): np.asarray(v, dtype=np.float32) for k, v in (full_tooth_centroids_mm or {}).items()}
    current_centroid_mm = centroids.get(int(tooth_id)) if tooth_id is not None and int(tooth_id) in centroids else None
    if current_centroid_mm is None:
        current_centroid_mm = center_mm
        if roi_origin_vox is not None:
            sp_full = np.asarray(spacing_full if spacing_full is not None else spacing_arr, dtype=np.float32)
            current_centroid_mm = np.asarray(roi_origin_vox, dtype=np.float32) * sp_full + center_mm

    if arch_centroid_mm is None and centroids:
        arch_centroid_mm = np.stack(list(centroids.values()), axis=0).mean(axis=0)

    mesial_axis, md_conf, md_source = _estimate_mesial_axis(tooth_id, centroids, long_axis, vt)
    mesial_axis = _unit(_project_out(mesial_axis, long_axis), fallback=_fallback_perpendicular(long_axis))
    buccal_axis, bl_conf, bl_source = _estimate_buccal_axis(
        long_axis,
        mesial_axis,
        current_centroid_mm=current_centroid_mm,
        arch_centroid_mm=arch_centroid_mm,
    )
    buccal_axis = _unit(_project_out(buccal_axis, long_axis, mesial_axis), fallback=np.cross(long_axis, mesial_axis))
    if buccal_axis is None:
        buccal_axis = _unit(np.cross(long_axis, mesial_axis), fallback=_fallback_perpendicular(long_axis))
    mesial_axis = _unit(np.cross(buccal_axis, long_axis), fallback=mesial_axis)

    if points_vox is not None and len(points_vox) > 0:
        origin_vox = np.asarray(points_vox, dtype=np.float32).mean(axis=0)
        origin_mm = origin_vox * spacing_arr
    else:
        origin_mm = center_mm
        origin_vox = origin_mm / spacing_arr

    total_var = float(np.sum(s ** 2)) if s.size else 0.0
    elongation = float((s[0] ** 2) / total_var) if total_var > 1e-6 else 0.0
    overall = float(np.clip(np.mean([long_conf, md_conf, bl_conf]), 0.0, 1.0))

    prior.update(
        {
            "origin_vox": [float(v) for v in origin_vox],
            "origin_mm": [float(v) for v in origin_mm],
            "tooth_centroid_vox": [float(v) for v in (center_mm / spacing_arr)],
            "tooth_centroid_mm": [float(v) for v in center_mm],
            "full_tooth_centroid_mm": [float(v) for v in current_centroid_mm],
            "arch_centroid_mm": [float(v) for v in arch_centroid_mm] if arch_centroid_mm is not None else None,
            "axes": {
                "long_axis_root_to_crown": [float(v) for v in long_axis],
                "mesial_axis": [float(v) for v in mesial_axis],
                "buccal_axis": [float(v) for v in buccal_axis],
            },
            "axis_sources": {
                "long_axis": long_source,
                "mesial_distal_axis": md_source,
                "buccal_lingual_axis": bl_source,
            },
            "confidence": {
                "long_axis": float(long_conf),
                "mesial_distal": float(md_conf),
                "buccal_lingual": float(bl_conf),
                "pca_elongation": elongation,
                "overall": overall,
            },
            "_runtime": {"tooth_mask": mask, "surface_points_mm": _surface_points_mm(mask, spacing_arr)},
        }
    )
    return prior


def geometry_prior_to_jsonable(prior):
    if prior is None:
        return None
    out = {}
    for key, value in prior.items():
        if key == "_runtime":
            continue
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        elif isinstance(value, dict):
            out[key] = geometry_prior_to_jsonable(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [
                item.item() if isinstance(item, (np.floating, np.integer)) else item
                for item in value
            ]
        else:
            out[key] = value
    return out


def _axes_from_prior(prior):
    axes = (prior or {}).get("axes", {})
    long_axis = _unit(axes.get("long_axis_root_to_crown"))
    mesial_axis = _unit(axes.get("mesial_axis"))
    buccal_axis = _unit(axes.get("buccal_axis"))
    if long_axis is None or mesial_axis is None or buccal_axis is None:
        return None
    mesial_axis = _unit(_project_out(mesial_axis, long_axis))
    buccal_axis = _unit(_project_out(buccal_axis, long_axis, mesial_axis))
    if mesial_axis is None or buccal_axis is None:
        return None
    return long_axis, mesial_axis, buccal_axis


def _periodic_interp(theta, values, theta_new):
    theta = np.asarray(theta, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    order = np.argsort(theta)
    theta = theta[order]
    values = values[order]
    keep = np.ones(theta.shape[0], dtype=bool)
    keep[1:] = np.diff(theta) > 1e-4
    theta = theta[keep]
    values = values[keep]
    if theta.shape[0] == 0:
        return np.zeros_like(theta_new, dtype=np.float32)
    if theta.shape[0] == 1:
        return np.full_like(theta_new, float(values[0]), dtype=np.float32)
    theta_ext = np.concatenate([theta, theta[:1] + 2.0 * np.pi])
    values_ext = np.concatenate([values, values[:1]])
    return np.interp(theta_new, theta_ext, values_ext).astype(np.float32)


def _circular_smooth(values, sigma_samples):
    if sigma_samples <= 0:
        return values.astype(np.float32)
    return gaussian_filter1d(values.astype(np.float32), sigma=float(sigma_samples), mode="wrap")


def check_curve_constraints(points_vox, spacing, geometry_prior, tolerance_mm=0.03):
    axes = _axes_from_prior(geometry_prior)
    pts = np.asarray(points_vox, dtype=np.float32)
    if axes is None or pts.ndim != 2 or pts.shape[0] < 8:
        return {
            "passed": False,
            "reason": "missing_axes_or_curve",
            "direction_values_mm": {},
        }

    long_axis, mesial_axis, buccal_axis = axes
    spacing_arr = np.asarray(spacing, dtype=np.float32)
    origin_mm = np.asarray(geometry_prior.get("origin_mm", (pts * spacing_arr).mean(axis=0)), dtype=np.float32)
    local = pts * spacing_arr - origin_mm
    x = local @ mesial_axis
    y = local @ buccal_axis
    z = local @ long_axis
    theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)

    def _value_at(target):
        idx = int(np.argmin(np.abs(np.angle(np.exp(1j * (theta - target))))))
        return float(z[idx])

    z_mesial = _value_at(0.0)
    z_buccal = _value_at(0.5 * np.pi)
    z_distal = _value_at(np.pi)
    z_lingual = _value_at(1.5 * np.pi)
    peak_margin = min(z_mesial, z_distal) - max(z_buccal, z_lingual)
    passed = bool(peak_margin > float(tolerance_mm))
    return {
        "passed": passed,
        "reason": None if passed else "mesial_distal_not_crownward_peak_or_buccal_lingual_not_rootward_valley",
        "tolerance_mm": float(tolerance_mm),
        "peak_margin_mm": float(peak_margin),
        "direction_values_mm": {
            "mesial": z_mesial,
            "distal": z_distal,
            "buccal": z_buccal,
            "lingual": z_lingual,
        },
    }


def fit_curve_with_geometry_prior(points_vox, spacing, step_mm, closed=True, smooth=0.0, geometry_prior=None):
    pts = np.asarray(points_vox, dtype=np.float32)
    spacing_arr = np.asarray(spacing, dtype=np.float32)
    report = {
        "used_geometry_prior": bool(geometry_prior is not None),
        "fallback": False,
        "fallback_reason": None,
        "n_input_points": int(pts.shape[0]) if pts.ndim == 2 else 0,
        "n_output_points": 0,
    }

    if geometry_prior is None:
        report["fallback"] = True
        report["fallback_reason"] = "no_geometry_prior"
        return None, report
    if not bool(closed):
        report["fallback"] = True
        report["fallback_reason"] = "open_curve_not_supported_by_geometry_prior"
        return None, report
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 3:
        report["fallback"] = True
        report["fallback_reason"] = "too_few_points"
        return None, report
    if geometry_prior.get("fallback") or not geometry_prior.get("enabled", True):
        report["fallback"] = True
        report["fallback_reason"] = geometry_prior.get("fallback_reason") or "geometry_prior_disabled"
        return None, report

    axes = _axes_from_prior(geometry_prior)
    if axes is None:
        report["fallback"] = True
        report["fallback_reason"] = "missing_geometry_axes"
        return None, report

    long_axis, mesial_axis, buccal_axis = axes
    origin_mm = np.asarray(geometry_prior.get("origin_mm", (pts * spacing_arr).mean(axis=0)), dtype=np.float32)
    pts_mm = pts * spacing_arr
    local = pts_mm - origin_mm
    x = local @ mesial_axis
    y = local @ buccal_axis
    z = local @ long_axis
    radius = np.sqrt(x * x + y * y)
    theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)

    valid = np.isfinite(theta) & np.isfinite(radius) & np.isfinite(z) & (radius > 1e-4)
    if int(valid.sum()) < 4:
        report["fallback"] = True
        report["fallback_reason"] = "degenerate_local_coordinates"
        return None, report
    theta = theta[valid]
    radius = radius[valid]
    z = z[valid]

    order = np.argsort(theta)
    pts_local_sorted = np.stack([x[valid], y[valid], z], axis=1)[order]
    loop = np.vstack([pts_local_sorted, pts_local_sorted[:1]])
    perimeter = float(np.linalg.norm(np.diff(loop, axis=0), axis=1).sum())
    n = max(64, int(math.ceil(perimeter / max(float(step_mm), 1e-3))))
    theta_dense = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)

    params = dict(geometry_prior.get("constraint_params") or {})
    radius_dense = _periodic_interp(theta, radius, theta_dense)
    min_radius = max(0.05, float(np.percentile(radius, 5)) * 0.35)
    radius_dense = np.maximum(radius_dense, min_radius)

    smooth_sigma_mm = float(params.get("radial_smooth_sigma_mm", max(0.0, float(smooth))))
    if smooth_sigma_mm > 0:
        sigma_samples = smooth_sigma_mm / max(float(step_mm), 1e-3)
        radius_dense = _circular_smooth(radius_dense, sigma_samples=sigma_samples)

    z_center = float(np.median(z))
    z_range = float(np.percentile(z, 95) - np.percentile(z, 5)) if z.size > 1 else 0.0
    radial_ref = float(np.median(radius_dense)) if radius_dense.size > 0 else 1.0
    min_amp = float(params.get("min_crown_root_amplitude_mm", 0.65))
    amp = max(min_amp, 0.35 * z_range, 0.025 * radial_ref)
    max_amp = params.get("max_crown_root_amplitude_mm", None)
    if max_amp is not None:
        amp = min(amp, float(max_amp))

    z_dense = z_center + amp * np.cos(2.0 * theta_dense)
    local_dense = np.stack(
        [
            radius_dense * np.cos(theta_dense),
            radius_dense * np.sin(theta_dense),
            z_dense,
        ],
        axis=1,
    ).astype(np.float32)

    dense_mm = (
        origin_mm[None, :]
        + local_dense[:, 0:1] * mesial_axis[None, :]
        + local_dense[:, 1:2] * buccal_axis[None, :]
        + local_dense[:, 2:3] * long_axis[None, :]
    )

    projection_report = {"enabled": False}
    runtime = geometry_prior.get("_runtime") or {}
    surface_points = runtime.get("surface_points_mm")
    if surface_points is not None and len(surface_points) > 0 and bool(params.get("project_to_surface", True)):
        max_dist = float(params.get("max_surface_projection_mm", 2.5))
        tree = cKDTree(np.asarray(surface_points, dtype=np.float32))
        d, nearest = tree.query(dense_mm, k=1)
        d = np.asarray(d, dtype=np.float32)
        snapped = d <= max_dist
        dense_mm_projected = dense_mm.copy()
        dense_mm_projected[snapped] = np.asarray(surface_points, dtype=np.float32)[nearest[snapped]]

        if bool(params.get("enforce_constraints_after_projection", True)):
            projected_local = dense_mm_projected - origin_mm
            px = projected_local @ mesial_axis
            py = projected_local @ buccal_axis
            ptheta = np.mod(np.arctan2(py, px), 2.0 * np.pi)
            pz = z_center + amp * np.cos(2.0 * ptheta)
            dense_mm_projected = (
                origin_mm[None, :]
                + px[:, None] * mesial_axis[None, :]
                + py[:, None] * buccal_axis[None, :]
                + pz[:, None] * long_axis[None, :]
            )

        dense_mm = dense_mm_projected.astype(np.float32)
        projection_report = {
            "enabled": True,
            "max_surface_projection_mm": max_dist,
            "snapped_points": int(snapped.sum()),
            "unsnapped_points": int((~snapped).sum()),
            "mean_nearest_surface_mm": float(np.mean(d)) if d.size else None,
            "p95_nearest_surface_mm": float(np.percentile(d, 95)) if d.size else None,
            "max_nearest_surface_mm": float(np.max(d)) if d.size else None,
        }

    dense_vox = dense_mm / spacing_arr
    constraint_report = check_curve_constraints(
        dense_vox,
        spacing_arr,
        geometry_prior,
        tolerance_mm=float(params.get("constraint_tolerance_mm", 0.03)),
    )
    report.update(
        {
            "n_output_points": int(dense_vox.shape[0]),
            "closed": bool(closed),
            "step_mm": float(step_mm),
            "local_origin_mm": [float(v) for v in origin_mm],
            "z_center_mm": z_center,
            "crown_root_amplitude_mm": float(amp),
            "radial_median_mm": float(np.median(radius_dense)) if radius_dense.size else None,
            "surface_projection": projection_report,
            "constraints": constraint_report,
        }
    )
    return dense_vox.astype(np.float32), report


def _distance_summary_mm(points_a_vox, points_b_vox, spacing):
    a = np.asarray(points_a_vox, dtype=np.float32)
    b = np.asarray(points_b_vox, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0 or b.shape[0] == 0:
        return None
    spacing_arr = np.asarray(spacing, dtype=np.float32)
    a_mm = a * spacing_arr
    b_mm = b * spacing_arr
    tree = cKDTree(b_mm)
    d, _ = tree.query(a_mm, k=1)
    d = np.asarray(d, dtype=np.float32)
    return {
        "count": int(d.shape[0]),
        "mean_mm": float(np.mean(d)),
        "p95_mm": float(np.percentile(d, 95)),
        "max_mm": float(np.max(d)),
    }


def curve_smoothness_summary(points_vox, spacing):
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return None
    pts_mm = pts * np.asarray(spacing, dtype=np.float32)
    loop = np.vstack([pts_mm, pts_mm[:2]])
    v1 = np.diff(loop[:-1], axis=0)
    v2 = np.diff(loop[1:], axis=0)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-6) & (n2 > 1e-6)
    if not np.any(valid):
        return None
    cosang = np.sum(v1[valid] * v2[valid], axis=1) / (n1[valid] * n2[valid])
    cosang = np.clip(cosang, -1.0, 1.0)
    angles = np.arccos(cosang)
    return {
        "mean_turn_angle_deg": float(np.degrees(np.mean(angles))),
        "p95_turn_angle_deg": float(np.degrees(np.percentile(angles, 95))),
        "max_turn_angle_deg": float(np.degrees(np.max(angles))),
    }


def compute_curve_fit_report(
    manual_points_vox,
    dense_curve_vox,
    spacing,
    C_gt=None,
    geometry_prior=None,
    fit_report=None,
):
    manual = np.asarray(manual_points_vox, dtype=np.float32)
    dense = np.asarray(dense_curve_vox, dtype=np.float32)
    report = {
        "fit": fit_report or {},
        "n_manual_points": int(manual.shape[0]) if manual.ndim == 2 else 0,
        "n_dense_points": int(dense.shape[0]) if dense.ndim == 2 else 0,
        "manual_to_curve": _distance_summary_mm(manual, dense, spacing),
        "smoothness": curve_smoothness_summary(dense, spacing),
    }

    if geometry_prior is not None:
        report["constraints"] = check_curve_constraints(
            dense,
            spacing,
            geometry_prior,
            tolerance_mm=float((geometry_prior.get("constraint_params") or {}).get("constraint_tolerance_mm", 0.03)),
        )
        report["direction_confidence"] = geometry_prior.get("confidence")

    if C_gt is not None:
        c_pts = np.argwhere(np.asarray(C_gt) > 0).astype(np.float32)
        report["curve_to_gt_curve"] = _distance_summary_mm(dense, c_pts, spacing)
        report["gt_curve_to_curve"] = _distance_summary_mm(c_pts, dense, spacing)
        report["gt_curve_voxels"] = int(c_pts.shape[0])
    else:
        report["curve_to_gt_curve"] = None
        report["gt_curve_to_curve"] = None
        report["gt_curve_voxels"] = 0

    return report
