import numpy as np
from monai.transforms import KeepLargestConnectedComponent
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt, label

from src.utils.geometry import tooth_surface

try:
    from skimage.morphology import skeletonize_3d as _skeletonize
except Exception:
    from skimage.morphology import skeletonize as _skeletonize

_keep_lcc = KeepLargestConnectedComponent(applied_labels=[1], is_onehot=False, connectivity=1)
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def compute_curve_overlap_metrics(mask_a, mask_b):
    a = (np.asarray(mask_a) > 0)
    b = (np.asarray(mask_b) > 0)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    vox_a = int(a.sum())
    vox_b = int(b.sum())
    denom_dice = vox_a + vox_b
    return {
        "equal_voxelwise": bool(np.array_equal(a, b)),
        "intersection_voxels": inter,
        "union_voxels": union,
        "curve_a_voxels": vox_a,
        "curve_b_voxels": vox_b,
        "iou": float(inter / union) if union > 0 else 1.0,
        "dice": float((2.0 * inter) / denom_dice) if denom_dice > 0 else 1.0,
    }


def extract_curve_from_heatmap_peak(mask_prob, tooth_mask=None, peak_threshold=0.999, keep_lcc=False):
    curve = (mask_prob >= float(peak_threshold)).astype(np.uint8)
    if tooth_mask is not None:
        curve = curve * (tooth_mask > 0).astype(np.uint8)
    if curve.sum() == 0:
        return curve
    if keep_lcc:
        curve = _keep_lcc(curve[None, ...])[0]
    return curve.astype(np.uint8)


def _filter_components(binary, min_voxels=8, max_components=4, keep_lcc=False):
    binary = (np.asarray(binary) > 0)
    if int(binary.sum()) == 0:
        return binary.astype(np.uint8), {
            "component_count": 0,
            "component_count_kept": 0,
            "largest_component_voxels": 0,
            "largest_component_ratio": 0.0,
            "candidate_voxels_after_components": 0,
        }

    labeled, n_components = label(binary, structure=_CONNECTIVITY_26)
    if n_components == 0:
        return np.zeros_like(binary, dtype=np.uint8), {
            "component_count": 0,
            "component_count_kept": 0,
            "largest_component_voxels": 0,
            "largest_component_ratio": 0.0,
            "candidate_voxels_after_components": 0,
        }

    sizes = np.bincount(labeled.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    min_voxels = max(1, int(min_voxels))
    max_components = max(1, int(max_components))

    if keep_lcc:
        keep_labels = [int(order[0]) + 1]
    else:
        keep_labels = [int(i) + 1 for i in order[:max_components] if int(sizes[i]) >= min_voxels]
        if not keep_labels:
            keep_labels = [int(order[0]) + 1]

    filtered = np.isin(labeled, keep_labels)
    largest = int(sizes[order[0]]) if sizes.size else 0
    kept_voxels = int(filtered.sum())
    stats = {
        "component_count": int(n_components),
        "component_count_kept": int(len(keep_labels)),
        "largest_component_voxels": largest,
        "largest_component_ratio": float(largest / max(1, int(binary.sum()))),
        "candidate_voxels_after_components": kept_voxels,
    }
    return filtered.astype(np.uint8), stats


def _threshold_heatmap_candidates(mask_prob, tooth_mask, cfg):
    prob = np.asarray(mask_prob, dtype=np.float32)
    shape = prob.shape
    tooth = (np.asarray(tooth_mask) > 0) if tooth_mask is not None else np.ones(shape, dtype=bool)
    if tooth.shape != shape:
        raise ValueError(f"tooth_mask shape {tooth.shape} does not match heatmap shape {shape}")

    diag = {
        "heatmap_max": float(np.max(prob)) if prob.size else 0.0,
        "selected_threshold": None,
        "candidate_voxels_before_components": 0,
        "candidate_voxels_after_components": 0,
        "component_count": 0,
        "component_count_kept": 0,
        "largest_component_voxels": 0,
        "largest_component_ratio": 0.0,
        "fallback_used": False,
        "empty_reason": None,
    }

    if prob.size == 0 or diag["heatmap_max"] <= 0.0 or int(tooth.sum()) == 0:
        diag["empty_reason"] = "empty_heatmap_or_tooth_mask"
        return np.zeros(shape, dtype=np.uint8), diag

    threshold = float(cfg.get("threshold_theta", 0.3))
    relative = float(cfg.get("relative_threshold", 0.45))
    fallback_min = float(cfg.get("fallback_min_threshold", 0.1))
    min_voxels = int(cfg.get("min_candidate_voxels", cfg.get("fit_pred_curve_min_points", 8)))
    min_voxels = max(1, min_voxels)

    primary = max(threshold, diag["heatmap_max"] * relative)
    lower = min(primary, max(0.0, fallback_min))
    if primary <= lower:
        thresholds = [primary]
    else:
        thresholds = np.linspace(primary, lower, num=6, dtype=np.float32).tolist()

    selected = np.zeros(shape, dtype=np.uint8)
    selected_stats = None
    selected_threshold = float(thresholds[-1])
    before = 0
    for idx, thr in enumerate(thresholds):
        candidate = (prob >= float(thr)) & tooth
        before = int(candidate.sum())
        filtered, comp_stats = _filter_components(
            candidate,
            min_voxels=int(cfg.get("min_component_voxels", 8)),
            max_components=int(cfg.get("max_curve_components", 4)),
            keep_lcc=bool(cfg.get("keep_lcc_for_curve", False)),
        )
        selected = filtered
        selected_stats = comp_stats
        selected_threshold = float(thr)
        if int(selected.sum()) >= min_voxels:
            diag["fallback_used"] = idx > 0
            break
        diag["fallback_used"] = True

    diag["selected_threshold"] = selected_threshold
    diag["candidate_voxels_before_components"] = before
    if selected_stats is not None:
        diag.update(selected_stats)
    if before < min_voxels:
        diag["empty_reason"] = "too_few_candidate_voxels"
        return np.zeros(shape, dtype=np.uint8), diag
    if int(selected.sum()) < min_voxels:
        diag["empty_reason"] = "too_few_component_voxels"
        return np.zeros(shape, dtype=np.uint8), diag
    return selected.astype(np.uint8), diag


def extract_curve(mask_prob, tooth_mask, threshold=0.3, keep_lcc=True):
    binary = (mask_prob >= threshold).astype(np.uint8)
    binary = binary * (tooth_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary
    try:
        skel = _skeletonize(binary).astype(np.uint8)
    except Exception:
        skel = binary
    if skel.sum() == 0:
        return skel
    if keep_lcc:
        skel = _keep_lcc(skel[None, ...])[0]
    return skel.astype(np.uint8)


def _linear_resample(points_mm, step_mm):
    if len(points_mm) < 2:
        return points_mm
    d = np.linalg.norm(np.diff(points_mm, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total <= 0:
        return points_mm
    n = max(2, int(np.ceil(total / step_mm)) + 1)
    t_new = np.linspace(0.0, total, n)
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(3):
        out[:, i] = np.interp(t_new, cum, points_mm[:, i])
    return out


def _linear_resample_closed(points_mm, step_mm):
    if len(points_mm) < 3:
        return points_mm
    pts = np.asarray(points_mm, dtype=np.float32)
    pts_loop = np.vstack([pts, pts[0]])
    d = np.linalg.norm(np.diff(pts_loop, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total <= 0:
        return pts
    n = max(8, int(np.ceil(total / max(step_mm, 1e-3))))
    t_new = np.linspace(0.0, total, n, endpoint=False)
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(3):
        out[:, i] = np.interp(t_new, cum, pts_loop[:, i])
    return out


def _sort_points_by_plane_angle(points_mm):
    pts = np.asarray(points_mm, dtype=np.float32)
    if pts.shape[0] <= 2:
        return pts
    center = pts.mean(axis=0, keepdims=True)
    centered = pts - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    proj = centered @ basis
    ang = np.arctan2(proj[:, 1], proj[:, 0])
    order = np.argsort(ang)
    pts_sorted = pts[order]
    start = int(np.argmin(pts_sorted[:, 2]))
    return np.roll(pts_sorted, -start, axis=0)


def _plane_angles(points_mm):
    pts = np.asarray(points_mm, dtype=np.float32)
    if pts.shape[0] <= 2:
        return np.zeros((pts.shape[0],), dtype=np.float32)
    center = pts.mean(axis=0, keepdims=True)
    centered = pts - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    proj = centered @ basis
    return np.arctan2(proj[:, 1], proj[:, 0]).astype(np.float32)


def _filter_curve_outliers(points_vox, spacing, max_mad=3.5):
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.shape[0] < 8:
        return pts
    pts_mm = pts * np.asarray(spacing, dtype=np.float32)
    center = np.median(pts_mm, axis=0, keepdims=True)
    dist = np.linalg.norm(pts_mm - center, axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    if mad <= 1e-6:
        return pts
    keep = dist <= med + float(max_mad) * 1.4826 * mad
    if int(keep.sum()) < max(4, int(0.5 * pts.shape[0])):
        return pts
    return pts[keep]


def _angle_coverage(points_vox, spacing):
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.shape[0] < 4:
        return 0.0
    angles = np.sort(_plane_angles(pts * np.asarray(spacing, dtype=np.float32)))
    if angles.shape[0] < 2:
        return 0.0
    gaps = np.diff(np.concatenate([angles, angles[:1] + (2.0 * np.pi)]))
    largest_gap = float(np.max(gaps))
    coverage = max(0.0, (2.0 * np.pi - largest_gap) / (2.0 * np.pi))
    return float(coverage)


def fit_curve_and_sample(points_vox, spacing, step_mm, closed=True, smooth=0.0):
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    spacing = np.asarray(spacing, dtype=np.float32)
    pts_mm = pts * spacing
    pts_round = np.round(pts_mm, 4)
    _, uniq_idx = np.unique(pts_round, axis=0, return_index=True)
    pts_mm = pts_mm[np.sort(uniq_idx)]
    if pts_mm.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if closed:
        pts_mm = _sort_points_by_plane_angle(pts_mm)
    else:
        pts_mm = pts_mm[np.argsort(pts_mm[:, 2])]

    if pts_mm.shape[0] >= 4:
        try:
            tck, _ = splprep(pts_mm.T, s=float(smooth), per=bool(closed))
            dense_mm = _linear_resample_closed(pts_mm, step_mm) if closed else _linear_resample(pts_mm, step_mm)
            n = max(8 if closed else 2, dense_mm.shape[0])
            u = np.linspace(0.0, 1.0, n, endpoint=not closed)
            pts_dense = np.array(splev(u, tck)).T
        except Exception:
            pts_dense = _linear_resample_closed(pts_mm, step_mm) if closed else _linear_resample(pts_mm, step_mm)
    else:
        pts_dense = _linear_resample_closed(pts_mm, step_mm) if closed else _linear_resample(pts_mm, step_mm)

    return (pts_dense / spacing).astype(np.float32)


def rasterize_polyline_to_mask(mask, points_xyz, close_loop=False):
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return

    shape = np.asarray(mask.shape, dtype=np.int32)

    def _mark(p):
        q = np.round(p).astype(np.int32)
        if np.all(q >= 0) and np.all(q < shape):
            mask[q[0], q[1], q[2]] = 1

    _mark(pts[0])
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        n = int(np.ceil(np.max(np.abs(p1 - p0)))) + 1
        n = max(2, n)
        for p in np.linspace(p0, p1, n):
            _mark(p)

    if close_loop and pts.shape[0] > 2:
        p0 = pts[-1]
        p1 = pts[0]
        n = int(np.ceil(np.max(np.abs(p1 - p0)))) + 1
        n = max(2, n)
        for p in np.linspace(p0, p1, n):
            _mark(p)


def fit_curve_mask(
    curve_mask,
    spacing,
    step_mm,
    closed,
    smooth,
    min_points,
    min_closed_angle_coverage=0.55,
    max_outlier_mad=3.5,
):
    pts = np.argwhere(curve_mask > 0).astype(np.float32)
    if pts.shape[0] < max(2, int(min_points)):
        return curve_mask.astype(np.uint8), np.zeros((0, 3), dtype=np.float32)

    pts = _filter_curve_outliers(pts, spacing, max_mad=max_outlier_mad)
    if pts.shape[0] < max(2, int(min_points)):
        return curve_mask.astype(np.uint8), np.zeros((0, 3), dtype=np.float32)

    fit_closed = bool(closed)
    if fit_closed and _angle_coverage(pts, spacing) < float(min_closed_angle_coverage):
        fit_closed = False

    try:
        dense_pts = fit_curve_and_sample(
            pts,
            spacing=np.asarray(spacing, dtype=np.float32),
            step_mm=float(step_mm),
            closed=fit_closed,
            smooth=float(smooth),
        )
    except Exception:
        return curve_mask.astype(np.uint8), np.zeros((0, 3), dtype=np.float32)

    if dense_pts.ndim != 2 or dense_pts.shape[0] < 2:
        return curve_mask.astype(np.uint8), np.zeros((0, 3), dtype=np.float32)

    fit_mask = np.zeros_like(curve_mask, dtype=np.uint8)
    rasterize_polyline_to_mask(fit_mask, dense_pts, close_loop=fit_closed)
    if int(fit_mask.sum()) == 0:
        return curve_mask.astype(np.uint8), dense_pts.astype(np.float32)
    return fit_mask.astype(np.uint8), dense_pts.astype(np.float32)


def project_curve_to_tooth_surface(curve_mask, tooth_mask):
    curve = np.asarray(curve_mask) > 0
    tooth = np.asarray(tooth_mask) > 0
    if curve.sum() == 0 or tooth.sum() == 0:
        return np.asarray(curve_mask, dtype=np.uint8)

    surface = tooth_surface(tooth.astype(np.uint8))
    if surface.sum() == 0:
        return np.asarray(curve_mask, dtype=np.uint8)

    _, nearest = distance_transform_edt(~surface, return_indices=True)
    out = np.zeros_like(curve_mask, dtype=np.uint8)
    sx = nearest[0][curve]
    sy = nearest[1][curve]
    sz = nearest[2][curve]
    out[sx, sy, sz] = 1
    return out.astype(np.uint8)


def postprocess_prediction_curve(mask_prob, tooth_mask, spacing, cfg):
    constrain_to_tooth = bool(cfg.get("constrain_curve_to_tooth_mask", False))
    curve_tooth_mask = tooth_mask if constrain_to_tooth else np.ones_like(tooth_mask, dtype=np.uint8)
    candidates, diag = _threshold_heatmap_candidates(mask_prob, curve_tooth_mask, cfg)
    if bool(cfg.get("return_postprocess_diagnostics", False)):
        cfg["postprocess_diagnostics"] = diag

    if int(candidates.sum()) == 0:
        c_pred = candidates.astype(np.uint8)
    else:
        try:
            c_pred = _skeletonize(candidates).astype(np.uint8)
        except Exception:
            c_pred = candidates.astype(np.uint8)
        if int(c_pred.sum()) > 0:
            c_pred, skel_stats = _filter_components(
                c_pred,
                min_voxels=int(cfg.get("min_skeleton_component_voxels", 2)),
                max_components=int(cfg.get("max_curve_components", 4)),
                keep_lcc=bool(cfg.get("keep_lcc_for_curve", False)),
            )
            if bool(cfg.get("return_postprocess_diagnostics", False)):
                cfg["postprocess_diagnostics"].update(
                    {
                        "skeleton_component_count": skel_stats["component_count"],
                        "skeleton_component_count_kept": skel_stats["component_count_kept"],
                        "skeleton_voxels": int(c_pred.sum()),
                    }
                )

    c_fit = c_pred
    dense_pts = np.zeros((0, 3), dtype=np.float32)
    if bool(cfg.get("fit_pred_curve", True)):
        c_fit, dense_pts = fit_curve_mask(
            c_pred,
            spacing=spacing,
            step_mm=float(cfg.get("fit_pred_curve_step_mm", cfg.get("dense_sample_step_mm", 0.2))),
            closed=bool(cfg.get("fit_pred_curve_closed", cfg.get("curve_closed", True))),
            smooth=float(cfg.get("fit_pred_curve_smooth", cfg.get("curve_smooth", 0.0))),
            min_points=int(cfg.get("fit_pred_curve_min_points", 8)),
            min_closed_angle_coverage=float(cfg.get("fit_pred_curve_min_closed_angle_coverage", 0.55)),
            max_outlier_mad=float(cfg.get("fit_pred_curve_max_outlier_mad", 3.5)),
        )

    if bool(cfg.get("constrain_curve_to_tooth_surface", False)):
        c_fit = project_curve_to_tooth_surface(c_fit, tooth_mask)

    return c_pred.astype(np.uint8), c_fit.astype(np.uint8), dense_pts.astype(np.float32)
