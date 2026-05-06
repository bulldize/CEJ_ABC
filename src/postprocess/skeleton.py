import numpy as np
from monai.transforms import KeepLargestConnectedComponent
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt

from src.utils.geometry import tooth_surface

try:
    from skimage.morphology import skeletonize_3d as _skeletonize
except Exception:
    from skimage.morphology import skeletonize as _skeletonize

_keep_lcc = KeepLargestConnectedComponent(applied_labels=[1], is_onehot=False, connectivity=1)


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


def extract_curve(mask_prob, tooth_mask, threshold=0.3, keep_lcc=True):
    # Minimal postprocess: threshold -> intersect with tooth -> skeletonize -> largest component
    # TODO: replace with a more robust centerline extractor if needed.
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
    # keep largest component via MONAI when desired.
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


def fit_curve_mask(curve_mask, spacing, step_mm, closed, smooth, min_points):
    pts = np.argwhere(curve_mask > 0).astype(np.float32)
    if pts.shape[0] < max(2, int(min_points)):
        return curve_mask.astype(np.uint8), np.zeros((0, 3), dtype=np.float32)

    try:
        dense_pts = fit_curve_and_sample(
            pts,
            spacing=np.asarray(spacing, dtype=np.float32),
            step_mm=float(step_mm),
            closed=bool(closed),
            smooth=float(smooth),
        )
    except Exception:
        return curve_mask.astype(np.uint8), np.zeros((0, 3), dtype=np.float32)

    if dense_pts.ndim != 2 or dense_pts.shape[0] < 2:
        return curve_mask.astype(np.uint8), np.zeros((0, 3), dtype=np.float32)

    fit_mask = np.zeros_like(curve_mask, dtype=np.uint8)
    rasterize_polyline_to_mask(fit_mask, dense_pts, close_loop=bool(closed))
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
    c_pred = extract_curve(
        mask_prob,
        curve_tooth_mask,
        threshold=float(cfg.get("threshold_theta", 0.3)),
        keep_lcc=bool(cfg.get("keep_lcc_for_curve", False)),
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
        )

    if bool(cfg.get("constrain_curve_to_tooth_surface", False)):
        c_fit = project_curve_to_tooth_surface(c_fit, tooth_mask)

    return c_pred.astype(np.uint8), c_fit.astype(np.uint8), dense_pts.astype(np.float32)
