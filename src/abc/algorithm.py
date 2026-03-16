import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter1d,
    label,
)
from scipy.spatial import cKDTree


DEFAULT_EXTRACT_CFG = {
    "pdl_threshold_mm": 1.5,
    "angular_bins": 180,
    "smooth_sigma_bins": 2.0,
    "gradient_snap": True,
    "gradient_weight": 1.0,
    "snap_radius_mm": 1.5,
    "min_tooth_voxels": 64,
    "keep_lcc": True,
    "jump_outlier_suppression": True,
    "jump_outlier_ratio": 3.5,
    "jump_outlier_max_iter": 3,
    "axis_outlier_suppression": True,
    "axis_outlier_window": 5,
    "axis_outlier_mad_k": 3.0,
    "axis_outlier_min_thr_mm": 0.6,
    "axis_outlier_max_iter": 2,
}


def _safe_unit(vec):
    n = float(np.linalg.norm(vec))
    if n <= 1e-8:
        return None
    return vec / n


def _surface_mask(mask):
    m = np.asarray(mask) > 0
    if m.sum() == 0:
        return m
    er = binary_erosion(m, iterations=1)
    return np.logical_and(m, np.logical_not(er))


def _compute_local_frame(tooth_mask, spacing_xyz):
    spacing = np.asarray(spacing_xyz, dtype=np.float32)
    tooth_vox = np.array(np.where(tooth_mask > 0)).T.astype(np.float32)
    if tooth_vox.shape[0] < 3:
        return None

    tooth_mm = tooth_vox * spacing[None, :]
    centroid = np.mean(tooth_mm, axis=0)
    centered = tooth_mm - centroid[None, :]

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    axis = _safe_unit(axis)
    if axis is None:
        return None
    if axis[2] < 0:
        axis = -axis

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, axis))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = ref - np.dot(ref, axis) * axis
    u = _safe_unit(u)
    if u is None:
        return None
    v = np.cross(axis, u)
    v = _safe_unit(v)
    if v is None:
        return None

    return centroid.astype(np.float32), axis.astype(np.float32), u.astype(np.float32), v.astype(np.float32)


def _pick_points_by_angle(candidates_mm, centroid_mm, axis, u, v, bins):
    rel = candidates_mm - centroid_mm[None, :]
    coord_u = rel @ u
    coord_v = rel @ v
    coord_axis = rel @ axis
    theta = np.arctan2(coord_v, coord_u)

    bins = int(max(12, bins))
    sampled = np.full((bins, 3), np.nan, dtype=np.float32)
    theta_bin = ((theta + np.pi) / (2.0 * np.pi) * bins).astype(np.int32) % bins
    bin_centers = -np.pi + (np.arange(bins, dtype=np.float32) + 0.5) * (2.0 * np.pi / float(bins))

    for b in range(bins):
        idx = np.where(theta_bin == b)[0]
        if idx.size == 0:
            continue
        best = idx[int(np.argmax(coord_axis[idx]))]
        sampled[b] = candidates_mm[best]

    missing = np.where(np.isnan(sampled[:, 0]))[0]
    if missing.size > 0:
        for b in missing:
            center = bin_centers[b]
            d_theta = np.abs(np.angle(np.exp(1j * (theta - center))))
            if d_theta.size == 0:
                continue
            near = np.argsort(d_theta)[: min(24, d_theta.size)]
            best = near[int(np.argmax(coord_axis[near]))]
            sampled[b] = candidates_mm[best]

    return sampled


def _smooth_points_ring(points_mm, sigma_bins):
    if sigma_bins <= 0:
        return points_mm
    out = points_mm.copy()
    for k in range(3):
        out[:, k] = gaussian_filter1d(out[:, k], sigma=float(sigma_bins), mode="wrap")
    return out


def _snap_points_to_gradient(points_mm, candidate_vox, candidate_mm, volume, spacing_xyz, snap_radius_mm, gradient_weight):
    if points_mm.shape[0] == 0 or candidate_vox.shape[0] == 0 or snap_radius_mm <= 0:
        return points_mm

    gx, gy, gz = np.gradient(volume.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)
    g_min = float(np.min(grad))
    g_max = float(np.max(grad))
    grad_norm = (grad - g_min) / (g_max - g_min + 1e-6)

    tree = cKDTree(candidate_mm)
    out = points_mm.copy()
    for i, p_mm in enumerate(points_mm):
        ids = tree.query_ball_point(p_mm, r=float(snap_radius_mm))
        if not ids:
            continue
        ids = np.asarray(ids, dtype=np.int32)
        vox = candidate_vox[ids]
        mm = candidate_mm[ids]

        dist = np.linalg.norm(mm - p_mm[None, :], axis=1)
        grad_val = grad_norm[vox[:, 0], vox[:, 1], vox[:, 2]]
        score = float(gradient_weight) * grad_val - dist / (float(snap_radius_mm) + 1e-6)
        best = int(np.argmax(score))
        out[i] = mm[best]

    return out


def _suppress_ring_jump_outliers(points_mm, ratio=3.5, max_iter=3):
    pts = np.asarray(points_mm, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[0] < 5 or pts.shape[1] != 3:
        return pts

    ratio = max(1.5, float(ratio))
    max_iter = max(0, int(max_iter))
    n = int(pts.shape[0])

    for _ in range(max_iter):
        nxt = np.roll(pts, -1, axis=0)
        seg = np.linalg.norm(nxt - pts, axis=1)
        med = float(np.median(seg))
        if not np.isfinite(med) or med <= 1e-6:
            break
        thr = med * ratio

        # Outlier point usually has abnormally long edges on both sides.
        bad = []
        for i in range(n):
            left = float(seg[(i - 1) % n])
            right = float(seg[i])
            if left > thr and right > thr:
                bad.append(i)

        if not bad:
            break

        updated = pts.copy()
        for i in bad:
            p_prev = pts[(i - 1) % n]
            p_next = pts[(i + 1) % n]
            updated[i] = 0.5 * (p_prev + p_next)
        pts = updated

    return pts


def _suppress_axis_projection_outliers(
    points_mm,
    axis,
    window=5,
    mad_k=3.0,
    min_thr_mm=0.6,
    max_iter=2,
):
    pts = np.asarray(points_mm, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[0] < 8 or pts.shape[1] != 3:
        return pts

    axis = np.asarray(axis, dtype=np.float32)
    if axis.shape != (3,):
        return pts
    n_axis = float(np.linalg.norm(axis))
    if n_axis <= 1e-6:
        return pts
    axis = axis / n_axis

    n = int(pts.shape[0])
    w = max(1, int(window))
    max_iter = max(0, int(max_iter))
    mad_k = max(1.0, float(mad_k))
    min_thr = max(0.0, float(min_thr_mm))

    for _ in range(max_iter):
        proj = pts @ axis
        updated = pts.copy()
        changed = 0
        for i in range(n):
            nb_idx = [((i + d) % n) for d in range(-w, w + 1) if d != 0]
            local = proj[np.asarray(nb_idx, dtype=np.int32)]
            med = float(np.median(local))
            mad = float(np.median(np.abs(local - med)))
            robust_sigma = 1.4826 * mad
            thr = max(min_thr, mad_k * robust_sigma)
            delta = float(proj[i] - med)
            if abs(delta) > thr:
                updated[i] = updated[i] - delta * axis
                changed += 1
        pts = updated
        if changed == 0:
            break

    return pts


def rasterize_closed_curve(points_vox, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    if points_vox is None:
        return mask
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
        return mask

    shp = np.asarray(shape, dtype=np.int32)

    def _mark(p):
        q = np.round(p).astype(np.int32)
        if np.all(q >= 0) and np.all(q < shp):
            mask[q[0], q[1], q[2]] = 1

    n_pts = pts.shape[0]
    for i in range(n_pts):
        p0 = pts[i]
        p1 = pts[(i + 1) % n_pts]
        steps = int(np.ceil(np.max(np.abs(p1 - p0)))) + 1
        steps = max(2, steps)
        segment = np.linspace(p0, p1, steps)
        for p in segment:
            _mark(p)

    return mask


def curve_length_mm(points_mm):
    pts = np.asarray(points_mm, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    closed = np.vstack([pts, pts[0]])
    seg = np.diff(closed, axis=0)
    return float(np.linalg.norm(seg, axis=1).sum())


def count_components(mask):
    _, n_comp = label(np.asarray(mask) > 0)
    return int(n_comp)


def extract_abc_curve(A_roi, T_roi, B_roi, spacing_xyz, cfg=None):
    params = dict(DEFAULT_EXTRACT_CFG)
    if cfg:
        params.update(cfg)

    A = np.asarray(A_roi, dtype=np.float32)
    tooth = np.asarray(T_roi) > 0
    bone = np.asarray(B_roi) > 0
    spacing = np.asarray(spacing_xyz, dtype=np.float32)

    empty = np.zeros(tooth.shape, dtype=np.uint8)
    if int(tooth.sum()) < int(params["min_tooth_voxels"]):
        return empty, np.zeros((0, 3), dtype=np.float32), {
            "status": "empty_tooth",
            "curve_length_mm": 0.0,
            "n_components": 0,
            "n_inner_wall_candidates": 0,
            "n_curve_points": 0,
        }

    frame = _compute_local_frame(tooth.astype(np.uint8), spacing)
    if frame is None:
        return empty, np.zeros((0, 3), dtype=np.float32), {
            "status": "frame_failed",
            "curve_length_mm": 0.0,
            "n_components": 0,
            "n_inner_wall_candidates": 0,
            "n_curve_points": 0,
        }
    centroid, axis, u, v = frame

    bone_surface = _surface_mask(bone)
    dist_to_tooth = distance_transform_edt(~tooth, sampling=spacing)
    inner_wall = np.logical_and(bone_surface, dist_to_tooth <= float(params["pdl_threshold_mm"]))
    if int(inner_wall.sum()) == 0:
        inner_wall = bone_surface

    cand_vox = np.array(np.where(inner_wall)).T.astype(np.int32)
    if cand_vox.shape[0] < 8:
        return empty, np.zeros((0, 3), dtype=np.float32), {
            "status": "no_inner_wall",
            "curve_length_mm": 0.0,
            "n_components": 0,
            "n_inner_wall_candidates": int(cand_vox.shape[0]),
            "n_curve_points": 0,
        }
    cand_mm = cand_vox.astype(np.float32) * spacing[None, :]

    sampled_mm = _pick_points_by_angle(
        cand_mm,
        centroid,
        axis,
        u,
        v,
        bins=int(params["angular_bins"]),
    )
    sampled_mm = _smooth_points_ring(sampled_mm, sigma_bins=float(params["smooth_sigma_bins"]))

    if bool(params["gradient_snap"]):
        sampled_mm = _snap_points_to_gradient(
            sampled_mm,
            cand_vox,
            cand_mm,
            A,
            spacing,
            snap_radius_mm=float(params["snap_radius_mm"]),
            gradient_weight=float(params["gradient_weight"]),
        )
    if bool(params.get("jump_outlier_suppression", True)):
        sampled_mm = _suppress_ring_jump_outliers(
            sampled_mm,
            ratio=float(params.get("jump_outlier_ratio", 3.5)),
            max_iter=int(params.get("jump_outlier_max_iter", 3)),
        )
    if bool(params.get("axis_outlier_suppression", True)):
        sampled_mm = _suppress_axis_projection_outliers(
            sampled_mm,
            axis=axis,
            window=int(params.get("axis_outlier_window", 5)),
            mad_k=float(params.get("axis_outlier_mad_k", 3.0)),
            min_thr_mm=float(params.get("axis_outlier_min_thr_mm", 0.6)),
            max_iter=int(params.get("axis_outlier_max_iter", 2)),
        )

    points_vox = sampled_mm / spacing[None, :]
    curve_mask = rasterize_closed_curve(points_vox, tooth.shape)

    if bool(params["keep_lcc"]):
        lbl, n_comp = label(curve_mask > 0)
        if n_comp > 1:
            ids, counts = np.unique(lbl[lbl > 0], return_counts=True)
            keep = int(ids[int(np.argmax(counts))])
            curve_mask = (lbl == keep).astype(np.uint8)

    meta = {
        "status": "ok",
        "curve_length_mm": curve_length_mm(sampled_mm),
        "n_components": count_components(curve_mask),
        "n_inner_wall_candidates": int(cand_vox.shape[0]),
        "n_curve_points": int(points_vox.shape[0]),
        "axis": axis.astype(np.float32).tolist(),
        "centroid_mm": centroid.astype(np.float32).tolist(),
    }
    return curve_mask.astype(np.uint8), points_vox.astype(np.float32), meta
