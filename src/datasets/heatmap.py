import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt


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
    # Best-fit plane by PCA/SVD, then sort by polar angle in that plane.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T  # (3,2), top-2 principal directions
    proj = centered @ basis
    ang = np.arctan2(proj[:, 1], proj[:, 0])
    order = np.argsort(ang)
    pts_sorted = pts[order]
    # Rotate to start from the minimum z (stable start, easier QA)
    start = int(np.argmin(pts_sorted[:, 2]))
    pts_sorted = np.roll(pts_sorted, -start, axis=0)
    return pts_sorted


def fit_curve_and_sample(points_vox, spacing, step_mm, closed=True, smooth=0.0):
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    spacing = np.asarray(spacing, dtype=np.float32)
    pts_mm = pts * spacing

    # Remove exact duplicates (after rounding) to reduce spline instability.
    pts_round = np.round(pts_mm, 4)
    _, uniq_idx = np.unique(pts_round, axis=0, return_index=True)
    pts_mm = pts_mm[np.sort(uniq_idx)]
    if pts_mm.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if closed:
        pts_mm = _sort_points_by_plane_angle(pts_mm)
    else:
        # For open curves, keep z-order as fallback.
        order = np.argsort(pts_mm[:, 2])
        pts_mm = pts_mm[order]

    if pts_mm.shape[0] >= 4:
        try:
            tck, _ = splprep(pts_mm.T, s=float(smooth), per=bool(closed))
            # estimate target count from polyline length
            dense_mm = _linear_resample_closed(pts_mm, step_mm) if closed else _linear_resample(pts_mm, step_mm)
            n = max(8 if closed else 2, dense_mm.shape[0])
            u = np.linspace(0.0, 1.0, n, endpoint=not closed)
            pts_dense = np.array(splev(u, tck)).T
        except Exception:
            pts_dense = _linear_resample_closed(pts_mm, step_mm) if closed else _linear_resample(pts_mm, step_mm)
    else:
        pts_dense = _linear_resample_closed(pts_mm, step_mm) if closed else _linear_resample(pts_mm, step_mm)

    pts_dense_vox = pts_dense / spacing
    return pts_dense_vox.astype(np.float32)


def _rasterize_polyline_mask(shape, points_vox, close_loop=True):
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return mask

    shape_arr = np.asarray(shape, dtype=np.int32)

    def _mark(p):
        q = np.round(p).astype(np.int32)
        if np.all(q >= 0) and np.all(q < shape_arr):
            mask[q[0], q[1], q[2]] = 1

    _mark(pts[0])
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        delta = np.abs(p1 - p0)
        n = int(np.ceil(np.max(delta))) + 1
        n = max(2, n)
        for p in np.linspace(p0, p1, n):
            _mark(p)

    if close_loop and pts.shape[0] > 2:
        p0 = pts[-1]
        p1 = pts[0]
        delta = np.abs(p1 - p0)
        n = int(np.ceil(np.max(delta))) + 1
        n = max(2, n)
        for p in np.linspace(p0, p1, n):
            _mark(p)

    return mask


def generate_heatmap_from_points(shape, points_vox, spacing, sigma_mm, connect_points=True, close_loop=True):
    if points_vox is None or len(points_vox) == 0:
        return np.zeros(shape, dtype=np.float32)
    pts = np.asarray(points_vox, dtype=np.float32)
    if connect_points:
        mask = _rasterize_polyline_mask(shape, pts, close_loop=bool(close_loop))
    else:
        mask = np.zeros(shape, dtype=np.uint8)
        pts_round = np.round(pts).astype(np.int64)
        for p in pts_round:
            x, y, z = p.tolist()
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                mask[x, y, z] = 1
    if mask.sum() == 0:
        return np.zeros(shape, dtype=np.float32)
    dist = distance_transform_edt(mask == 0, sampling=spacing)
    h = np.exp(-(dist ** 2) / (2.0 * (sigma_mm ** 2)))
    return h.astype(np.float32)
