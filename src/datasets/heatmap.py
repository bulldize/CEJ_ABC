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


def fit_curve_and_sample(points_vox, spacing, step_mm):
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    spacing = np.asarray(spacing, dtype=np.float32)
    pts_mm = pts * spacing
    # sort by z (mm) for stable ordering
    order = np.argsort(pts_mm[:, 2])
    pts_mm = pts_mm[order]

    if pts_mm.shape[0] >= 4:
        try:
            tck, _ = splprep(pts_mm.T, s=0)
            # estimate length using linear segments
            dense_mm = _linear_resample(pts_mm, step_mm)
            n = max(2, dense_mm.shape[0])
            u = np.linspace(0.0, 1.0, n)
            pts_dense = np.array(splev(u, tck)).T
        except Exception:
            pts_dense = _linear_resample(pts_mm, step_mm)
    else:
        pts_dense = _linear_resample(pts_mm, step_mm)

    pts_dense_vox = pts_dense / spacing
    return pts_dense_vox.astype(np.float32)


def generate_heatmap_from_points(shape, points_vox, spacing, sigma_mm):
    if points_vox is None or len(points_vox) == 0:
        return np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.asarray(points_vox, dtype=np.float32)
    pts_round = np.round(pts).astype(np.int64)
    for p in pts_round:
        x, y, z = p.tolist()
        if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
            mask[x, y, z] = 1
    dist = distance_transform_edt(mask == 0, sampling=spacing)
    h = np.exp(-(dist ** 2) / (2.0 * (sigma_mm ** 2)))
    return h.astype(np.float32)
