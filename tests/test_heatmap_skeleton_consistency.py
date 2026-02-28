import numpy as np

from src.datasets.heatmap import generate_heatmap_from_points
from src.postprocess.skeleton import compute_curve_overlap_metrics, extract_curve_from_heatmap_peak


def _make_closed_ring_points(center=(32.0, 32.0), radius=12.0, z=20.0, num=180):
    t = np.linspace(0.0, 2.0 * np.pi, num, endpoint=False)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    z_arr = np.full_like(x, z)
    return np.stack([x, y, z_arr], axis=1).astype(np.float32)


def _rasterize_polyline(shape, points_vox, close_loop=True):
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
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
        n = max(2, int(np.ceil(np.max(np.abs(p1 - p0)))) + 1)
        for p in np.linspace(p0, p1, n):
            _mark(p)

    if close_loop and pts.shape[0] > 2:
        p0 = pts[-1]
        p1 = pts[0]
        n = max(2, int(np.ceil(np.max(np.abs(p1 - p0)))) + 1)
        for p in np.linspace(p0, p1, n):
            _mark(p)

    return mask


def test_heatmap_peak_skeleton_matches_interpolated_curve():
    shape = (64, 64, 48)
    spacing = (1.0, 1.0, 1.0)
    curve_pts = _make_closed_ring_points()

    h = generate_heatmap_from_points(
        shape=shape,
        points_vox=curve_pts,
        spacing=spacing,
        sigma_mm=1.0,
        connect_points=True,
        close_loop=True,
    )

    tooth_mask = np.ones(shape, dtype=np.uint8)
    c_from_heatmap = extract_curve_from_heatmap_peak(h, tooth_mask, peak_threshold=0.999, keep_lcc=False)
    c_interp = _rasterize_polyline(shape, curve_pts, close_loop=True)

    metrics = compute_curve_overlap_metrics(c_from_heatmap, c_interp)
    assert metrics["iou"] >= 0.95
    assert metrics["dice"] >= 0.97
