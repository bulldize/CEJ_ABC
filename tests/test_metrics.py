import numpy as np
from scipy.spatial import cKDTree


def compute_distances(points_vox, curve_mask, spacing):
    curve_pts = np.array(np.where(curve_mask > 0)).T.astype(np.float32)
    spacing = np.asarray(spacing, dtype=np.float32)
    curve_mm = curve_pts * spacing
    pts_mm = np.asarray(points_vox, dtype=np.float32) * spacing
    tree = cKDTree(curve_mm)
    d, _ = tree.query(pts_mm, k=1)
    return d.astype(np.float32)


def test_point_to_curve_zero():
    shape = (10, 10, 10)
    curve = np.zeros(shape, dtype=np.uint8)
    for i in range(2, 8):
        curve[i, i, i] = 1
    pts = np.array([[2, 2, 2], [4, 4, 4], [7, 7, 7]], dtype=np.float32)
    d = compute_distances(pts, curve, spacing=(1.0, 1.0, 1.0))
    assert np.allclose(d, 0.0)
