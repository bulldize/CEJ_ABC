import numpy as np

from src.export_pseudo_gt_full import _dilate_mask_mm


def test_zero_tube_radius_keeps_thin_curve_mask():
    mask = np.zeros((24, 24, 24), dtype=np.uint8)
    for i in range(4, 20):
        mask[i, i, i] = 1

    out = _dilate_mask_mm(mask, spacing_xyz=(0.5, 0.5, 0.5), radius_mm=0.0)
    assert np.array_equal(out.astype(np.uint8), mask.astype(np.uint8))


def test_negative_tube_radius_keeps_thin_curve_mask():
    mask = np.zeros((24, 24, 24), dtype=np.uint8)
    mask[12, 12, 10:14] = 1

    out = _dilate_mask_mm(mask, spacing_xyz=(1.0, 1.0, 1.0), radius_mm=-1.0)
    assert np.array_equal(out.astype(np.uint8), mask.astype(np.uint8))
