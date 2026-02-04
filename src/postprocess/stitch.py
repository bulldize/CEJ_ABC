import numpy as np


def stitch_curve_to_full(full_shape, curve_roi, roi_origin):
    full = np.zeros(full_shape, dtype=curve_roi.dtype)
    x0, y0, z0 = roi_origin
    x1, y1, z1 = x0 + curve_roi.shape[0], y0 + curve_roi.shape[1], z0 + curve_roi.shape[2]
    full[x0:x1, y0:y1, z0:z1] = curve_roi
    return full


def build_full_output(B_full, curves_full):
    # Y in {0,1,2}: 1 tooth, 2 curve
    Y = np.zeros_like(B_full, dtype=np.uint8)
    tooth_mask = (B_full >= 10).astype(np.uint8)
    Y[tooth_mask > 0] = 1
    for curve in curves_full:
        Y[curve > 0] = 2
    return Y
