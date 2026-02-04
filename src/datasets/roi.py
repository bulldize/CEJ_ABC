import numpy as np
from src.utils.geometry import compute_bbox, pad_bbox, crop_with_bbox


def crop_roi(volume, mask, padding_vox):
    bbox = compute_bbox(mask)
    if bbox is None:
        return None
    mins, maxs = bbox
    mins, maxs = pad_bbox(mins, maxs, padding_vox, volume.shape)
    vol_roi, origin = crop_with_bbox(volume, mins, maxs)
    mask_roi, _ = crop_with_bbox(mask, mins, maxs)
    return vol_roi, mask_roi, origin, mins, maxs


def stitch_back(full_shape, roi, origin):
    full = np.zeros(full_shape, dtype=roi.dtype)
    x0, y0, z0 = origin
    x1, y1, z1 = x0 + roi.shape[0], y0 + roi.shape[1], z0 + roi.shape[2]
    full[x0:x1, y0:y1, z0:z1] = roi
    return full
