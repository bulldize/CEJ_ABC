import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def vox_to_world(points_vox, affine):
    if points_vox is None or len(points_vox) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(points_vox, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    pts_w = (affine @ pts_h.T).T
    return pts_w[:, :3]


def world_to_vox(points_world, affine):
    if points_world is None or len(points_world) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(points_world, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    inv = np.linalg.inv(affine)
    pts_v = (inv @ pts_h.T).T
    return pts_v[:, :3]


def compute_bbox(mask):
    idx = np.array(np.where(mask > 0))
    if idx.size == 0:
        return None
    mins = idx.min(axis=1)
    maxs = idx.max(axis=1)
    return mins, maxs


def pad_bbox(mins, maxs, pad_vox, shape):
    mins = np.maximum(mins - pad_vox, 0)
    maxs = np.minimum(maxs + pad_vox, np.array(shape) - 1)
    return mins, maxs


def crop_with_bbox(arr, mins, maxs):
    x0, y0, z0 = mins
    x1, y1, z1 = maxs + 1
    return arr[x0:x1, y0:y1, z0:z1], (x0, y0, z0)


def points_full_to_roi(points_vox, roi_origin):
    if points_vox is None or len(points_vox) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points_vox, dtype=np.float32) - np.asarray(roi_origin, dtype=np.float32)


def points_roi_to_full(points_vox, roi_origin):
    if points_vox is None or len(points_vox) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points_vox, dtype=np.float32) + np.asarray(roi_origin, dtype=np.float32)


def tooth_surface(mask):
    if mask.sum() == 0:
        return mask.astype(bool)
    eroded = binary_erosion(mask, iterations=1)
    surface = mask.astype(bool) ^ eroded
    return surface


def distance_to_surface(mask, spacing):
    surface = tooth_surface(mask)
    if surface.sum() == 0:
        return np.full(mask.shape, np.inf, dtype=np.float32)
    dist = distance_transform_edt(~surface, sampling=spacing)
    return dist.astype(np.float32)
