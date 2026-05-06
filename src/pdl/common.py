import numpy as np


def map_pulp_to_tooth(labels):
    out = np.asarray(labels).copy()
    pulp_mask = out >= 100
    out[pulp_mask] = out[pulp_mask] % 100
    return out


def get_tooth_labels(labels):
    uniq = np.unique(labels)
    tooth_ids = [int(v) for v in uniq if 10 <= int(v) < 100]
    return sorted(tooth_ids)


def tooth_mask_from_labels(labels):
    arr = np.asarray(labels)
    return ((arr >= 10) & (arr < 100)).astype(np.uint8)


def bone_mask_from_labels(labels):
    arr = np.asarray(labels)
    return ((arr > 0) & (arr < 10)).astype(np.uint8)


def compute_padding_vox(padding_mm, spacing_xyz):
    sp = np.asarray(spacing_xyz, dtype=np.float32)
    sp = np.where(sp <= 0, 1.0, sp)
    pad = int(np.round(float(padding_mm) / float(np.min(sp))))
    pad = max(0, pad)
    return np.array([pad, pad, pad], dtype=np.int32)
