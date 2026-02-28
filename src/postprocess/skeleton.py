import numpy as np
from monai.transforms import KeepLargestConnectedComponent

try:
    from skimage.morphology import skeletonize_3d as _skeletonize
except Exception:
    from skimage.morphology import skeletonize as _skeletonize

_keep_lcc = KeepLargestConnectedComponent(applied_labels=[1], is_onehot=False, connectivity=1)


def compute_curve_overlap_metrics(mask_a, mask_b):
    a = (np.asarray(mask_a) > 0)
    b = (np.asarray(mask_b) > 0)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    vox_a = int(a.sum())
    vox_b = int(b.sum())
    denom_dice = vox_a + vox_b
    return {
        "equal_voxelwise": bool(np.array_equal(a, b)),
        "intersection_voxels": inter,
        "union_voxels": union,
        "curve_a_voxels": vox_a,
        "curve_b_voxels": vox_b,
        "iou": float(inter / union) if union > 0 else 1.0,
        "dice": float((2.0 * inter) / denom_dice) if denom_dice > 0 else 1.0,
    }


def extract_curve_from_heatmap_peak(mask_prob, tooth_mask=None, peak_threshold=0.999, keep_lcc=False):
    curve = (mask_prob >= float(peak_threshold)).astype(np.uint8)
    if tooth_mask is not None:
        curve = curve * (tooth_mask > 0).astype(np.uint8)
    if curve.sum() == 0:
        return curve
    if keep_lcc:
        curve = _keep_lcc(curve[None, ...])[0]
    return curve.astype(np.uint8)


def extract_curve(mask_prob, tooth_mask, threshold=0.3, keep_lcc=True):
    # Minimal postprocess: threshold -> intersect with tooth -> skeletonize -> largest component
    # TODO: replace with a more robust centerline extractor if needed.
    binary = (mask_prob >= threshold).astype(np.uint8)
    binary = binary * (tooth_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary
    try:
        skel = _skeletonize(binary).astype(np.uint8)
    except Exception:
        skel = binary
    if skel.sum() == 0:
        return skel
    # keep largest component via MONAI when desired.
    if keep_lcc:
        skel = _keep_lcc(skel[None, ...])[0]
    return skel.astype(np.uint8)
