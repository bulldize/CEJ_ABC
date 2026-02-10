import numpy as np
from monai.transforms import KeepLargestConnectedComponent

try:
    from skimage.morphology import skeletonize_3d as _skeletonize
except Exception:
    from skimage.morphology import skeletonize as _skeletonize

_keep_lcc = KeepLargestConnectedComponent(applied_labels=[1], is_onehot=False, connectivity=1)


def extract_curve(mask_prob, tooth_mask, threshold=0.3):
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
    # keep largest component via MONAI
    skel = _keep_lcc(skel[None, ...])[0]
    return skel.astype(np.uint8)
