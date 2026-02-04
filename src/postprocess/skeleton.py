import numpy as np
from skimage.measure import label

try:
    from skimage.morphology import skeletonize_3d as _skeletonize
except Exception:
    from skimage.morphology import skeletonize as _skeletonize


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
    labels = label(skel, connectivity=1)
    if labels.max() <= 1:
        return skel
    # keep largest component
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return (labels == keep).astype(np.uint8)
