import numpy as np
from monai.transforms import NormalizeIntensity, ScaleIntensity
from monai.utils import convert_to_numpy


def normalize_intensity(volume, clip_percentiles=(1.0, 99.0), mode="zscore"):
    vol = volume.astype(np.float32)
    if clip_percentiles is not None:
        low, high = clip_percentiles
        lo = np.percentile(vol, float(low))
        hi = np.percentile(vol, float(high))
        vol = np.clip(vol, lo, hi)
    if mode == "zscore":
        vol = NormalizeIntensity(nonzero=False, channel_wise=False)(vol)
    elif mode == "minmax":
        vol = ScaleIntensity(minv=0.0, maxv=1.0)(vol)
    vol = convert_to_numpy(vol)
    return vol.astype(np.float32)
