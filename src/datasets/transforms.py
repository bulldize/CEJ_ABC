import numpy as np


def normalize_intensity(volume, clip_percentiles=(1.0, 99.0), mode="zscore"):
    vol = volume.astype(np.float32)
    if clip_percentiles is not None:
        low, high = np.percentile(vol, clip_percentiles)
        vol = np.clip(vol, low, high)
    if mode == "zscore":
        mean = vol.mean()
        std = vol.std() + 1e-6
        vol = (vol - mean) / std
    elif mode == "minmax":
        vmin = vol.min()
        vmax = vol.max() + 1e-6
        vol = (vol - vmin) / (vmax - vmin)
    return vol
