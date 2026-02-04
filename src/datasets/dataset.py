import glob
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.io import load_volume
from src.datasets.transforms import normalize_intensity


def list_tooth_dirs(processed_dir):
    pattern = os.path.join(processed_dir, "*", "tooth_*")
    return sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])


class ToothDataset(Dataset):
    def __init__(self, processed_dir, processed_format="nii.gz", use_mask_channel=True,
                 clip_percentiles=(1.0, 99.0), norm_mode="zscore"):
        self.processed_dir = processed_dir
        self.processed_format = processed_format
        self.use_mask_channel = use_mask_channel
        self.clip_percentiles = clip_percentiles
        self.norm_mode = norm_mode
        self.tooth_dirs = list_tooth_dirs(processed_dir)

    def __len__(self):
        return len(self.tooth_dirs)

    def __getitem__(self, idx):
        tdir = self.tooth_dirs[idx]
        fmt = self.processed_format
        a_path = os.path.join(tdir, f"A_t.{fmt}")
        t_path = os.path.join(tdir, f"T_t.{fmt}")
        h_path = os.path.join(tdir, f"H_GT.{fmt}")
        roi_meta_path = os.path.join(tdir, "roi_meta.json")

        A, spacing, affine = load_volume(a_path, dtype=np.float32)
        T, _, _ = load_volume(t_path, dtype=np.uint8)
        H, _, _ = load_volume(h_path, dtype=np.float32)

        A = normalize_intensity(A, self.clip_percentiles, self.norm_mode)

        if self.use_mask_channel:
            x = np.stack([A, T.astype(np.float32)], axis=0)
        else:
            x = A[None, ...]
        y = H[None, ...]

        with open(roi_meta_path, "r") as f:
            roi_meta = json.load(f)

        sample = {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.float32)),
            "spacing": spacing,
            "affine": affine,
            "roi_meta": roi_meta,
            "tooth_dir": tdir,
        }
        return sample
