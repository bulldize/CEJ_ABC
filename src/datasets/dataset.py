import glob
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from monai.data import CacheDataset, Dataset as MonaiDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    DivisiblePadd,
    DivisiblePad,
    ClipIntensityPercentilesd,
    NormalizeIntensityd,
    ScaleIntensityd,
    EnsureTyped,
    ConcatItemsd,
    CopyItemsd,
)

from src.datasets.io import load_volume
from src.datasets.transforms import normalize_intensity


def list_tooth_dirs(processed_dir):
    pattern = os.path.join(processed_dir, "*", "tooth_*")
    return sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])


def _build_supervised_items(processed_dir, processed_format="nii.gz"):
    items = []
    for tdir in list_tooth_dirs(processed_dir):
        fmt = processed_format
        items.append({
            "image": os.path.join(tdir, f"A_t.{fmt}"),
            "mask": os.path.join(tdir, f"T_t.{fmt}"),
            "label": os.path.join(tdir, f"H_GT.{fmt}"),
            "tooth_dir": tdir,
        })
    return items


def _build_supervised_transforms(clip_percentiles, norm_mode, use_mask_channel, pad_divisor=None):
    t = [
        LoadImaged(keys=["image", "mask", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "mask", "label"]),
    ]
    if pad_divisor is not None and int(pad_divisor) > 1:
        t.append(DivisiblePadd(keys=["image", "mask", "label"], k=int(pad_divisor), method="end"))
    if clip_percentiles is not None:
        low, high = clip_percentiles
        t.append(ClipIntensityPercentilesd(keys=["image"], lower=low, upper=high))
    if norm_mode == "zscore":
        t.append(NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False))
    elif norm_mode == "minmax":
        t.append(ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0))
    t.append(EnsureTyped(keys=["image", "mask", "label"], dtype=np.float32))
    if use_mask_channel:
        t.append(ConcatItemsd(keys=["image", "mask"], name="x", dim=0))
    else:
        t.append(CopyItemsd(keys=["image"], names=["x"]))
    t.append(CopyItemsd(keys=["label"], names=["y"]))
    return Compose(t)


class ToothDataset(Dataset):
    def __init__(self, processed_dir, processed_format="nii.gz", use_mask_channel=True,
                 clip_percentiles=(1.0, 99.0), norm_mode="zscore", cache_rate=0.0, pad_divisor=None):
        items = _build_supervised_items(processed_dir, processed_format)
        transforms = _build_supervised_transforms(clip_percentiles, norm_mode, use_mask_channel, pad_divisor)
        if cache_rate and cache_rate > 0:
            self.ds = CacheDataset(items, transform=transforms, cache_rate=float(cache_rate), num_workers=0)
        else:
            self.ds = MonaiDataset(items, transform=transforms)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


class ToothDatasetUnsupervised(Dataset):
    def __init__(self, processed_dir, processed_format="nii.gz", use_mask_channel=True,
                 clip_percentiles=(1.0, 99.0), norm_mode="zscore", pad_divisor=None):
        self.processed_dir = processed_dir
        self.processed_format = processed_format
        self.use_mask_channel = use_mask_channel
        self.clip_percentiles = clip_percentiles
        self.norm_mode = norm_mode
        self.tooth_dirs = list_tooth_dirs(processed_dir)
        self.pad_divisor = pad_divisor

    def __len__(self):
        return len(self.tooth_dirs)

    def __getitem__(self, idx):
        tdir = self.tooth_dirs[idx]
        fmt = self.processed_format
        a_path = os.path.join(tdir, f"A_t.{fmt}")
        t_path = os.path.join(tdir, f"T_t.{fmt}")
        roi_meta_path = os.path.join(tdir, "roi_meta.json")

        A, spacing, affine = load_volume(a_path, dtype=np.float32)
        T, _, _ = load_volume(t_path, dtype=np.uint8)

        A = normalize_intensity(A, self.clip_percentiles, self.norm_mode)
        a = torch.from_numpy(A[None, ...].astype(np.float32))
        t = torch.from_numpy(T[None, ...].astype(np.float32))
        if self.pad_divisor is not None and int(self.pad_divisor) > 1:
            padder = DivisiblePad(k=int(self.pad_divisor), method="end")
            a = padder(a)
            t = padder(t)

        roi_meta = {}
        if os.path.exists(roi_meta_path):
            with open(roi_meta_path, "r") as f:
                roi_meta = json.load(f)

        sample = {
            "a": a,
            "t": t,
            "spacing": spacing,
            "affine": affine,
            "roi_meta": roi_meta,
            "tooth_dir": tdir,
        }
        return sample
