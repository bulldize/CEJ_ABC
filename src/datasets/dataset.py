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
    NormalizeIntensityd,
    ScaleIntensityd,
    EnsureTyped,
    ConcatItemsd,
    CopyItemsd,
    MapTransform,
)

from src.datasets.io import load_volume
from src.datasets.points import get_points_for_tooth, load_points
from src.datasets.transforms import normalize_intensity


class ClipIntensityPercentilesd(MapTransform):
    def __init__(self, keys, lower, upper, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.lower = float(lower)
        self.upper = float(upper)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if torch.is_tensor(img):
                q = torch.as_tensor([self.lower / 100.0, self.upper / 100.0], device=img.device)
                lo, hi = torch.quantile(img.float(), q)
                if torch.isfinite(lo) and torch.isfinite(hi) and hi >= lo:
                    d[key] = torch.clamp(img, min=float(lo.item()), max=float(hi.item()))
                continue
            arr = np.asarray(img)
            lo, hi = np.percentile(arr, [self.lower, self.upper])
            if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
                d[key] = np.clip(arr, lo, hi)
        return d


def list_tooth_dirs(processed_dir):
    pattern = os.path.join(processed_dir, "*", "tooth_*")
    return sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])


def _parse_tooth_id_from_dir(tdir):
    name = os.path.basename(tdir)
    if name.startswith("tooth_"):
        raw = name[len("tooth_"):]
        try:
            return int(raw)
        except ValueError:
            return raw
    return name


def _case_tooth_from_dir(tdir):
    case_id = os.path.basename(os.path.dirname(tdir))
    tooth_id = _parse_tooth_id_from_dir(tdir)
    roi_meta_path = os.path.join(tdir, "roi_meta.json")
    if os.path.exists(roi_meta_path):
        try:
            with open(roi_meta_path, "r") as f:
                roi_meta = json.load(f)
            case_id = roi_meta.get("case_id", case_id)
            tooth_id = roi_meta.get("tooth_id", tooth_id)
        except Exception:
            pass
    return str(case_id), tooth_id


def _count_points(points_path, tooth_id):
    try:
        points_data = load_points(points_path)
        pts = get_points_for_tooth(points_data, tooth_id)
        return int(len(pts))
    except Exception:
        return 0


def _heatmap_max(label_path):
    if not os.path.exists(label_path):
        return None
    try:
        H, _, _ = load_volume(label_path, dtype=np.float32)
    except Exception:
        return None
    if H.size == 0:
        return 0.0
    return float(np.nanmax(H))


def _empty_case_stats():
    return {
        "total_tooth_dirs": 0,
        "used_labeled_tooth_dirs": 0,
        "skipped_unlabeled_tooth_dirs": 0,
        "invalid_tooth_dirs": 0,
        "used_teeth": [],
        "skipped_teeth": [],
        "invalid_teeth": [],
    }


def build_supervised_audit(processed_dir, processed_format="nii.gz"):
    fmt = processed_format
    audit = {
        "processed_dir": str(processed_dir),
        "processed_format": fmt,
        "total_tooth_dirs": 0,
        "used_labeled_tooth_dirs": 0,
        "skipped_unlabeled_tooth_dirs": 0,
        "invalid_tooth_dirs": 0,
        "used_labeled_teeth": [],
        "skipped_unlabeled_teeth": [],
        "invalid_teeth": [],
        "per_case": {},
    }

    for tdir in list_tooth_dirs(processed_dir):
        case_id, tooth_id = _case_tooth_from_dir(tdir)
        tooth_label = str(tooth_id)
        label_path = os.path.join(tdir, f"H_GT.{fmt}")
        points_path = os.path.join(tdir, "points.json")
        n_points = _count_points(points_path, tooth_id)
        h_gt_max = _heatmap_max(label_path)
        has_points = n_points > 0
        has_h_gt_signal = h_gt_max is not None and h_gt_max > 0.0
        has_label_file = os.path.exists(label_path)

        record = {
            "case_id": case_id,
            "tooth_id": tooth_id,
            "tooth_dir": tdir,
            "points_path": points_path,
            "label_path": label_path,
            "n_points": n_points,
            "h_gt_max": h_gt_max,
            "has_points": has_points,
            "has_h_gt_signal": has_h_gt_signal,
            "has_label_file": has_label_file,
        }

        case_stats = audit["per_case"].setdefault(case_id, _empty_case_stats())
        audit["total_tooth_dirs"] += 1
        case_stats["total_tooth_dirs"] += 1

        if has_h_gt_signal:
            record["reason"] = "labeled"
            audit["used_labeled_tooth_dirs"] += 1
            audit["used_labeled_teeth"].append(record)
            case_stats["used_labeled_tooth_dirs"] += 1
            case_stats["used_teeth"].append(tooth_label)
            continue

        if not has_label_file:
            record["reason"] = "missing_h_gt"
        elif has_points:
            record["reason"] = "points_present_empty_h_gt"
            audit["invalid_tooth_dirs"] += 1
            audit["invalid_teeth"].append(record)
            case_stats["invalid_tooth_dirs"] += 1
            case_stats["invalid_teeth"].append(tooth_label)
            continue
        else:
            record["reason"] = "no_points_and_empty_h_gt"
        audit["skipped_unlabeled_tooth_dirs"] += 1
        audit["skipped_unlabeled_teeth"].append(record)
        case_stats["skipped_unlabeled_tooth_dirs"] += 1
        case_stats["skipped_teeth"].append(tooth_label)

    audit["per_case"] = {
        case_id: audit["per_case"][case_id]
        for case_id in sorted(audit["per_case"].keys())
    }
    for stats in audit["per_case"].values():
        stats["used_teeth"] = sorted(stats["used_teeth"])
        stats["skipped_teeth"] = sorted(stats["skipped_teeth"])
        stats["invalid_teeth"] = sorted(stats["invalid_teeth"])
    return audit


def write_supervised_audit(audit, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    audit_path = os.path.join(out_dir, "supervised_dataset_audit.json")
    skipped_json_path = os.path.join(out_dir, "skipped_unlabeled_teeth.json")
    skipped_txt_path = os.path.join(out_dir, "skipped_unlabeled_teeth.txt")
    invalid_json_path = os.path.join(out_dir, "invalid_labeled_teeth.json")
    invalid_txt_path = os.path.join(out_dir, "invalid_labeled_teeth.txt")

    with open(audit_path, "w") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    with open(skipped_json_path, "w") as f:
        json.dump(audit["skipped_unlabeled_teeth"], f, ensure_ascii=False, indent=2)
    with open(skipped_txt_path, "w") as f:
        for rec in audit["skipped_unlabeled_teeth"]:
            f.write(
                "\t".join(
                    [
                        str(rec["case_id"]),
                        str(rec["tooth_id"]),
                        str(rec["n_points"]),
                        str(rec["h_gt_max"]),
                        str(rec["reason"]),
                        str(rec["tooth_dir"]),
                    ]
                )
                + "\n"
            )
    with open(invalid_json_path, "w") as f:
        json.dump(audit.get("invalid_teeth", []), f, ensure_ascii=False, indent=2)
    with open(invalid_txt_path, "w") as f:
        for rec in audit.get("invalid_teeth", []):
            f.write(
                "\t".join(
                    [
                        str(rec["case_id"]),
                        str(rec["tooth_id"]),
                        str(rec["n_points"]),
                        str(rec["h_gt_max"]),
                        str(rec["reason"]),
                        str(rec["tooth_dir"]),
                    ]
                )
                + "\n"
            )
    return {
        "audit": audit_path,
        "skipped_json": skipped_json_path,
        "skipped_txt": skipped_txt_path,
        "invalid_json": invalid_json_path,
        "invalid_txt": invalid_txt_path,
    }


def _build_supervised_items(processed_dir, processed_format="nii.gz", audit=None):
    if audit is None:
        audit = build_supervised_audit(processed_dir, processed_format)
    fmt = processed_format
    items = []
    for rec in audit["used_labeled_teeth"]:
        tdir = rec["tooth_dir"]
        items.append({
            "image": os.path.join(tdir, f"A_t.{fmt}"),
            "mask": os.path.join(tdir, f"T_t.{fmt}"),
            "label": os.path.join(tdir, f"H_GT.{fmt}"),
            "tooth_dir": tdir,
            "case_id": rec["case_id"],
            "tooth_id": rec["tooth_id"],
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
        self.audit = build_supervised_audit(processed_dir, processed_format)
        items = _build_supervised_items(processed_dir, processed_format, audit=self.audit)
        self.items = items
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
