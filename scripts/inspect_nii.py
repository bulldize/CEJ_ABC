#!/usr/bin/env python3
import json
import sys
import numpy as np
import nibabel as nib


def _basic_stats(arr: np.ndarray) -> dict:
    # Use float64 for stable stats
    data = arr.astype(np.float64, copy=False)
    stats = {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }
    # Percentiles for quick debug
    for p in [0.5, 1, 5, 50, 95, 99, 99.5]:
        stats[f"p{p}"] = float(np.percentile(data, p))
    return stats


def _unique_counts(arr: np.ndarray, max_unique: int = 1000) -> dict:
    uniq, counts = np.unique(arr, return_counts=True)
    info = {
        "unique_count": int(uniq.size),
    }
    if uniq.size <= max_unique:
        info["unique_values"] = uniq.tolist()
        info["unique_counts"] = counts.tolist()
    return info


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _inspect_one(path: str) -> dict:
    img = nib.load(path)
    hdr = img.header
    affine = img.affine
    out = {
        "path": path,
        "shape": list(img.shape),
        "dtype": str(img.get_data_dtype()),
        "zooms": list(hdr.get_zooms()[: len(img.shape)]),
        "sform_code": int(hdr["sform_code"]),
        "qform_code": int(hdr["qform_code"]),
        "sform": img.get_sform().tolist(),
        "qform": img.get_qform().tolist(),
        "affine": affine.tolist(),
    }
    data = np.asanyarray(img.dataobj)
    out["stats"] = _basic_stats(data)
    if np.issubdtype(data.dtype, np.integer):
        out["labels"] = _unique_counts(data)
    return out


def main():
    # No-args mode: inspect the default raw case files.
    paths = sys.argv[1:]
    if not paths:
        paths = [
            "data/raw/case0002/A.nii.gz",
            "data/raw/case0002/B.nii.gz",
        ]

    results = [_inspect_one(p) for p in paths]
    # Print as JSON array for easy copy/paste
    print(json.dumps(results, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
