import argparse
import json
import os
import numpy as np
from scipy.ndimage import zoom

from src.datasets.io import load_volume, save_volume
from src.datasets.roi import crop_roi
from src.utils.config import load_config, ensure_dir
from src.utils.log import get_logger


logger = get_logger("preprocess_unsup")


def map_pulp_to_tooth(B):
    B = B.copy()
    mask = B >= 100
    B[mask] = B[mask] % 100
    return B


def get_tooth_labels(B, label_min=None, label_max=None):
    labels = np.unique(B)
    labels = [int(l) for l in labels if int(l) != 0]
    if label_min is not None:
        labels = [l for l in labels if l >= int(label_min)]
    if label_max is not None:
        labels = [l for l in labels if l <= int(label_max)]
    return sorted(labels)


def resample_roi(A_roi, T_roi, spacing, target_spacing):
    spacing = np.asarray(spacing, dtype=np.float32)
    target = np.asarray(target_spacing, dtype=np.float32)
    if np.allclose(spacing, target):
        return A_roi, T_roi, spacing, False, np.array([1.0, 1.0, 1.0])
    scale = spacing / target
    A_rs = zoom(A_roi, scale, order=1)
    T_rs = zoom(T_roi, scale, order=0)
    return A_rs, T_rs, target, True, scale


def _resolve_nifti_path(path):
    if not os.path.isdir(path):
        return path
    base = os.path.basename(path)
    candidate = os.path.join(path, base)
    if os.path.exists(candidate):
        return candidate
    for name in os.listdir(path):
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            return os.path.join(path, name)
    return path


def preprocess_unsup(cfg):
    unsup = cfg["unsup"]

    source_dir = unsup.get("source_dir")
    a_path = unsup.get("source_a_path") or os.path.join(source_dir, unsup["source_a_name"])
    b_path = unsup.get("source_b_path") or os.path.join(source_dir, unsup["source_b_name"])
    case_id = unsup.get("case_id", "unsup_case")

    a_path = _resolve_nifti_path(a_path)
    b_path = _resolve_nifti_path(b_path)

    A, spacing, affine = load_volume(a_path, dtype=np.float32)
    B, spacing_b, affine_b = load_volume(b_path, dtype=np.int16)

    if A.shape != B.shape:
        raise ValueError(f"shape mismatch A{A.shape} vs B{B.shape}")
    if not np.allclose(spacing, spacing_b, atol=1e-3):
        logger.warning("spacing mismatch A%s vs B%s", spacing, spacing_b)
    if not np.allclose(affine, affine_b, atol=1e-3):
        logger.warning("affine mismatch between A and B")

    if unsup.get("map_pulp_to_tooth", True):
        B = map_pulp_to_tooth(B)

    use_label_mask = bool(unsup.get("use_label_mask", True))
    if use_label_mask:
        labels = get_tooth_labels(B, unsup.get("tooth_label_min"), unsup.get("tooth_label_max"))
    else:
        labels = [1]

    fallback_single = bool(unsup.get("fallback_single_tooth", True))
    fallback_mask = None
    if not labels:
        if fallback_single:
            labels = [1]
            fallback_mask = (B > 0).astype(np.uint8) if use_label_mask else np.ones_like(B, dtype=np.uint8)
            logger.warning("no tooth labels found; fallback to single ROI using non-zero mask")
        else:
            logger.warning("no tooth labels found; skip preprocessing")
            return

    padding_mm = cfg["preprocess"]["roi_padding_mm"]
    pad_vox = np.round(np.array(padding_mm) / np.array(spacing)).astype(int)
    min_voxels = int(unsup.get("min_tooth_voxels", 0))

    processed_case_dir = ensure_dir(os.path.join(unsup["processed_dir"], case_id))
    fmt = unsup.get("processed_format", "nii.gz")

    for tooth_id in labels:
        if fallback_mask is not None and tooth_id == 1:
            T_t = fallback_mask
        else:
            T_t = (B == tooth_id).astype(np.uint8) if use_label_mask else np.ones_like(B, dtype=np.uint8)
        if min_voxels > 0 and T_t.sum() < min_voxels:
            logger.info("skip tooth=%s (voxels=%d < %d)", tooth_id, int(T_t.sum()), min_voxels)
            continue

        out = crop_roi(A, T_t, pad_vox)
        if out is None:
            continue
        A_roi, T_roi, origin, mins, maxs = out

        roi_shape_full = A_roi.shape
        spacing_full = spacing
        if cfg["preprocess"]["resample_to_target"]:
            A_roi, T_roi, spacing, resampled, scale = resample_roi(
                A_roi, T_roi, spacing, cfg["preprocess"]["target_spacing_mm"]
            )
        else:
            resampled, scale = False, np.array([1.0, 1.0, 1.0])

        tooth_dir = ensure_dir(os.path.join(processed_case_dir, f"tooth_{tooth_id}"))
        save_volume(os.path.join(tooth_dir, f"A_t.{fmt}"), A_roi, affine=None, spacing=spacing)
        save_volume(os.path.join(tooth_dir, f"T_t.{fmt}"), T_roi.astype(np.uint8), affine=None, spacing=spacing)

        roi_meta = {
            "case_id": case_id,
            "tooth_id": int(tooth_id),
            "roi_origin_in_full": [int(v) for v in origin],
            "roi_shape": [int(v) for v in A_roi.shape],
            "roi_shape_full": [int(v) for v in roi_shape_full],
            "full_shape": [int(v) for v in A.shape],
            "spacing": [float(v) for v in spacing],
            "spacing_full": [float(v) for v in spacing_full],
            "affine": affine.tolist(),
            "coord_type": "voxel",
            "space": "roi",
            "bbox_full": {"min": [int(v) for v in mins], "max": [int(v) for v in maxs]},
            "padding_mm": float(padding_mm),
            "target_spacing_mm": [float(v) for v in cfg["preprocess"]["target_spacing_mm"]],
            "resampled": bool(resampled),
            "resample_scale": [float(v) for v in scale],
            "unsupervised": True,
        }
        with open(os.path.join(tooth_dir, "roi_meta.json"), "w") as f:
            json.dump(roi_meta, f)

        logger.info("processed unsup case=%s tooth=%s", case_id, tooth_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unsup_mae.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preprocess_unsup(cfg)


if __name__ == "__main__":
    main()
