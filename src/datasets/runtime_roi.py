import json
import os

import numpy as np
from scipy.ndimage import zoom

from src.datasets.io import load_volume, save_volume
from src.datasets.raw_cases import collect_raw_cases
from src.datasets.roi import crop_roi
from src.utils.config import ensure_dir
from src.utils.log import get_logger

logger = get_logger("runtime_roi")


def map_pulp_to_tooth(seg):
    out = np.asarray(seg).copy()
    mask = out >= 100
    out[mask] = out[mask] % 100
    return out


def get_tooth_labels(seg):
    labels = np.unique(seg)
    labels = [int(v) for v in labels if 10 <= int(v) < 100]
    return sorted(labels)


def resample_roi_for_inference(a_roi, t_roi, spacing, target_spacing):
    spacing = np.asarray(spacing, dtype=np.float32)
    target = np.asarray(target_spacing, dtype=np.float32)
    if np.allclose(spacing, target):
        return a_roi, t_roi, spacing, False, np.array([1.0, 1.0, 1.0], dtype=np.float32)

    scale = spacing / target
    a_rs = zoom(a_roi, scale, order=1)
    t_rs = zoom(t_roi, scale, order=0)
    return a_rs, t_rs, target, True, scale


def preprocess_case_for_inference(case_rec, cfg):
    case_id = case_rec["case_id"]
    raw_a = case_rec["a_path"]
    raw_b = case_rec["b_path"]
    raw_meta = case_rec.get("meta_path", None)
    if raw_meta and not os.path.exists(raw_meta):
        raw_meta = None

    a_full, spacing, affine = load_volume(raw_a, meta_path=raw_meta, dtype=np.float32)
    b_full, spacing_b, affine_b = load_volume(raw_b, meta_path=raw_meta, dtype=np.int16)
    if a_full.shape != b_full.shape:
        raise ValueError(f"shape mismatch A{a_full.shape} vs B{b_full.shape} for case={case_id}")
    if not np.allclose(spacing, spacing_b, atol=1e-3):
        logger.warning("spacing mismatch A%s vs B%s for case=%s", spacing, spacing_b, case_id)
    if affine is not None and affine_b is not None and not np.allclose(affine, affine_b, atol=1e-3):
        logger.warning("affine mismatch for case=%s", case_id)

    b_full = map_pulp_to_tooth(b_full)
    tooth_labels = get_tooth_labels(b_full)
    if not tooth_labels:
        logger.warning("no tooth labels found for case=%s", case_id)
        return []

    preprocess_cfg = cfg.get("preprocess", {})
    padding_mm = float(preprocess_cfg.get("roi_padding_mm", 8.0))
    pad_vox = np.round(np.asarray(padding_mm, dtype=np.float32) / np.asarray(spacing, dtype=np.float32)).astype(int)
    processed_case_dir = ensure_dir(os.path.join(cfg["data"]["processed_dir"], case_id))
    fmt = cfg["data"].get("processed_format", "nii.gz")
    written = []

    for tooth_id in tooth_labels:
        tooth_mask = (b_full == tooth_id).astype(np.uint8)
        cropped = crop_roi(a_full, tooth_mask, pad_vox)
        if cropped is None:
            continue

        a_roi, t_roi, origin, mins, maxs = cropped
        roi_shape_full = a_roi.shape
        spacing_full = spacing
        if bool(preprocess_cfg.get("resample_to_target", False)):
            a_roi, t_roi, spacing_roi, resampled, scale = resample_roi_for_inference(
                a_roi,
                t_roi,
                spacing,
                preprocess_cfg.get("target_spacing_mm", spacing),
            )
        else:
            spacing_roi = spacing
            resampled = False
            scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        tooth_dir = ensure_dir(os.path.join(processed_case_dir, f"tooth_{tooth_id}"))
        save_volume(os.path.join(tooth_dir, f"A_t.{fmt}"), a_roi, affine=None, spacing=spacing_roi)
        save_volume(os.path.join(tooth_dir, f"T_t.{fmt}"), t_roi.astype(np.uint8), affine=None, spacing=spacing_roi)

        roi_meta = {
            "case_id": case_id,
            "tooth_id": tooth_id,
            "roi_origin_in_full": [int(v) for v in origin],
            "roi_shape": [int(v) for v in a_roi.shape],
            "roi_shape_full": [int(v) for v in roi_shape_full],
            "full_shape": [int(v) for v in a_full.shape],
            "spacing": [float(v) for v in spacing_roi],
            "spacing_full": [float(v) for v in spacing_full],
            "affine": affine.tolist() if affine is not None else None,
            "coord_type": "voxel",
            "space": "roi",
            "bbox_full": {"min": [int(v) for v in mins], "max": [int(v) for v in maxs]},
            "padding_mm": float(padding_mm),
            "target_spacing_mm": [
                float(v) for v in preprocess_cfg.get("target_spacing_mm", spacing_roi)
            ],
            "resampled": bool(resampled),
            "resample_scale": [float(v) for v in scale],
            "has_points": False,
            "runtime_inference_only": True,
        }
        with open(os.path.join(tooth_dir, "roi_meta.json"), "w") as f:
            json.dump(roi_meta, f)

        written.append(tooth_dir)
        logger.info("prepared runtime ROI case=%s tooth=%s", case_id, tooth_id)

    return written


def preprocess_raw_cases_for_inference(cfg):
    cases = collect_raw_cases(cfg["data"])
    if not cases:
        logger.warning("no raw cases found for data.raw_dir=%s", cfg["data"].get("raw_dir"))
        return []

    written = []
    for case_rec in cases:
        written.extend(preprocess_case_for_inference(case_rec, cfg))
    return written
