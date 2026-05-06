import argparse
import json
import os

import numpy as np
from scipy.ndimage import zoom

from src.datasets.io import load_volume, save_volume
from src.datasets.raw_cases import collect_raw_cases
from src.datasets.roi import crop_roi
from src.pdl.common import (
    bone_mask_from_labels,
    compute_padding_vox,
    get_tooth_labels,
    map_pulp_to_tooth,
)
from src.utils.config import ensure_dir, load_config
from src.utils.log import get_logger

logger = get_logger("pdl_preprocess")

DEFAULT_CONFIG = "/Users/bulldize/Desktop/华西口腔_CEJ_ABC/runtime/pdl/configs/pdl_runtime.yaml"


def _resample_roi(A_roi, T_roi, B_roi, spacing_xyz, target_spacing_xyz):
    spacing = np.asarray(spacing_xyz, dtype=np.float32)
    target = np.asarray(target_spacing_xyz, dtype=np.float32)
    if np.allclose(spacing, target):
        return A_roi, T_roi, B_roi, spacing, False, np.array([1.0, 1.0, 1.0], dtype=np.float32)

    scale = spacing / target
    A_rs = zoom(A_roi, scale, order=1)
    T_rs = zoom(T_roi, scale, order=0)
    B_rs = zoom(B_roi, scale, order=0)
    return A_rs, T_rs, B_rs, target, True, scale


def preprocess_case(case_rec, cfg):
    case_id = case_rec["case_id"]
    raw_a = case_rec["a_path"]
    raw_b = case_rec["b_path"]
    raw_meta = case_rec.get("meta_path", None)
    if raw_meta and not os.path.exists(raw_meta):
        raw_meta = None

    A, spacing, affine = load_volume(raw_a, meta_path=raw_meta, dtype=np.float32)
    B_raw, spacing_b, affine_b = load_volume(raw_b, meta_path=raw_meta, dtype=np.int16)
    if A.shape != B_raw.shape:
        logger.error("shape mismatch A%s vs B%s for case=%s", A.shape, B_raw.shape, case_id)
        return
    if not np.allclose(spacing, spacing_b, atol=1e-3):
        logger.warning("spacing mismatch A%s vs B%s for case=%s", spacing, spacing_b, case_id)
    if not np.allclose(affine, affine_b, atol=1e-3):
        logger.warning("affine mismatch for case=%s", case_id)

    B = map_pulp_to_tooth(B_raw)
    bone_mask = bone_mask_from_labels(B)
    tooth_labels = get_tooth_labels(B)

    if not tooth_labels:
        logger.warning("no tooth labels found for case=%s", case_id)
        return

    pad_vox = compute_padding_vox(cfg["pdl_preprocess"]["roi_padding_mm"], spacing)
    processed_case_dir = ensure_dir(os.path.join(cfg["data"]["processed_dir"], case_id))
    fmt = cfg["data"]["processed_format"]

    for tooth_id in tooth_labels:
        tooth_mask = (B == tooth_id).astype(np.uint8)
        out = crop_roi(A, tooth_mask, pad_vox)
        if out is None:
            continue
        A_roi, T_roi, origin, mins, maxs = out
        x0, y0, z0 = [int(v) for v in mins]
        x1, y1, z1 = [int(v) + 1 for v in maxs]
        B_roi = bone_mask[x0:x1, y0:y1, z0:z1].astype(np.uint8)

        roi_shape_full = A_roi.shape
        spacing_full = np.asarray(spacing, dtype=np.float32)
        if bool(cfg["pdl_preprocess"].get("resample_to_target", False)):
            A_roi, T_roi, B_roi, spacing_tooth, resampled, scale = _resample_roi(
                A_roi,
                T_roi,
                B_roi,
                spacing_full,
                cfg["pdl_preprocess"]["target_spacing_mm"],
            )
        else:
            spacing_tooth = spacing_full
            resampled = False
            scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        tooth_dir = ensure_dir(os.path.join(processed_case_dir, f"tooth_{tooth_id}"))
        save_volume(os.path.join(tooth_dir, f"A_t.{fmt}"), A_roi, spacing=spacing_tooth)
        save_volume(os.path.join(tooth_dir, f"T_t.{fmt}"), T_roi.astype(np.uint8), spacing=spacing_tooth)
        save_volume(os.path.join(tooth_dir, f"B_t.{fmt}"), B_roi.astype(np.uint8), spacing=spacing_tooth)

        roi_meta = {
            "case_id": case_id,
            "tooth_id": int(tooth_id),
            "roi_origin_in_full": [int(v) for v in origin],
            "roi_shape": [int(v) for v in A_roi.shape],
            "roi_shape_full": [int(v) for v in roi_shape_full],
            "full_shape": [int(v) for v in A.shape],
            "spacing": [float(v) for v in spacing_tooth],
            "spacing_full": [float(v) for v in spacing_full],
            "affine": affine.tolist(),
            "bbox_full": {
                "min": [int(v) for v in mins],
                "max": [int(v) for v in maxs],
            },
            "padding_mm": float(cfg["pdl_preprocess"]["roi_padding_mm"]),
            "target_spacing_mm": [float(v) for v in cfg["pdl_preprocess"]["target_spacing_mm"]],
            "resampled": bool(resampled),
            "resample_scale": [float(v) for v in scale],
        }
        with open(os.path.join(tooth_dir, "roi_meta.json"), "w", encoding="utf-8") as f:
            json.dump(roi_meta, f, ensure_ascii=False, indent=2)

        logger.info("pdl preprocess case=%s tooth=%s", case_id, tooth_id)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cases = collect_raw_cases(cfg["data"])
    if not cases:
        logger.warning("no cases found for data.raw_dir=%s", cfg["data"]["raw_dir"])
        return

    for case_rec in cases:
        preprocess_case(case_rec, cfg)


if __name__ == "__main__":
    main()
