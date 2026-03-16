import argparse
import json
import os
import shutil

import numpy as np
from scipy.ndimage import label

from src.datasets.io import load_volume, save_volume
from src.utils.config import ensure_dir, load_config
from src.utils.log import get_logger

logger = get_logger("abc_export")

MINIMAL_EXPORT_FILES = {
    "ABC_medical_compare_full.nii.gz",
    "ABC_model_compare_full.nii.gz",
    "export_meta.json",
}


def _build_three_segment_compare(tooth_mask, segment2_mask, segment3_mask):
    tooth = np.asarray(tooth_mask) > 0
    seg2 = np.asarray(segment2_mask) > 0
    seg3 = np.asarray(segment3_mask) > 0

    out = np.zeros(tooth.shape, dtype=np.uint8)
    out[tooth] = 1
    out[seg2] = 2
    out[seg3] = 3

    vals, cnts = np.unique(out, return_counts=True)
    return out, {
        "label_counts": {int(v): int(c) for v, c in zip(vals, cnts)},
        "segment2_segment3_overlap_voxels": int(np.logical_and(seg2, seg3).sum()),
    }


def _curve_summary(curve_mask, spacing_xyz):
    curve = (np.asarray(curve_mask) > 0).astype(np.uint8)
    _, n_comp = label(curve)
    voxel_count = int(curve.sum())
    spacing = np.asarray(spacing_xyz, dtype=np.float32)
    approx_length = float(voxel_count * np.min(spacing))
    return {
        "curve_voxels": voxel_count,
        "curve_components": int(n_comp),
        "approx_length_mm": approx_length,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/abc_default.yaml")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output folder for ABC compare NIfTI. Default: outputs/abc/pseudo_gt_review_nifti",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    infer_root = os.path.join(cfg["data"]["output_dir"], "infer")
    out_root = args.out_dir or os.path.join(cfg["data"]["output_dir"], "pseudo_gt_review_nifti")
    out_root = ensure_dir(out_root)

    if not os.path.isdir(infer_root):
        logger.warning("infer root not found: %s", infer_root)
        return

    case_ids = sorted([d for d in os.listdir(infer_root) if os.path.isdir(os.path.join(infer_root, d))])
    if not case_ids:
        logger.warning("no case folders under %s", infer_root)
        return

    keep_minimal = bool(cfg.get("abc_export", {}).get("keep_minimal_files", True))

    for case_id in case_ids:
        y_path = os.path.join(infer_root, case_id, "Y_ABC_pred.nii.gz")
        if not os.path.exists(y_path):
            continue

        Y, spacing, affine = load_volume(y_path, dtype=np.uint8)
        tooth = (Y == 1).astype(np.uint8)
        abc_curve = (Y == 2).astype(np.uint8)
        cej_placeholder = np.zeros_like(abc_curve, dtype=np.uint8)

        medical_compare, medical_stats = _build_three_segment_compare(
            tooth,
            cej_placeholder,
            abc_curve,
        )
        model_compare, model_stats = _build_three_segment_compare(
            tooth,
            cej_placeholder,
            abc_curve,
        )

        out_case_dir = ensure_dir(os.path.join(out_root, case_id))
        if keep_minimal:
            for stale_name in os.listdir(out_case_dir):
                if stale_name in MINIMAL_EXPORT_FILES:
                    continue
                stale_path = os.path.join(out_case_dir, stale_name)
                if os.path.isdir(stale_path):
                    shutil.rmtree(stale_path)
                else:
                    os.remove(stale_path)

        save_volume(
            os.path.join(out_case_dir, "ABC_medical_compare_full.nii.gz"),
            medical_compare,
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )
        save_volume(
            os.path.join(out_case_dir, "ABC_model_compare_full.nii.gz"),
            model_compare,
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )

        meta = {
            "case_id": case_id,
            "source_y_abc_pred": y_path,
            "compare_label_definition": {
                "0": "background",
                "1": "tooth_body",
                "2": "reserved_channel",
                "3": "abc_curve",
            },
            "reserved_channel_used": False,
            "curve_summary": _curve_summary(abc_curve, spacing),
            "compare_files": {
                "ABC_medical_compare_full.nii.gz": {
                    "segment_1": "tooth_body",
                    "segment_2": "reserved_channel(empty)",
                    "segment_3": "abc_curve_rule_extracted",
                    "voxel_stats": medical_stats,
                },
                "ABC_model_compare_full.nii.gz": {
                    "segment_1": "tooth_body",
                    "segment_2": "reserved_channel(empty)",
                    "segment_3": "abc_curve_pred",
                    "voxel_stats": model_stats,
                },
            },
        }
        with open(os.path.join(out_case_dir, "export_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("abc export case=%s -> %s", case_id, out_case_dir)

    logger.info("abc export done. output root: %s", out_root)


if __name__ == "__main__":
    main()
