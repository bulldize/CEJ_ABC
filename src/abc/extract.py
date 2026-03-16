import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.ndimage import zoom

from src.abc.algorithm import extract_abc_curve
from src.abc.common import map_pulp_to_tooth, tooth_mask_from_labels
from src.datasets.io import load_volume, save_volume
from src.datasets.raw_cases import collect_raw_case_map
from src.utils.config import ensure_dir, load_config
from src.utils.log import get_logger

logger = get_logger("abc_extract")


def _resize_to_shape(vol, shape, order):
    out = vol
    if out.shape != tuple(shape):
        scale = [shape[i] / out.shape[i] for i in range(3)]
        out = zoom(out, scale, order=order)
        out = out[: shape[0], : shape[1], : shape[2]]
        pad = [shape[i] - out.shape[i] for i in range(3)]
        if any(v > 0 for v in pad):
            out = np.pad(out, [(0, pad[0]), (0, pad[1]), (0, pad[2])], mode="constant")
    return out


def _stitch_patch_binary_or(full, patch, origin_xyz):
    x0, y0, z0 = [int(v) for v in origin_xyz]
    x1 = min(full.shape[0], x0 + patch.shape[0])
    y1 = min(full.shape[1], y0 + patch.shape[1])
    z1 = min(full.shape[2], z0 + patch.shape[2])
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return

    px, py, pz = x1 - x0, y1 - y0, z1 - z0
    full[x0:x1, y0:y1, z0:z1] = np.maximum(
        full[x0:x1, y0:y1, z0:z1],
        patch[:px, :py, :pz],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/abc_default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fmt = cfg["data"]["processed_format"]

    processed_dir = cfg["data"]["processed_dir"]
    tooth_dirs = sorted(glob.glob(os.path.join(processed_dir, "*", "tooth_*")))
    if not tooth_dirs:
        logger.warning("no ABC processed teeth found under %s", processed_dir)
        return

    infer_root = ensure_dir(os.path.join(cfg["data"]["output_dir"], "infer"))
    case_curves = {}
    metrics_rows = []

    for tooth_dir in tooth_dirs:
        roi_meta_path = os.path.join(tooth_dir, "roi_meta.json")
        if not os.path.exists(roi_meta_path):
            continue
        with open(roi_meta_path, "r", encoding="utf-8") as f:
            roi_meta = json.load(f)

        case_id = roi_meta["case_id"]
        tooth_id = int(roi_meta["tooth_id"])

        A_path = os.path.join(tooth_dir, f"A_t.{fmt}")
        T_path = os.path.join(tooth_dir, f"T_t.{fmt}")
        B_path = os.path.join(tooth_dir, f"B_t.{fmt}")
        if not (os.path.exists(A_path) and os.path.exists(T_path) and os.path.exists(B_path)):
            logger.warning("skip missing ROI files: %s", tooth_dir)
            continue

        A_roi, spacing, affine = load_volume(A_path, dtype=np.float32)
        T_roi, _, _ = load_volume(T_path, dtype=np.uint8)
        B_roi, _, _ = load_volume(B_path, dtype=np.uint8)

        curve_roi, points_vox, meta = extract_abc_curve(
            A_roi,
            T_roi,
            B_roi,
            spacing_xyz=spacing,
            cfg=cfg.get("abc_extract", {}),
            tooth_id=tooth_id,
        )

        out_tooth_dir = ensure_dir(os.path.join(infer_root, case_id, f"tooth_{tooth_id}"))
        save_volume(
            os.path.join(out_tooth_dir, "C_ABC.nii.gz"),
            curve_roi.astype(np.uint8),
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )
        np.save(os.path.join(out_tooth_dir, "abc_curve_points_vox.npy"), points_vox.astype(np.float32))

        tooth_meta = dict(meta)
        tooth_meta.update(
            {
                "case_id": case_id,
                "tooth_id": tooth_id,
                "roi_origin_in_full": roi_meta["roi_origin_in_full"],
                "resampled": bool(roi_meta.get("resampled", False)),
            }
        )
        with open(os.path.join(out_tooth_dir, "abc_meta.json"), "w", encoding="utf-8") as f:
            json.dump(tooth_meta, f, ensure_ascii=False, indent=2)

        curve_for_stitch = curve_roi.astype(np.uint8)
        if roi_meta.get("resampled", False):
            scale = np.array(roi_meta.get("resample_scale", [1.0, 1.0, 1.0]), dtype=np.float32)
            inv_scale = 1.0 / scale
            curve_for_stitch = zoom(curve_for_stitch, inv_scale, order=0)
            curve_for_stitch = _resize_to_shape(curve_for_stitch, roi_meta["roi_shape_full"], order=0)
            curve_for_stitch = (curve_for_stitch > 0).astype(np.uint8)

        case_curves.setdefault(case_id, []).append(
            {
                "tooth_id": tooth_id,
                "origin": roi_meta["roi_origin_in_full"],
                "curve": curve_for_stitch,
            }
        )

        metrics_rows.append(
            {
                "case_id": case_id,
                "tooth_id": tooth_id,
                "status": meta.get("status", "unknown"),
                "curve_length_mm": float(meta.get("curve_length_mm", 0.0)),
                "n_components": int(meta.get("n_components", 0)),
                "n_inner_wall_candidates": int(meta.get("n_inner_wall_candidates", 0)),
                "n_curve_points": int(meta.get("n_curve_points", 0)),
            }
        )
        logger.info("abc extract case=%s tooth=%s status=%s", case_id, tooth_id, meta.get("status", "unknown"))

    raw_case_map = collect_raw_case_map(cfg["data"])
    for case_id, items in case_curves.items():
        case_rec = raw_case_map.get(case_id, None)
        if case_rec is None:
            logger.warning("raw case record not found for case=%s; skip full stitch", case_id)
            continue
        b_path = case_rec["b_path"]
        meta_path = case_rec.get("meta_path", None)
        if meta_path and not os.path.exists(meta_path):
            meta_path = None
        if not os.path.exists(b_path):
            logger.warning("raw label missing for case=%s", case_id)
            continue

        B_full, spacing_full, affine_full = load_volume(b_path, meta_path=meta_path, dtype=np.int16)
        B_full = map_pulp_to_tooth(B_full)
        tooth_full = tooth_mask_from_labels(B_full)

        curve_full = np.zeros(B_full.shape, dtype=np.uint8)
        for item in items:
            _stitch_patch_binary_or(curve_full, item["curve"], item["origin"])

        Y_abc = np.zeros(B_full.shape, dtype=np.uint8)
        Y_abc[tooth_full > 0] = 1
        Y_abc[curve_full > 0] = 2

        out_case_dir = ensure_dir(os.path.join(infer_root, case_id))
        save_volume(
            os.path.join(out_case_dir, "ABC_curve_full.nii.gz"),
            curve_full.astype(np.uint8),
            affine=affine_full,
            spacing=spacing_full,
            dtype=np.uint8,
        )
        save_volume(
            os.path.join(out_case_dir, "Y_ABC_pred.nii.gz"),
            Y_abc.astype(np.uint8),
            affine=affine_full,
            spacing=spacing_full,
            dtype=np.uint8,
        )
        logger.info("abc stitched case=%s", case_id)

    metrics_path = os.path.join(infer_root, "metrics_per_tooth.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "tooth_id",
                "status",
                "curve_length_mm",
                "n_components",
                "n_inner_wall_candidates",
                "n_curve_points",
            ],
        )
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)
    logger.info("abc per-tooth metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
