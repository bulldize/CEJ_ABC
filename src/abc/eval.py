import argparse
import csv
import glob
import json
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import label

from src.datasets.io import load_volume
from src.utils.config import ensure_dir, load_config
from src.utils.log import get_logger

logger = get_logger("abc_eval")

PER_TOOTH_FIELDS = [
    "case_id",
    "tooth_id",
    "status",
    "meta_exists",
    "curve_exists",
    "meta_curve_length_mm",
    "meta_n_components",
    "meta_n_inner_wall_candidates",
    "meta_n_curve_points",
    "curve_voxels",
    "curve_components",
    "curve_length_mm_from_mask",
]


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_int(v):
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _parse_tooth_id(tooth_dir_name):
    if tooth_dir_name.startswith("tooth_"):
        raw = tooth_dir_name[len("tooth_") :]
        try:
            return int(raw)
        except Exception:
            return raw
    return tooth_dir_name


def _summary_stats(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _curve_metrics(curve_path):
    if not os.path.exists(curve_path):
        return {
            "curve_exists": 0,
            "curve_voxels": 0,
            "curve_components": 0,
            "curve_length_mm_from_mask": 0.0,
        }

    curve, spacing, _ = load_volume(curve_path, dtype=np.uint8)
    curve = (curve > 0).astype(np.uint8)
    curve_voxels = int(curve.sum())
    _, n_components = label(curve)

    spacing = np.asarray(spacing, dtype=np.float32)
    spacing = np.where(spacing <= 0, 1.0, spacing)
    curve_length_mm = float(curve_voxels * float(np.min(spacing)))

    return {
        "curve_exists": 1,
        "curve_voxels": curve_voxels,
        "curve_components": int(n_components),
        "curve_length_mm_from_mask": curve_length_mm,
    }


def _read_meta(meta_path):
    if not os.path.exists(meta_path):
        return {
            "meta_exists": 0,
            "status": "missing_meta",
            "meta_curve_length_mm": None,
            "meta_n_components": None,
            "meta_n_inner_wall_candidates": None,
            "meta_n_curve_points": None,
        }

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "meta_exists": 1,
        "status": str(meta.get("status", "unknown")),
        "meta_curve_length_mm": _safe_float(meta.get("curve_length_mm")),
        "meta_n_components": _safe_int(meta.get("n_components")),
        "meta_n_inner_wall_candidates": _safe_int(meta.get("n_inner_wall_candidates")),
        "meta_n_curve_points": _safe_int(meta.get("n_curve_points")),
    }


def _collect_rows(infer_root):
    tooth_dirs = sorted(glob.glob(os.path.join(infer_root, "*", "tooth_*")))
    rows = []
    for tooth_dir in tooth_dirs:
        if not os.path.isdir(tooth_dir):
            continue

        case_id = os.path.basename(os.path.dirname(tooth_dir))
        tooth_dir_name = os.path.basename(tooth_dir)
        tooth_id = _parse_tooth_id(tooth_dir_name)

        meta = _read_meta(os.path.join(tooth_dir, "abc_meta.json"))
        curve = _curve_metrics(os.path.join(tooth_dir, "C_ABC.nii.gz"))

        row = {
            "case_id": case_id,
            "tooth_id": tooth_id,
            "status": meta["status"],
            "meta_exists": meta["meta_exists"],
            "curve_exists": curve["curve_exists"],
            "meta_curve_length_mm": meta["meta_curve_length_mm"],
            "meta_n_components": meta["meta_n_components"],
            "meta_n_inner_wall_candidates": meta["meta_n_inner_wall_candidates"],
            "meta_n_curve_points": meta["meta_n_curve_points"],
            "curve_voxels": curve["curve_voxels"],
            "curve_components": curve["curve_components"],
            "curve_length_mm_from_mask": curve["curve_length_mm_from_mask"],
        }
        rows.append(row)
    return rows


def _build_summary(rows, infer_root):
    status_counts = Counter([str(r["status"]) for r in rows])

    curve_lengths_mask = [r["curve_length_mm_from_mask"] for r in rows if int(r["curve_exists"]) == 1]
    curve_voxels = [r["curve_voxels"] for r in rows if int(r["curve_exists"]) == 1]
    curve_components = [r["curve_components"] for r in rows if int(r["curve_exists"]) == 1]
    meta_curve_lengths = [r["meta_curve_length_mm"] for r in rows if r["meta_curve_length_mm"] is not None]
    meta_curve_points = [r["meta_n_curve_points"] for r in rows if r["meta_n_curve_points"] is not None]

    case_ids = sorted({str(r["case_id"]) for r in rows})
    total_teeth = len(rows)
    with_meta = int(sum(int(r["meta_exists"]) for r in rows))
    with_curve = int(sum(int(r["curve_exists"]) for r in rows))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_infer_dir": infer_root,
        "total_cases": len(case_ids),
        "total_teeth": total_teeth,
        "teeth_with_meta": with_meta,
        "teeth_without_meta": int(total_teeth - with_meta),
        "teeth_with_curve": with_curve,
        "teeth_without_curve": int(total_teeth - with_curve),
        "status_counts": dict(sorted(status_counts.items(), key=lambda kv: kv[0])),
        "curve_length_mm_from_mask": _summary_stats(curve_lengths_mask),
        "curve_voxels": _summary_stats(curve_voxels),
        "curve_components": _summary_stats(curve_components),
        "meta_curve_length_mm": _summary_stats(meta_curve_lengths),
        "meta_n_curve_points": _summary_stats(meta_curve_points),
    }


def run_eval(cfg):
    infer_root = os.path.join(cfg["data"]["output_dir"], "infer")
    out_root = ensure_dir(os.path.join(cfg["data"]["output_dir"], "eval"))

    if not os.path.isdir(infer_root):
        logger.warning("infer root not found: %s", infer_root)
        return None, None

    rows = _collect_rows(infer_root)
    if not rows:
        logger.warning("no tooth-level infer folders found under %s", infer_root)
        return None, None

    per_tooth_csv = os.path.join(out_root, "metrics_per_tooth.csv")
    with open(per_tooth_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_TOOTH_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = _build_summary(rows, infer_root)
    summary_json = os.path.join(out_root, "metrics_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("abc eval written: %s", out_root)
    return per_tooth_csv, summary_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/abc_default.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_eval(cfg)


if __name__ == "__main__":
    main()
