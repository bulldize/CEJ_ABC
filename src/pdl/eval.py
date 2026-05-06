import argparse
import csv
import glob
import json
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from src.utils.config import ensure_dir, load_config
from src.utils.log import get_logger

logger = get_logger("pdl_eval")

DEFAULT_CONFIG = "/Users/bulldize/Desktop/华西口腔_CEJ_ABC/runtime/pdl/configs/pdl_runtime.yaml"

PER_TOOTH_FIELDS = [
    "case_id",
    "tooth_id",
    "status",
    "anchorage_area_mm2",
    "n_tooth_vertices",
    "n_bone_vertices",
    "n_seed_vertices",
    "n_attached_vertices",
    "n_boundary_candidates",
    "n_boundary_points",
    "n_cycles",
    "mean_distance_mm_attached",
    "p95_distance_mm_attached",
    "mean_parallelism_attached",
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


def _summary(values):
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


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_TOOTH_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _collect_rows(infer_root):
    rows = []
    tooth_dirs = sorted(glob.glob(os.path.join(infer_root, "*", "tooth_*")))
    for tooth_dir in tooth_dirs:
        case_id = os.path.basename(os.path.dirname(tooth_dir))
        tooth_id = os.path.basename(tooth_dir).replace("tooth_", "")
        area_path = os.path.join(tooth_dir, "anchorage_area.json")

        if not os.path.exists(area_path):
            rows.append(
                {
                    "case_id": case_id,
                    "tooth_id": tooth_id,
                    "status": "missing_metrics",
                    "anchorage_area_mm2": None,
                    "n_tooth_vertices": None,
                    "n_bone_vertices": None,
                    "n_seed_vertices": None,
                    "n_attached_vertices": None,
                    "n_boundary_candidates": None,
                    "n_boundary_points": None,
                    "n_cycles": None,
                    "mean_distance_mm_attached": None,
                    "p95_distance_mm_attached": None,
                    "mean_parallelism_attached": None,
                }
            )
            continue

        with open(area_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        m = payload.get("metrics", {})

        rows.append(
            {
                "case_id": str(payload.get("case_id", case_id)),
                "tooth_id": str(payload.get("tooth_id", tooth_id)),
                "status": str(payload.get("status", "unknown")),
                "anchorage_area_mm2": _safe_float(payload.get("anchorage_area_mm2")),
                "n_tooth_vertices": _safe_int(m.get("n_tooth_vertices")),
                "n_bone_vertices": _safe_int(m.get("n_bone_vertices")),
                "n_seed_vertices": _safe_int(m.get("n_seed_vertices")),
                "n_attached_vertices": _safe_int(m.get("n_attached_vertices")),
                "n_boundary_candidates": _safe_int(m.get("n_boundary_candidates")),
                "n_boundary_points": _safe_int(m.get("n_boundary_points")),
                "n_cycles": _safe_int(m.get("n_cycles")),
                "mean_distance_mm_attached": _safe_float(m.get("mean_distance_mm_attached")),
                "p95_distance_mm_attached": _safe_float(m.get("p95_distance_mm_attached")),
                "mean_parallelism_attached": _safe_float(m.get("mean_parallelism_attached")),
            }
        )
    return rows


def _build_summary(rows, infer_dir):
    status_counts = Counter([str(r.get("status", "unknown")) for r in rows])
    ok_rows = [r for r in rows if str(r.get("status")) == "ok"]
    area_vals = [r.get("anchorage_area_mm2") for r in ok_rows if r.get("anchorage_area_mm2") is not None]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_infer_dir": infer_dir,
        "total_teeth": int(len(rows)),
        "ok_teeth": int(len(ok_rows)),
        "failed_teeth": int(len(rows) - len(ok_rows)),
        "status_counts": dict(sorted(status_counts.items(), key=lambda kv: kv[0])),
        "anchorage_area_mm2": _summary(area_vals),
    }


def _build_quality_report(rows, case_id=None):
    report = {
        "case_id": case_id,
        "total": int(len(rows)),
        "ok": int(sum(1 for r in rows if str(r.get("status")) == "ok")),
        "failed": int(sum(1 for r in rows if str(r.get("status")) != "ok")),
        "failed_items": [
            {
                "case_id": str(r.get("case_id")),
                "tooth_id": str(r.get("tooth_id")),
                "status": str(r.get("status")),
            }
            for r in rows
            if str(r.get("status")) != "ok"
        ],
    }
    return report


def run_eval(cfg):
    infer_root = os.path.join(cfg["data"]["output_dir"], "infer")
    out_root = ensure_dir(os.path.join(cfg["data"]["output_dir"], "eval"))

    if not os.path.isdir(infer_root):
        logger.warning("infer root not found: %s", infer_root)
        return None, None, None

    rows = _collect_rows(infer_root)
    if not rows:
        logger.warning("no tooth-level infer folders found under %s", infer_root)
        return None, None, None

    per_tooth_csv = os.path.join(out_root, "metrics_per_tooth.csv")
    _write_csv(per_tooth_csv, rows)

    summary = _build_summary(rows, infer_root)
    summary_json = os.path.join(out_root, "metrics_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    quality_report = _build_quality_report(rows, case_id=None)
    quality_path = os.path.join(out_root, "quality_report.json")
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    case_ids = sorted({str(r["case_id"]) for r in rows})
    for case_id in case_ids:
        case_rows = [r for r in rows if str(r["case_id"]) == case_id]
        case_dir = ensure_dir(os.path.join(infer_root, case_id))
        _write_csv(os.path.join(case_dir, "metrics_per_tooth.csv"), case_rows)

        case_summary = _build_summary(case_rows, os.path.join(infer_root, case_id))
        case_summary["case_id"] = case_id
        with open(os.path.join(case_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(case_summary, f, ensure_ascii=False, indent=2)

        case_quality = _build_quality_report(case_rows, case_id=case_id)
        with open(os.path.join(case_dir, "quality_report.json"), "w", encoding="utf-8") as f:
            json.dump(case_quality, f, ensure_ascii=False, indent=2)

    logger.info("pdl eval written: %s", out_root)
    return per_tooth_csv, summary_json, quality_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_eval(cfg)


if __name__ == "__main__":
    main()
