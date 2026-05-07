import argparse
import csv
import glob
import json
import os
import numpy as np
from scipy.spatial import cKDTree
from monai.data.utils import affine_to_spacing

from src.datasets.io import load_volume
from src.datasets.points import load_points, get_points_for_tooth
from src.utils.config import load_config, ensure_dir
from src.utils.log import get_logger

logger = get_logger("eval")


def _ensure_spacing(spacing, affine):
    if spacing is not None:
        return spacing
    if affine is None:
        return (1.0, 1.0, 1.0)
    return tuple(affine_to_spacing(affine))


def _missing_curve_penalty_mm(curve_mask, spacing):
    spacing = np.asarray(spacing, dtype=np.float32)
    shape = np.asarray(curve_mask.shape, dtype=np.float32)
    diag = np.maximum(shape - 1.0, 0.0) * spacing
    penalty = float(np.linalg.norm(diag))
    if penalty > 0.0 and np.isfinite(penalty):
        return penalty
    return float(np.max(spacing)) if spacing.size > 0 else 1.0


def compute_distances(points_vox, curve_mask, spacing):
    if points_vox is None or len(points_vox) == 0:
        return np.array([], dtype=np.float32)
    spacing = np.asarray(spacing, dtype=np.float32)
    curve_pts = np.array(np.where(curve_mask > 0)).T.astype(np.float32)
    if curve_pts.shape[0] == 0:
        penalty = _missing_curve_penalty_mm(curve_mask, spacing)
        return np.full((len(points_vox),), penalty, dtype=np.float32)
    curve_mm = curve_pts * spacing
    pts_mm = np.asarray(points_vox, dtype=np.float32) * spacing
    tree = cKDTree(curve_mm)
    d, _ = tree.query(pts_mm, k=1)
    return d.astype(np.float32)


def summarize_metrics(distances, taus):
    if distances.size == 0:
        return {"mean": None, "p95": None, "sr": {str(t): None for t in taus}}
    mean = float(np.mean(distances))
    p95 = float(np.percentile(distances, 95))
    sr = {str(t): float(np.mean(distances <= t)) for t in taus}
    return {"mean": mean, "p95": p95, "sr": sr}


def _load_points_for_tooth(tdir, tooth_id):
    points_path = os.path.join(tdir, "points.json")
    points_data = load_points(points_path)
    return get_points_for_tooth(points_data, tooth_id)


def _fallback_shape_spacing(tdir, processed_format):
    for name, dtype in (("T_t", np.uint8), ("H_GT", np.float32), ("A_t", np.float32)):
        path = os.path.join(tdir, f"{name}.{processed_format}")
        if not os.path.exists(path):
            continue
        arr, spacing, affine = load_volume(path, dtype=dtype)
        return arr.shape, _ensure_spacing(spacing, affine)
    return (1, 1, 1), (1.0, 1.0, 1.0)


def evaluate_predictions(
    processed_dir,
    infer_dir,
    out_dir,
    taus,
    processed_format="nii.gz",
    prediction_name="C_pred_fit.nii.gz",
):
    tooth_dirs = sorted(glob.glob(os.path.join(processed_dir, "*", "tooth_*")))
    if not tooth_dirs:
        raise RuntimeError(f"no processed teeth found in {processed_dir}")

    out_dir = ensure_dir(out_dir)
    per_tooth_csv = os.path.join(out_dir, "metrics_per_tooth.csv")
    summary_json = os.path.join(out_dir, "metrics_summary.json")

    rows = []
    all_dists = []
    missing_prediction_count = 0

    for tdir in tooth_dirs:
        roi_meta_path = os.path.join(tdir, "roi_meta.json")
        with open(roi_meta_path, "r") as f:
            roi_meta = json.load(f)
        case_id = roi_meta["case_id"]
        tooth_id = roi_meta["tooth_id"]

        pts = _load_points_for_tooth(tdir, tooth_id)

        pred_path = os.path.join(infer_dir, case_id, f"tooth_{tooth_id}", prediction_name)
        if not os.path.exists(pred_path):
            shape, spacing = _fallback_shape_spacing(tdir, processed_format)
            penalty = _missing_curve_penalty_mm(np.zeros(shape, dtype=np.uint8), spacing)
            if len(pts) > 0:
                d = np.full((len(pts),), penalty, dtype=np.float32)
                all_dists.append(d)
                metrics = summarize_metrics(d, taus)
            else:
                metrics = summarize_metrics(np.array([penalty], dtype=np.float32), taus)
            missing_prediction_count += 1
            row = {
                "case_id": case_id,
                "tooth_id": tooth_id,
                "n_points": int(len(pts)),
                "status": "missing_prediction",
                "pred_path": pred_path,
                "mean_dist_mm": metrics["mean"],
                "p95_dist_mm": metrics["p95"],
            }
            for t in taus:
                row[f"sr@{t}"] = metrics["sr"][str(t)]
            rows.append(row)
            logger.warning("missing prediction case=%s tooth=%s path=%s", case_id, tooth_id, pred_path)
            continue

        C_pred, spacing, affine = load_volume(pred_path, dtype=np.uint8)
        spacing = _ensure_spacing(spacing, affine)
        d = compute_distances(pts, C_pred, spacing)
        metrics = summarize_metrics(d, taus)

        row = {
            "case_id": case_id,
            "tooth_id": tooth_id,
            "n_points": int(len(pts)),
            "status": "ok",
            "pred_path": pred_path,
            "mean_dist_mm": metrics["mean"],
            "p95_dist_mm": metrics["p95"],
        }
        for t in taus:
            row[f"sr@{t}"] = metrics["sr"][str(t)]
        rows.append(row)
        if d.size > 0:
            all_dists.append(d)

    # write per-tooth csv
    fieldnames = ["case_id", "tooth_id", "n_points", "status", "pred_path", "mean_dist_mm", "p95_dist_mm"] + [f"sr@{t}" for t in taus]
    with open(per_tooth_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    if all_dists:
        all_dists = np.concatenate(all_dists)
    else:
        all_dists = np.array([], dtype=np.float32)
    summary = summarize_metrics(all_dists, taus)
    summary.update(
        {
            "prediction_name": prediction_name,
            "tooth_count": len(rows),
            "missing_prediction_count": int(missing_prediction_count),
            "failed_tooth_count": int(sum(1 for r in rows if r["status"] != "ok")),
            "ok_tooth_count": int(sum(1 for r in rows if r["status"] == "ok")),
        }
    )

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("saved metrics to %s", out_dir)
    return {
        "per_tooth_csv": per_tooth_csv,
        "summary_json": summary_json,
        "summary": summary,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    taus = cfg["eval"]["eval_taus_mm"]

    processed_dir = cfg["data"]["processed_dir"]
    infer_dir = os.path.join(cfg["data"]["output_dir"], "infer")
    out_dir = os.path.join(cfg["data"]["output_dir"], "eval")
    prediction_name = cfg.get("eval", {}).get("prediction_name", "C_pred_fit.nii.gz")
    evaluate_predictions(
        processed_dir=processed_dir,
        infer_dir=infer_dir,
        out_dir=out_dir,
        taus=taus,
        processed_format=cfg["data"].get("processed_format", "nii.gz"),
        prediction_name=prediction_name,
    )


if __name__ == "__main__":
    main()
