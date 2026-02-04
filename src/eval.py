import argparse
import csv
import glob
import json
import os
import numpy as np
from scipy.spatial import cKDTree

from src.datasets.io import load_volume
from src.datasets.points import load_points, get_points_for_tooth
from src.utils.config import load_config, ensure_dir
from src.utils.log import get_logger

logger = get_logger("eval")


def compute_distances(points_vox, curve_mask, spacing):
    if points_vox is None or len(points_vox) == 0:
        return np.array([], dtype=np.float32)
    curve_pts = np.array(np.where(curve_mask > 0)).T.astype(np.float32)
    if curve_pts.shape[0] == 0:
        return np.full((len(points_vox),), np.inf, dtype=np.float32)
    spacing = np.asarray(spacing, dtype=np.float32)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    taus = cfg["eval"]["eval_taus_mm"]

    processed_dir = cfg["data"]["processed_dir"]
    infer_dir = os.path.join(cfg["data"]["output_dir"], "infer")

    tooth_dirs = sorted(glob.glob(os.path.join(processed_dir, "*", "tooth_*")))
    if not tooth_dirs:
        logger.warning("no processed teeth found")
        return

    out_dir = ensure_dir(os.path.join(cfg["data"]["output_dir"], "eval"))
    per_tooth_csv = os.path.join(out_dir, "metrics_per_tooth.csv")
    summary_json = os.path.join(out_dir, "metrics_summary.json")

    rows = []
    all_dists = []

    for tdir in tooth_dirs:
        roi_meta_path = os.path.join(tdir, "roi_meta.json")
        with open(roi_meta_path, "r") as f:
            roi_meta = json.load(f)
        case_id = roi_meta["case_id"]
        tooth_id = roi_meta["tooth_id"]

        points_path = os.path.join(tdir, "points.json")
        points_data = load_points(points_path)
        pts = get_points_for_tooth(points_data, tooth_id)

        pred_path = os.path.join(infer_dir, case_id, f"tooth_{tooth_id}", "C_pred.nii.gz")
        if not os.path.exists(pred_path):
            continue

        C_pred, spacing, _ = load_volume(pred_path, dtype=np.uint8)
        d = compute_distances(pts, C_pred, spacing)
        metrics = summarize_metrics(d, taus)

        row = {
            "case_id": case_id,
            "tooth_id": tooth_id,
            "n_points": int(len(pts)),
            "mean_dist_mm": metrics["mean"],
            "p95_dist_mm": metrics["p95"],
        }
        for t in taus:
            row[f"sr@{t}"] = metrics["sr"][str(t)]
        rows.append(row)
        if d.size > 0 and np.isfinite(d).any():
            all_dists.append(d[np.isfinite(d)])

    # write per-tooth csv
    fieldnames = ["case_id", "tooth_id", "n_points", "mean_dist_mm", "p95_dist_mm"] + [f"sr@{t}" for t in taus]
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

    with open(summary_json, "w") as f:
        json.dump(summary, f)

    logger.info("saved metrics to %s", out_dir)


if __name__ == "__main__":
    main()
