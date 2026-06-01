import argparse
import glob
import json
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.datasets.cej_geometry import (
    build_geometry_prior,
    compute_curve_fit_report,
    geometry_prior_to_jsonable,
)
from src.datasets.heatmap import fit_curve_and_sample, generate_heatmap_from_points
from src.datasets.io import load_volume, save_volume
from src.datasets.points import load_points, get_points_for_tooth
from src.postprocess.skeleton import extract_curve_from_heatmap_peak
from src.utils.config import load_config
from src.utils.log import get_logger


logger = get_logger("upgrade_processed_geometry_gt")


def _case_tooth_dirs(processed_dir):
    case_dirs = sorted(d for d in glob.glob(os.path.join(processed_dir, "*")) if os.path.isdir(d))
    for case_dir in case_dirs:
        tooth_dirs = sorted(glob.glob(os.path.join(case_dir, "tooth_*")))
        if tooth_dirs:
            yield case_dir, tooth_dirs


def _load_roi_centroids(tooth_dirs, fmt):
    centroids = {}
    arch_pts = []
    for tdir in tooth_dirs:
        meta_path = os.path.join(tdir, "roi_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tooth_id = int(meta["tooth_id"])
        T, spacing, _ = load_volume(os.path.join(tdir, f"T_t.{fmt}"), dtype=np.uint8)
        pts = np.argwhere(T > 0).astype(np.float32)
        if pts.shape[0] == 0:
            continue
        origin = np.asarray(meta.get("roi_origin_in_full", [0, 0, 0]), dtype=np.float32)
        spacing_full = np.asarray(meta.get("spacing_full", spacing), dtype=np.float32)
        center_mm = (pts.mean(axis=0) + origin) * spacing_full
        centroids[tooth_id] = center_mm.astype(np.float32)
        arch_pts.append(center_mm)
    arch_centroid = np.stack(arch_pts, axis=0).mean(axis=0).astype(np.float32) if arch_pts else None
    return centroids, arch_centroid


def upgrade_processed_dir(cfg):
    processed_dir = cfg["data"]["processed_dir"]
    fmt = cfg["data"].get("processed_format", "nii.gz")
    preprocess_cfg = cfg.get("preprocess", {})
    geometry_cfg = preprocess_cfg.get("geometry_prior", {})
    step_mm = float(preprocess_cfg.get("dense_sample_step_mm", 0.2))
    sigma_mm = float(preprocess_cfg.get("sigma_mm", 1.0))
    curve_closed = bool(preprocess_cfg.get("curve_closed", True))
    curve_smooth = float(preprocess_cfg.get("curve_smooth", 0.0))
    peak_threshold = float(preprocess_cfg.get("gt_curve_peak_threshold", 0.999))
    write_shape_prior = bool(
        geometry_cfg.get("write_shape_prior_heatmap", preprocess_cfg.get("write_shape_prior_heatmap", False))
    )
    shape_prior_sigma_mm = float(
        geometry_cfg.get("shape_prior_sigma_mm", preprocess_cfg.get("shape_prior_sigma_mm", 2.0))
    )
    shape_prior_name = str(geometry_cfg.get("shape_prior_name", preprocess_cfg.get("shape_prior_name", "H_SHAPE_PRIOR")))

    upgraded = 0
    for _, tooth_dirs in _case_tooth_dirs(processed_dir):
        centroids, arch_centroid = _load_roi_centroids(tooth_dirs, fmt)
        for tdir in tooth_dirs:
            meta_path = os.path.join(tdir, "roi_meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            case_id = meta["case_id"]
            tooth_id = int(meta["tooth_id"])

            A, spacing, _ = load_volume(os.path.join(tdir, f"A_t.{fmt}"), dtype=np.float32)
            T, _, _ = load_volume(os.path.join(tdir, f"T_t.{fmt}"), dtype=np.uint8)
            points_data = load_points(os.path.join(tdir, "points.json"))
            pts = get_points_for_tooth(points_data, tooth_id).astype(np.float32)

            geometry_prior = build_geometry_prior(
                T,
                spacing=spacing,
                tooth_id=tooth_id,
                case_id=case_id,
                points_vox=pts,
                full_tooth_centroids_mm=centroids,
                arch_centroid_mm=arch_centroid,
                roi_origin_vox=meta.get("roi_origin_in_full", [0, 0, 0]),
                spacing_full=meta.get("spacing_full", spacing),
                constraint_params=geometry_cfg.get("constraints", {}),
            )
            dense_pts, fit_report = fit_curve_and_sample(
                pts,
                spacing=spacing,
                step_mm=step_mm,
                closed=curve_closed,
                smooth=curve_smooth,
                geometry_prior=geometry_prior,
                return_report=True,
            )
            H_GT = generate_heatmap_from_points(
                A.shape,
                dense_pts,
                spacing=spacing,
                sigma_mm=sigma_mm,
                connect_points=True,
                close_loop=curve_closed,
            )
            C_GT = extract_curve_from_heatmap_peak(H_GT, T, peak_threshold=peak_threshold, keep_lcc=False)
            curve_fit_report = compute_curve_fit_report(
                pts,
                dense_pts,
                spacing,
                C_gt=C_GT,
                geometry_prior=geometry_prior,
                fit_report=fit_report,
            )
            geometry_prior["fit_summary"] = {
                "fallback": bool(fit_report.get("fallback", False)),
                "fallback_reason": fit_report.get("fallback_reason"),
                "n_output_points": int(fit_report.get("n_output_points", 0)),
                "constraints_passed": bool((curve_fit_report.get("constraints") or {}).get("passed", False)),
                "manual_to_curve_mean_mm": (curve_fit_report.get("manual_to_curve") or {}).get("mean_mm"),
            }

            np.save(os.path.join(tdir, "curve_dense_points.npy"), dense_pts.astype(np.float32))
            save_volume(os.path.join(tdir, f"H_GT.{fmt}"), H_GT.astype(np.float32), affine=None, spacing=spacing)
            save_volume(os.path.join(tdir, f"C_GT.{fmt}"), C_GT.astype(np.uint8), affine=None, spacing=spacing)
            if write_shape_prior:
                H_shape_prior = generate_heatmap_from_points(
                    A.shape,
                    dense_pts,
                    spacing=spacing,
                    sigma_mm=shape_prior_sigma_mm,
                    connect_points=True,
                    close_loop=curve_closed,
                )
                save_volume(
                    os.path.join(tdir, f"{shape_prior_name}.{fmt}"),
                    H_shape_prior.astype(np.float32),
                    affine=None,
                    spacing=spacing,
                )
                curve_fit_report["shape_prior"] = {
                    "path": os.path.join(tdir, f"{shape_prior_name}.{fmt}"),
                    "sigma_mm": shape_prior_sigma_mm,
                    "max": float(np.max(H_shape_prior)) if H_shape_prior.size else 0.0,
                }
            with open(os.path.join(tdir, "geometry_prior.json"), "w", encoding="utf-8") as f:
                json.dump(geometry_prior_to_jsonable(geometry_prior), f)
            with open(os.path.join(tdir, "curve_fit_report.json"), "w", encoding="utf-8") as f:
                json.dump(curve_fit_report, f)
            upgraded += 1
            logger.info("upgraded case=%s tooth=%s", case_id, tooth_id)

    logger.info("upgraded %d tooth ROIs in %s", upgraded, processed_dir)
    return upgraded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    upgrade_processed_dir(cfg)


if __name__ == "__main__":
    main()
