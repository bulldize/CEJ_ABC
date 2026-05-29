import argparse
import json
import os
import numpy as np
from scipy.ndimage import zoom

from src.datasets.io import load_volume, save_volume
from src.datasets.raw_cases import collect_raw_cases
from src.datasets.points import load_points, get_points_for_tooth, ensure_voxel_points, save_points
from src.datasets.mark_points import ensure_mark_points
from src.datasets.roi import crop_roi
from src.datasets.heatmap import fit_curve_and_sample, generate_heatmap_from_points
from src.datasets.cej_geometry import (
    build_geometry_prior,
    compute_curve_fit_report,
    geometry_prior_to_jsonable,
)
from src.utils.config import load_config, ensure_dir
from src.utils.geometry import points_full_to_roi, distance_to_surface
from src.utils.log import get_logger
from src.postprocess.skeleton import extract_curve_from_heatmap_peak


logger = get_logger("preprocess")


def map_pulp_to_tooth(B):
    B = B.copy()
    mask = B >= 100
    B[mask] = B[mask] % 100
    return B


def get_tooth_labels(B):
    labels = np.unique(B)
    labels = [int(l) for l in labels if 10 <= l < 100]
    return sorted(labels)


def resample_roi(A_roi, T_roi, points_roi, spacing, target_spacing):
    spacing = np.asarray(spacing, dtype=np.float32)
    target = np.asarray(target_spacing, dtype=np.float32)
    if np.allclose(spacing, target):
        return A_roi, T_roi, points_roi, spacing, False, np.array([1.0, 1.0, 1.0])
    scale = spacing / target
    A_rs = zoom(A_roi, scale, order=1)
    T_rs = zoom(T_roi, scale, order=0)
    pts_rs = points_roi * scale
    return A_rs, T_rs, pts_rs, target, True, scale


def preprocess_case(case_rec, cfg):
    case_id = case_rec["case_id"]
    case_dir = case_rec.get("case_dir", None)
    raw_a = case_rec["a_path"]
    raw_b = case_rec["b_path"]
    raw_points = case_rec.get("points_path", "")
    raw_meta = case_rec.get("meta_path", None)
    if raw_meta and not os.path.exists(raw_meta):
        raw_meta = None

    A, spacing, affine = load_volume(raw_a, meta_path=raw_meta, dtype=np.float32)
    spacing = tuple(spacing)
    B, spacing_b, affine_b = load_volume(raw_b, meta_path=raw_meta, dtype=np.int16)
    if A.shape != B.shape:
        logger.error("shape mismatch A%s vs B%s for case=%s", A.shape, B.shape, case_id)
        return
    if not np.allclose(spacing, spacing_b, atol=1e-3):
        logger.warning("spacing mismatch A%s vs B%s for case=%s", spacing, spacing_b, case_id)
    if not np.allclose(affine, affine_b, atol=1e-3):
        logger.warning("affine mismatch for case=%s", case_id)

    if case_dir:
        try:
            converted = ensure_mark_points(case_dir, case_id, A.shape, affine, cfg)
            if converted:
                logger.info("converted cej_points_ras.xlsx for case=%s", case_id)
        except Exception as e:
            logger.error("failed to convert cej_points_ras.xlsx for case=%s: %s", case_id, e)
            raise

    B = map_pulp_to_tooth(B)
    tooth_labels = get_tooth_labels(B)

    points_exist = os.path.exists(raw_points)
    points_data = load_points(raw_points)
    coord_type = points_data.get("coord_type", "voxel")
    points_dict = points_data.get("points", {}) or {}
    has_points = any(len(v) > 0 for v in points_dict.values())
    if not has_points:
        if points_exist:
            logger.info("points.json empty for case=%s; writing empty heatmaps for smoke test", case_id)
        else:
            logger.info("no points.json for case=%s; writing empty heatmaps for smoke test", case_id)

    spacing_case = tuple(spacing)
    spacing_case_arr = np.asarray(spacing_case, dtype=np.float32)
    padding_mm = cfg["preprocess"]["roi_padding_mm"]
    pad_vox = np.round(np.array(padding_mm) / spacing_case_arr).astype(int)

    processed_case_dir = ensure_dir(os.path.join(cfg["data"]["processed_dir"], case_id))
    full_tooth_centroids_mm = {}
    for label in tooth_labels:
        vox = np.argwhere(B == label).astype(np.float32)
        if vox.shape[0] > 0:
            full_tooth_centroids_mm[int(label)] = vox.mean(axis=0) * spacing_case_arr
    arch_centroid_mm = None
    if full_tooth_centroids_mm:
        arch_centroid_mm = np.stack(list(full_tooth_centroids_mm.values()), axis=0).mean(axis=0)
    geometry_cfg = cfg.get("preprocess", {}).get("geometry_prior", {})
    use_geometry_prior = bool(geometry_cfg.get("enabled", True))

    for tooth_id in tooth_labels:
        T_t = (B == tooth_id).astype(np.uint8)
        out = crop_roi(A, T_t, pad_vox)
        if out is None:
            continue
        A_roi, T_roi, origin, mins, maxs = out

        point_dist_report = None
        if has_points:
            pts = get_points_for_tooth(points_data, tooth_id)
            pts_vox = ensure_voxel_points(pts, coord_type, affine)
            # remove outliers by surface distance + report distribution
            if pts_vox.shape[0] > 0:
                dist = distance_to_surface(T_t, spacing)
                keep = []
                distances = []
                invalid = 0
                removed = 0
                thresh = cfg["preprocess"]["point_surface_dist_thresh_mm"]
                for p in pts_vox:
                    x, y, z = np.round(p).astype(int)
                    if 0 <= x < dist.shape[0] and 0 <= y < dist.shape[1] and 0 <= z < dist.shape[2]:
                        d = float(dist[x, y, z])
                        distances.append(d)
                        if d <= thresh:
                            keep.append(p)
                        else:
                            removed += 1
                    else:
                        distances.append(float("inf"))
                        invalid += 1
                pts_vox = np.asarray(keep, dtype=np.float32)
                finite = np.asarray([d for d in distances if np.isfinite(d)], dtype=np.float32)
                point_dist_report = {
                    "case_id": case_id,
                    "tooth_id": tooth_id,
                    "threshold_mm": float(thresh),
                    "n_points": int(len(distances)),
                    "n_in_bounds": int(len(finite)),
                    "n_removed": int(removed),
                    "n_invalid": int(invalid),
                    "mean_mm": float(np.mean(finite)) if finite.size > 0 else None,
                    "p95_mm": float(np.percentile(finite, 95)) if finite.size > 0 else None,
                    "max_mm": float(np.max(finite)) if finite.size > 0 else None,
                    "distances_mm": [float(d) if np.isfinite(d) else None for d in distances],
                }
                if removed > 0 or invalid > 0:
                    logger.warning(
                        "point filter case=%s tooth=%s removed=%d invalid=%d",
                        case_id, tooth_id, removed, invalid
                    )
            pts_roi = points_full_to_roi(pts_vox, origin)
        else:
            pts_vox = np.zeros((0, 3), dtype=np.float32)
            pts_roi = pts_vox

        # optional resample
        roi_shape_full = A_roi.shape
        spacing_full = spacing_case
        spacing_roi = spacing_case
        if cfg["preprocess"]["resample_to_target"]:
            A_roi, T_roi, pts_roi, spacing_roi, resampled, scale = resample_roi(
                A_roi, T_roi, pts_roi, spacing_case, cfg["preprocess"]["target_spacing_mm"]
            )
        else:
            resampled, scale = False, np.array([1.0, 1.0, 1.0])

        tooth_dir = ensure_dir(os.path.join(processed_case_dir, f"tooth_{tooth_id}"))
        fmt = cfg["data"]["processed_format"]

        save_volume(os.path.join(tooth_dir, f"A_t.{fmt}"), A_roi, affine=None, spacing=spacing_roi)
        save_volume(os.path.join(tooth_dir, f"T_t.{fmt}"), T_roi.astype(np.uint8), affine=None, spacing=spacing_roi)

        geometry_prior = None
        if use_geometry_prior:
            geometry_prior = build_geometry_prior(
                T_roi,
                spacing=spacing_roi,
                tooth_id=tooth_id,
                case_id=case_id,
                points_vox=pts_roi,
                full_tooth_centroids_mm=full_tooth_centroids_mm,
                arch_centroid_mm=arch_centroid_mm,
                roi_origin_vox=origin,
                spacing_full=spacing_full,
                constraint_params=geometry_cfg.get("constraints", {}),
            )

        dense_pts, fit_report = fit_curve_and_sample(
            pts_roi,
            spacing_roi,
            cfg["preprocess"]["dense_sample_step_mm"],
            closed=bool(cfg["preprocess"].get("curve_closed", True)),
            smooth=float(cfg["preprocess"].get("curve_smooth", 0.0)),
            geometry_prior=geometry_prior,
            return_report=True,
        )
        H_GT = generate_heatmap_from_points(
            A_roi.shape,
            dense_pts,
            spacing_roi,
            cfg["preprocess"]["sigma_mm"],
            connect_points=True,
            close_loop=bool(cfg["preprocess"].get("curve_closed", True)),
        )
        C_GT = extract_curve_from_heatmap_peak(
            H_GT,
            T_roi,
            peak_threshold=float(cfg["preprocess"].get("gt_curve_peak_threshold", 0.999)),
            keep_lcc=False,
        )
        curve_fit_report = compute_curve_fit_report(
            pts_roi,
            dense_pts,
            spacing_roi,
            C_gt=C_GT,
            geometry_prior=geometry_prior,
            fit_report=fit_report,
        )
        if geometry_prior is not None:
            geometry_prior["fit_summary"] = {
                "fallback": bool(fit_report.get("fallback", False)),
                "fallback_reason": fit_report.get("fallback_reason"),
                "n_output_points": int(fit_report.get("n_output_points", 0)),
                "constraints_passed": bool((curve_fit_report.get("constraints") or {}).get("passed", False)),
                "manual_to_curve_mean_mm": (
                    curve_fit_report.get("manual_to_curve") or {}
                ).get("mean_mm"),
            }
        save_volume(os.path.join(tooth_dir, f"H_GT.{fmt}"), H_GT.astype(np.float32), affine=None, spacing=spacing_roi)
        save_volume(os.path.join(tooth_dir, f"C_GT.{fmt}"), C_GT.astype(np.uint8), affine=None, spacing=spacing_roi)
        save_points(
            os.path.join(tooth_dir, "points.json"),
            case_id,
            "voxel",
            "roi",
            {str(tooth_id): pts_roi.tolist()},
        )
        np.save(os.path.join(tooth_dir, "curve_dense_points.npy"), dense_pts.astype(np.float32))
        if geometry_prior is not None:
            with open(os.path.join(tooth_dir, "geometry_prior.json"), "w") as f:
                json.dump(geometry_prior_to_jsonable(geometry_prior), f)
        with open(os.path.join(tooth_dir, "curve_fit_report.json"), "w") as f:
            json.dump(curve_fit_report, f)

        roi_meta = {
            "case_id": case_id,
            "tooth_id": tooth_id,
            "roi_origin_in_full": [int(v) for v in origin],
            "roi_shape": [int(v) for v in A_roi.shape],
            "roi_shape_full": [int(v) for v in roi_shape_full],
            "full_shape": [int(v) for v in A.shape],
            "spacing": [float(v) for v in spacing_roi],
            "spacing_full": [float(v) for v in spacing_full],
            "affine": affine.tolist(),
            "coord_type": "voxel",
            "space": "roi",
            "bbox_full": {"min": [int(v) for v in mins], "max": [int(v) for v in maxs]},
            "padding_mm": float(padding_mm),
            "target_spacing_mm": [float(v) for v in cfg["preprocess"]["target_spacing_mm"]],
            "resampled": bool(resampled),
            "resample_scale": [float(v) for v in scale],
            "has_points": bool(has_points),
        }
        with open(os.path.join(tooth_dir, "roi_meta.json"), "w") as f:
            json.dump(roi_meta, f)
        if point_dist_report is not None:
            with open(os.path.join(tooth_dir, "point_surface_dist.json"), "w") as f:
                json.dump(point_dist_report, f)

        logger.info("processed case=%s tooth=%s", case_id, tooth_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cases = collect_raw_cases(cfg["data"])
    if not cases:
        logger.warning("no cases found for data.raw_dir=%s", cfg["data"]["raw_dir"])
        return

    for case_rec in cases:
        preprocess_case(case_rec, cfg)


if __name__ == "__main__":
    main()
