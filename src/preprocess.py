import argparse
import glob
import json
import os
import numpy as np
from scipy.ndimage import zoom

from src.datasets.io import load_volume, save_volume
from src.datasets.points import load_points, get_points_for_tooth, ensure_voxel_points, save_points
from src.datasets.roi import crop_roi
from src.datasets.heatmap import fit_curve_and_sample, generate_heatmap_from_points
from src.utils.config import load_config, ensure_dir
from src.utils.geometry import points_full_to_roi, distance_to_surface
from src.utils.log import get_logger


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


def preprocess_case(case_dir, cfg):
    raw_a = os.path.join(case_dir, cfg["data"]["raw_a_name"])
    raw_b = os.path.join(case_dir, cfg["data"]["raw_b_name"])
    raw_points = os.path.join(case_dir, cfg["data"]["raw_points_name"])
    raw_meta = os.path.join(case_dir, cfg["data"]["raw_meta_name"])

    A, spacing, affine = load_volume(raw_a, meta_path=raw_meta, dtype=np.float32)
    B, spacing_b, affine_b = load_volume(raw_b, meta_path=raw_meta, dtype=np.int16)
    if A.shape != B.shape:
        logger.error("shape mismatch A%s vs B%s for case=%s", A.shape, B.shape, os.path.basename(case_dir))
        return
    if not np.allclose(spacing, spacing_b, atol=1e-3):
        logger.warning("spacing mismatch A%s vs B%s for case=%s", spacing, spacing_b, os.path.basename(case_dir))
    if not np.allclose(affine, affine_b, atol=1e-3):
        logger.warning("affine mismatch for case=%s", os.path.basename(case_dir))

    B = map_pulp_to_tooth(B)
    tooth_labels = get_tooth_labels(B)

    points_data = load_points(raw_points)
    coord_type = points_data.get("coord_type", "voxel")

    padding_mm = cfg["preprocess"]["roi_padding_mm"]
    pad_vox = np.round(np.array(padding_mm) / np.array(spacing)).astype(int)

    case_id = os.path.basename(case_dir)
    processed_case_dir = ensure_dir(os.path.join(cfg["data"]["processed_dir"], case_id))

    for tooth_id in tooth_labels:
        T_t = (B == tooth_id).astype(np.uint8)
        out = crop_roi(A, T_t, pad_vox)
        if out is None:
            continue
        A_roi, T_roi, origin, mins, maxs = out

        pts = get_points_for_tooth(points_data, tooth_id)
        pts_vox = ensure_voxel_points(pts, coord_type, affine)
        # remove outliers by surface distance + report distribution
        point_dist_report = None
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

        # optional resample
        roi_shape_full = A_roi.shape
        spacing_full = spacing
        if cfg["preprocess"]["resample_to_target"]:
            A_roi, T_roi, pts_roi, spacing, resampled, scale = resample_roi(
                A_roi, T_roi, pts_roi, spacing, cfg["preprocess"]["target_spacing_mm"]
            )
        else:
            resampled, scale = False, np.array([1.0, 1.0, 1.0])

        dense_pts = fit_curve_and_sample(pts_roi, spacing, cfg["preprocess"]["dense_sample_step_mm"])
        H_GT = generate_heatmap_from_points(A_roi.shape, dense_pts, spacing, cfg["preprocess"]["sigma_mm"])

        tooth_dir = ensure_dir(os.path.join(processed_case_dir, f"tooth_{tooth_id}"))
        fmt = cfg["data"]["processed_format"]

        save_volume(os.path.join(tooth_dir, f"A_t.{fmt}"), A_roi, affine=None, spacing=spacing)
        save_volume(os.path.join(tooth_dir, f"T_t.{fmt}"), T_roi.astype(np.uint8), affine=None, spacing=spacing)
        save_volume(os.path.join(tooth_dir, f"H_GT.{fmt}"), H_GT.astype(np.float32), affine=None, spacing=spacing)

        save_points(os.path.join(tooth_dir, "points.json"), case_id, "voxel", "roi", {str(tooth_id): pts_roi.tolist()})
        np.save(os.path.join(tooth_dir, "curve_dense_points.npy"), dense_pts.astype(np.float32))

        roi_meta = {
            "case_id": case_id,
            "tooth_id": tooth_id,
            "roi_origin_in_full": [int(v) for v in origin],
            "roi_shape": [int(v) for v in A_roi.shape],
            "roi_shape_full": [int(v) for v in roi_shape_full],
            "full_shape": [int(v) for v in A.shape],
            "spacing": [float(v) for v in spacing],
            "spacing_full": [float(v) for v in spacing_full],
            "affine": affine.tolist(),
            "coord_type": "voxel",
            "space": "roi",
            "bbox_full": {"min": [int(v) for v in mins], "max": [int(v) for v in maxs]},
            "padding_mm": float(padding_mm),
            "target_spacing_mm": [float(v) for v in cfg["preprocess"]["target_spacing_mm"]],
            "resampled": bool(resampled),
            "resample_scale": [float(v) for v in scale],
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
    raw_dir = cfg["data"]["raw_dir"]
    case_dirs = [d for d in glob.glob(os.path.join(raw_dir, "*")) if os.path.isdir(d)]

    if not case_dirs:
        logger.warning("no cases found in %s", raw_dir)
        return

    for case_dir in case_dirs:
        preprocess_case(case_dir, cfg)


if __name__ == "__main__":
    main()
