import argparse
import glob
import json
import os
import shutil

import numpy as np
from scipy.ndimage import distance_transform_edt, zoom

from src.datasets.io import load_volume, save_volume
from src.datasets.raw_cases import collect_raw_case_map
from src.postprocess.skeleton import (
    compute_curve_overlap_metrics,
    extract_curve,
    extract_curve_from_heatmap_peak,
)
from src.utils.config import ensure_dir, load_config
from src.utils.log import get_logger

logger = get_logger("export_pseudo_gt_full")

MINIMAL_EXPORT_FILES = {
    "CEJ_medical_compare_full.nii.gz",
    "CEJ_qc_compare_full.nii.gz",
    "CEJ_model_compare_full.nii.gz",
    "export_meta.json",
}


def _build_three_segment_compare(tooth_mask, segment2_mask, segment3_mask):
    tooth = (np.asarray(tooth_mask) > 0)
    seg2 = (np.asarray(segment2_mask) > 0)
    seg3 = (np.asarray(segment3_mask) > 0)
    out = np.zeros(tooth.shape, dtype=np.uint8)
    out[tooth] = 1
    out[seg2] = 2
    out[seg3] = 3
    vals, cnts = np.unique(out, return_counts=True)
    return out, {
        "label_counts": {int(v): int(c) for v, c in zip(vals, cnts)},
        "segment2_segment3_overlap_voxels": int(np.logical_and(seg2, seg3).sum()),
    }


def map_pulp_to_tooth(B):
    B = B.copy()
    mask = B >= 100
    B[mask] = B[mask] % 100
    return B


def resize_to_shape(vol, shape, order):
    out = vol
    if out.shape != tuple(shape):
        scale = [shape[i] / out.shape[i] for i in range(3)]
        out = zoom(out, scale, order=order)
        out = out[: shape[0], : shape[1], : shape[2]]
        pad = [shape[i] - out.shape[i] for i in range(3)]
        if any(p > 0 for p in pad):
            out = np.pad(out, [(0, pad[0]), (0, pad[1]), (0, pad[2])], mode="constant")
    return out


def _stitch_patch_max(full, patch, origin):
    x0, y0, z0 = [int(v) for v in origin]
    x1 = min(full.shape[0], x0 + patch.shape[0])
    y1 = min(full.shape[1], y0 + patch.shape[1])
    z1 = min(full.shape[2], z0 + patch.shape[2])
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return
    px = x1 - x0
    py = y1 - y0
    pz = z1 - z0
    full[x0:x1, y0:y1, z0:z1] = np.maximum(full[x0:x1, y0:y1, z0:z1], patch[:px, :py, :pz])


def _stitch_patch_binary_or(full, patch, origin):
    x0, y0, z0 = [int(v) for v in origin]
    x1 = min(full.shape[0], x0 + patch.shape[0])
    y1 = min(full.shape[1], y0 + patch.shape[1])
    z1 = min(full.shape[2], z0 + patch.shape[2])
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return
    px = x1 - x0
    py = y1 - y0
    pz = z1 - z0
    full[x0:x1, y0:y1, z0:z1] = np.maximum(full[x0:x1, y0:y1, z0:z1], patch[:px, :py, :pz])


def _rasterize_polyline_to_mask(mask, points_xyz, close_loop=False):
    if points_xyz is None:
        return
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return

    shape = np.array(mask.shape, dtype=np.int32)

    def _mark(p):
        q = np.round(p).astype(np.int32)
        if np.all(q >= 0) and np.all(q < shape):
            mask[q[0], q[1], q[2]] = 1

    _mark(pts[0])
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        delta = np.abs(p1 - p0)
        n = int(np.ceil(np.max(delta))) + 1
        n = max(2, n)
        seg = np.linspace(p0, p1, n)
        for p in seg:
            _mark(p)
    if close_loop and pts.shape[0] > 2:
        p0 = pts[-1]
        p1 = pts[0]
        delta = np.abs(p1 - p0)
        n = int(np.ceil(np.max(delta))) + 1
        n = max(2, n)
        seg = np.linspace(p0, p1, n)
        for p in seg:
            _mark(p)


def _dilate_mask_mm(mask, spacing_xyz, radius_mm):
    if radius_mm is None or radius_mm <= 0:
        return (mask > 0).astype(np.uint8)
    base = (mask > 0).astype(np.uint8)
    if base.sum() == 0:
        return base
    spacing = tuple(float(v) for v in spacing_xyz)
    dist = distance_transform_edt(base == 0, sampling=spacing)
    return (dist <= float(radius_mm)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output folder for full-mouth pseudo-GT NIfTI. Default: outputs/pseudo_gt_review_nifti",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = args.out_dir or os.path.join(cfg["data"]["output_dir"], "pseudo_gt_review_nifti")
    out_root = ensure_dir(out_root)
    root_ds_store = os.path.join(out_root, ".DS_Store")
    if os.path.exists(root_ds_store):
        os.remove(root_ds_store)

    processed_dir = cfg["data"]["processed_dir"]
    raw_case_map = collect_raw_case_map(cfg["data"])
    fmt = cfg["data"]["processed_format"]

    threshold = float(cfg.get("viz", {}).get("pseudo_gt_skeleton_threshold", cfg["infer"]["threshold_theta"]))
    curve_closed = bool(cfg.get("preprocess", {}).get("curve_closed", True))
    export_cfg = cfg.get("export_pseudo_gt", {})
    keep_lcc = bool(export_cfg.get("keep_lcc_for_skeleton", False))
    skeleton_from_heatmap_peak = bool(export_cfg.get("skeleton_from_heatmap_peak", True))
    heatmap_peak_threshold = float(export_cfg.get("heatmap_peak_threshold", 0.999))
    nonoverlap_tubes = bool(export_cfg.get("nonoverlap_tubes_for_slicer", True))
    skeleton_tube_radius_mm = float(export_cfg.get("skeleton_tube_radius_mm", 0.6))
    interp_tube_radius_mm = float(export_cfg.get("interp_tube_radius_mm", 1.2))
    points_tube_radius_mm = float(export_cfg.get("points_tube_radius_mm", 1.0))
    consistency_iou_min = float(export_cfg.get("consistency_iou_min", 0.95))
    consistency_dice_min = float(export_cfg.get("consistency_dice_min", 0.97))
    fail_on_consistency_violation = bool(export_cfg.get("fail_on_consistency_violation", True))

    tooth_dirs = sorted(glob.glob(os.path.join(processed_dir, "*", "tooth_*")))
    if not tooth_dirs:
        logger.warning("no processed teeth found under %s", processed_dir)
        return

    cases = {}
    for tdir in tooth_dirs:
        roi_meta_path = os.path.join(tdir, "roi_meta.json")
        if not os.path.exists(roi_meta_path):
            continue
        with open(roi_meta_path, "r") as f:
            roi_meta = json.load(f)
        case_id = roi_meta["case_id"]
        tooth_id = roi_meta["tooth_id"]

        h_path = os.path.join(tdir, f"H_GT.{fmt}")
        t_path = os.path.join(tdir, f"T_t.{fmt}")
        dense_curve_path = os.path.join(tdir, "curve_dense_points.npy")
        points_path = os.path.join(tdir, "points.json")
        if not os.path.exists(h_path) or not os.path.exists(t_path):
            continue

        H_roi, _, _ = load_volume(h_path, dtype=np.float32)
        T_roi, _, _ = load_volume(t_path, dtype=np.uint8)
        if skeleton_from_heatmap_peak:
            C_roi = extract_curve_from_heatmap_peak(
                H_roi,
                T_roi,
                peak_threshold=heatmap_peak_threshold,
                keep_lcc=keep_lcc,
            )
        else:
            C_roi = extract_curve(H_roi, T_roi, threshold=threshold, keep_lcc=keep_lcc)
        dense_curve_roi = np.zeros((0, 3), dtype=np.float32)
        if os.path.exists(dense_curve_path):
            try:
                dense_curve_roi = np.load(dense_curve_path).astype(np.float32)
            except Exception as e:
                logger.warning("failed loading dense curve points %s: %s", dense_curve_path, e)
        manual_points_roi = np.zeros((0, 3), dtype=np.float32)
        if os.path.exists(points_path):
            try:
                with open(points_path, "r", encoding="utf-8") as f:
                    points_obj = json.load(f)
                points_dict = points_obj.get("points", {}) or {}
                manual_points_roi = np.asarray(points_dict.get(str(tooth_id), []), dtype=np.float32)
            except Exception as e:
                logger.warning("failed loading points %s: %s", points_path, e)

        if roi_meta.get("resampled", False):
            scale = np.array(roi_meta.get("resample_scale", [1.0, 1.0, 1.0]), dtype=np.float32)
            inv_scale = 1.0 / scale
            H_roi = zoom(H_roi, inv_scale, order=1)
            H_roi = resize_to_shape(H_roi, roi_meta["roi_shape_full"], order=1)
            T_roi = zoom(T_roi, inv_scale, order=0)
            T_roi = resize_to_shape(T_roi, roi_meta["roi_shape_full"], order=0).astype(np.uint8)
            C_roi = zoom(C_roi, inv_scale, order=0)
            C_roi = resize_to_shape(C_roi, roi_meta["roi_shape_full"], order=0).astype(np.uint8)
            if dense_curve_roi.shape[0] > 0:
                dense_curve_roi = dense_curve_roi * inv_scale
            if manual_points_roi.shape[0] > 0:
                manual_points_roi = manual_points_roi * inv_scale

        cases.setdefault(case_id, []).append(
            {
                "tooth_id": tooth_id,
                "origin": roi_meta["roi_origin_in_full"],
                "heatmap": H_roi.astype(np.float32),
                "tooth_mask": T_roi.astype(np.uint8),
                "curve": C_roi.astype(np.uint8),
                "dense_curve_roi": dense_curve_roi.astype(np.float32),
                "manual_points_roi": manual_points_roi.astype(np.float32),
            }
        )

    if not cases:
        logger.warning("no pseudo-GT ROI found to export")
        return

    all_violations = []

    for case_id, items in cases.items():
        case_rec = raw_case_map.get(case_id, None)
        if case_rec is None:
            logger.warning("skip case=%s because raw record not found", case_id)
            continue
        a_path = case_rec["a_path"]
        b_path = case_rec["b_path"]
        meta_path = case_rec.get("meta_path", None)
        if meta_path and not os.path.exists(meta_path):
            meta_path = None

        if not os.path.exists(a_path) or not os.path.exists(b_path):
            logger.warning("skip case=%s due to missing raw A/B", case_id)
            continue

        A_full, spacing, affine = load_volume(a_path, meta_path=meta_path, dtype=np.float32)
        B_full, _, _ = load_volume(b_path, meta_path=meta_path, dtype=np.int16)
        B_tooth = map_pulp_to_tooth(B_full)
        tooth_mask_full = (B_tooth >= 10).astype(np.uint8)

        C_full = np.zeros(A_full.shape, dtype=np.uint8)
        C_interp_full = np.zeros(A_full.shape, dtype=np.uint8)
        P_manual_full = np.zeros(A_full.shape, dtype=np.uint8)
        per_tooth_consistency = []
        case_violations = []
        for it in items:
            _stitch_patch_binary_or(C_full, it["curve"], it["origin"])
            dense_curve_roi = it.get("dense_curve_roi", None)
            interp_roi = np.zeros_like(it["tooth_mask"], dtype=np.uint8)
            if dense_curve_roi is not None and len(dense_curve_roi) > 0:
                _rasterize_polyline_to_mask(interp_roi, dense_curve_roi, close_loop=curve_closed)
                interp_roi = interp_roi * (it["tooth_mask"] > 0).astype(np.uint8)
                _stitch_patch_binary_or(C_interp_full, interp_roi, it["origin"])

            tooth_metrics = compute_curve_overlap_metrics(it["curve"], interp_roi)
            tooth_metrics["tooth_id"] = int(it["tooth_id"])
            tooth_metrics["passes_threshold"] = bool(
                tooth_metrics["iou"] >= consistency_iou_min and tooth_metrics["dice"] >= consistency_dice_min
            )
            if not tooth_metrics["passes_threshold"]:
                case_violations.append(
                    {
                        "case_id": case_id,
                        "tooth_id": int(it["tooth_id"]),
                        "iou": float(tooth_metrics["iou"]),
                        "dice": float(tooth_metrics["dice"]),
                        "curve_a_voxels": int(tooth_metrics["curve_a_voxels"]),
                        "curve_b_voxels": int(tooth_metrics["curve_b_voxels"]),
                    }
                )
            per_tooth_consistency.append(tooth_metrics)

            manual_points_roi = it.get("manual_points_roi", None)
            if manual_points_roi is not None and len(manual_points_roi) > 0:
                origin = np.asarray(it["origin"], dtype=np.float32)
                manual_points_full = manual_points_roi + origin[None, :]
                for p in manual_points_full:
                    q = np.round(p).astype(np.int32)
                    if (
                        0 <= q[0] < P_manual_full.shape[0]
                        and 0 <= q[1] < P_manual_full.shape[1]
                        and 0 <= q[2] < P_manual_full.shape[2]
                    ):
                        P_manual_full[q[0], q[1], q[2]] = 1

        case_metrics = compute_curve_overlap_metrics(C_full, C_interp_full)
        curve_iou = float(case_metrics["iou"])
        curve_dice = float(case_metrics["dice"])
        curve_equal = bool(case_metrics["equal_voxelwise"])

        pred_curve_full = np.zeros_like(C_full, dtype=np.uint8)
        pred_full_path = os.path.join(cfg["data"]["output_dir"], "infer", case_id, "Y_pred.nii.gz")
        if os.path.exists(pred_full_path):
            pred_full_y, _, _ = load_volume(pred_full_path, dtype=np.uint8)
            if pred_full_y.shape != C_full.shape:
                pred_full_y = resize_to_shape(pred_full_y, C_full.shape, order=0)
            pred_curve_full = (pred_full_y == 2).astype(np.uint8)
        else:
            logger.warning("prediction not found for case=%s, model compare will contain reference only", case_id)

        medical_compare, medical_stats = _build_three_segment_compare(
            tooth_mask_full,
            P_manual_full,
            C_interp_full,
        )
        qc_compare, qc_stats = _build_three_segment_compare(
            tooth_mask_full,
            C_interp_full,
            C_full,
        )
        model_compare, model_stats = _build_three_segment_compare(
            tooth_mask_full,
            C_interp_full,
            pred_curve_full,
        )

        out_case_dir = ensure_dir(os.path.join(out_root, case_id))
        for stale_name in os.listdir(out_case_dir):
            if stale_name in MINIMAL_EXPORT_FILES:
                continue
            stale_path = os.path.join(out_case_dir, stale_name)
            if os.path.isdir(stale_path):
                shutil.rmtree(stale_path)
            else:
                os.remove(stale_path)

        save_volume(
            os.path.join(out_case_dir, "CEJ_medical_compare_full.nii.gz"),
            medical_compare,
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )
        save_volume(
            os.path.join(out_case_dir, "CEJ_qc_compare_full.nii.gz"),
            qc_compare,
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )
        save_volume(
            os.path.join(out_case_dir, "CEJ_model_compare_full.nii.gz"),
            model_compare,
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )

        meta = {
            "case_id": case_id,
            "source_processed_dir": processed_dir,
            "n_teeth_stitched": len(items),
            "skeleton_threshold": threshold,
            "keep_lcc_for_skeleton": keep_lcc,
            "skeleton_from_heatmap_peak": skeleton_from_heatmap_peak,
            "heatmap_peak_threshold": heatmap_peak_threshold,
            "nonoverlap_tubes_for_slicer": nonoverlap_tubes,
            "tube_radius_mm": {
                "skeleton": skeleton_tube_radius_mm,
                "interpolated_curve": interp_tube_radius_mm,
                "manual_points": points_tube_radius_mm,
            },
            "consistency_thresholds": {
                "iou_min": consistency_iou_min,
                "dice_min": consistency_dice_min,
                "fail_on_consistency_violation": fail_on_consistency_violation,
            },
            "curve_consistency": {
                "case_level": case_metrics,
                "per_tooth": per_tooth_consistency,
                "n_violations": len(case_violations),
                "violation_tooth_ids": [int(v["tooth_id"]) for v in case_violations],
            },
            "volumes": sorted(list(MINIMAL_EXPORT_FILES - {"export_meta.json"})),
            "compare_label_definition": {
                "0": "background",
                "1": "tooth_body",
                "2": "segment_2",
                "3": "segment_3",
            },
            "compare_files": {
                "CEJ_medical_compare_full.nii.gz": {
                    "segment_1": "tooth_body",
                    "segment_2": "manual_points",
                    "segment_3": "pseudo_gt_interpolated_curve",
                    "voxel_stats": medical_stats,
                },
                "CEJ_qc_compare_full.nii.gz": {
                    "segment_1": "tooth_body",
                    "segment_2": "pseudo_gt_interpolated_curve",
                    "segment_3": "pseudo_gt_skeleton",
                    "voxel_stats": qc_stats,
                },
                "CEJ_model_compare_full.nii.gz": {
                    "segment_1": "tooth_body",
                    "segment_2": "pseudo_gt_interpolated_curve",
                    "segment_3": "predicted_curve",
                    "voxel_stats": model_stats,
                },
            },
        }
        with open(os.path.join(out_case_dir, "export_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(
            "exported CEJ compare NIfTI set for case=%s -> %s",
            case_id,
            out_case_dir,
        )
        logger.info(
            "curve consistency case=%s | equal=%s iou=%.6f dice=%.6f",
            case_id,
            curve_equal,
            curve_iou,
            curve_dice,
        )
        if case_violations:
            all_violations.extend(case_violations)
            logger.error(
                "curve consistency violations case=%s | %s",
                case_id,
                ", ".join(
                    [
                        f"tooth_{v['tooth_id']}(iou={v['iou']:.4f},dice={v['dice']:.4f},a={v['curve_a_voxels']},b={v['curve_b_voxels']})"
                        for v in case_violations
                    ]
                ),
            )

    logger.info("done. output root: %s", out_root)
    if fail_on_consistency_violation and all_violations:
        logger.error(
            "consistency gate failed: %d violating teeth (iou>=%.3f and dice>=%.3f required)",
            len(all_violations),
            consistency_iou_min,
            consistency_dice_min,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
