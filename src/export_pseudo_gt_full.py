import argparse
import glob
import json
import os

import numpy as np
from scipy.ndimage import distance_transform_edt, zoom

from src.datasets.io import load_volume, save_volume
from src.datasets.raw_cases import collect_raw_case_map
from src.postprocess.skeleton import extract_curve, extract_curve_from_heatmap_peak
from src.utils.config import ensure_dir, load_config
from src.utils.log import get_logger

logger = get_logger("export_pseudo_gt_full")


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

        H_full = np.zeros(A_full.shape, dtype=np.float32)
        C_full = np.zeros(A_full.shape, dtype=np.uint8)
        C_interp_full = np.zeros(A_full.shape, dtype=np.uint8)
        P_manual_full = np.zeros(A_full.shape, dtype=np.uint8)
        for it in items:
            _stitch_patch_max(H_full, it["heatmap"], it["origin"])
            _stitch_patch_binary_or(C_full, it["curve"], it["origin"])
            dense_curve_roi = it.get("dense_curve_roi", None)
            if dense_curve_roi is not None and len(dense_curve_roi) > 0:
                interp_roi = np.zeros_like(it["tooth_mask"], dtype=np.uint8)
                _rasterize_polyline_to_mask(interp_roi, dense_curve_roi, close_loop=curve_closed)
                interp_roi = interp_roi * (it["tooth_mask"] > 0).astype(np.uint8)
                _stitch_patch_binary_or(C_interp_full, interp_roi, it["origin"])
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

        C_skeleton_tube_full = _dilate_mask_mm(C_full, spacing, skeleton_tube_radius_mm)
        C_interp_tube_full = _dilate_mask_mm(C_interp_full, spacing, interp_tube_radius_mm)
        if nonoverlap_tubes:
            C_interp_tube_full = np.logical_and(C_interp_tube_full > 0, C_skeleton_tube_full == 0).astype(np.uint8)
        P_manual_tube_full = _dilate_mask_mm(P_manual_full, spacing, points_tube_radius_mm)

        inter = np.logical_and(C_full > 0, C_interp_full > 0).sum()
        union = np.logical_or(C_full > 0, C_interp_full > 0).sum()
        sum_ab = (C_full > 0).sum() + (C_interp_full > 0).sum()
        curve_iou = float(inter / union) if union > 0 else 1.0
        curve_dice = float((2.0 * inter) / sum_ab) if sum_ab > 0 else 1.0
        curve_equal = bool(np.array_equal(C_full, C_interp_full))

        Y_pseudo = np.zeros(A_full.shape, dtype=np.int16)
        Y_pseudo[(B_tooth >= 10)] = 1
        Y_pseudo[C_skeleton_tube_full > 0] = 2

        Y_pseudo_review = np.zeros(A_full.shape, dtype=np.int16)
        Y_pseudo_review[(B_tooth >= 10)] = 1
        Y_pseudo_review[C_skeleton_tube_full > 0] = 2
        Y_pseudo_review[C_interp_tube_full > 0] = 3

        # Optional version with manual points embedded as label 4.
        Y_pseudo_review_with_points = Y_pseudo_review.copy()
        Y_pseudo_review_with_points[P_manual_tube_full > 0] = 4
        vals, counts = np.unique(Y_pseudo_review, return_counts=True)
        label_stats = {int(v): int(c) for v, c in zip(vals, counts)}

        out_case_dir = ensure_dir(os.path.join(out_root, case_id))
        save_volume(os.path.join(out_case_dir, "A_full.nii.gz"), A_full, affine=affine, spacing=spacing, dtype=np.float32)
        save_volume(os.path.join(out_case_dir, "B_tooth_full.nii.gz"), B_tooth.astype(np.int16), affine=affine, spacing=spacing, dtype=np.int16)
        save_volume(
            os.path.join(out_case_dir, "H_pseudo_gt_full.nii.gz"),
            H_full,
            affine=affine,
            spacing=spacing,
            dtype=np.float32,
        )
        save_volume(
            os.path.join(out_case_dir, "C_pseudo_gt_skeleton_full.nii.gz"),
            C_full,
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )
        save_volume(
            os.path.join(out_case_dir, "C_pseudo_gt_skeleton_tube_full.nii.gz"),
            C_skeleton_tube_full,
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )
        save_volume(
            os.path.join(out_case_dir, "C_pseudo_gt_interp_curve_full.nii.gz"),
            C_interp_full,
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )
        save_volume(
            os.path.join(out_case_dir, "C_pseudo_gt_interp_curve_tube_full.nii.gz"),
            C_interp_tube_full,
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )
        save_volume(
            os.path.join(out_case_dir, "P_manual_points_full.nii.gz"),
            P_manual_full,
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )
        save_volume(
            os.path.join(out_case_dir, "P_manual_points_tube_full.nii.gz"),
            P_manual_tube_full,
            affine=affine,
            spacing=spacing,
            dtype=np.uint8,
        )
        save_volume(
            os.path.join(out_case_dir, "Y_pseudo_gt_full.nii.gz"),
            Y_pseudo,
            affine=affine,
            spacing=spacing,
            dtype=np.int16,
        )
        save_volume(
            os.path.join(out_case_dir, "Y_pseudo_gt_review_full.nii.gz"),
            Y_pseudo_review,
            affine=affine,
            spacing=spacing,
            dtype=np.int16,
        )
        save_volume(
            os.path.join(out_case_dir, "Y_pseudo_gt_review_slicer_full.nii.gz"),
            Y_pseudo_review,
            affine=affine,
            spacing=spacing,
            dtype=np.int16,
        )
        save_volume(
            os.path.join(out_case_dir, "Y_pseudo_gt_review_with_points_full.nii.gz"),
            Y_pseudo_review_with_points,
            affine=affine,
            spacing=spacing,
            dtype=np.int16,
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
            "curve_consistency": {
                "equal_voxelwise": curve_equal,
                "iou": curve_iou,
                "dice": curve_dice,
            },
            "volumes": [
                "A_full.nii.gz",
                "B_tooth_full.nii.gz",
                "H_pseudo_gt_full.nii.gz",
                "C_pseudo_gt_skeleton_full.nii.gz",
                "C_pseudo_gt_skeleton_tube_full.nii.gz",
                "C_pseudo_gt_interp_curve_full.nii.gz",
                "C_pseudo_gt_interp_curve_tube_full.nii.gz",
                "P_manual_points_full.nii.gz",
                "P_manual_points_tube_full.nii.gz",
                "Y_pseudo_gt_full.nii.gz",
                "Y_pseudo_gt_review_full.nii.gz",
                "Y_pseudo_gt_review_slicer_full.nii.gz",
                "Y_pseudo_gt_review_with_points_full.nii.gz",
            ],
            "label_definition_Y_pseudo_gt_full": {"0": "background", "1": "tooth", "2": "pseudo_gt_skeleton_tube"},
            "label_definition_Y_pseudo_gt_review_full": {
                "0": "background",
                "1": "tooth",
                "2": "pseudo_gt_skeleton_tube",
                "3": "pseudo_gt_interpolated_curve_tube",
            },
            "label_definition_Y_pseudo_gt_review_with_points_full": {
                "0": "background",
                "1": "tooth",
                "2": "pseudo_gt_skeleton_tube",
                "3": "pseudo_gt_interpolated_curve_tube",
                "4": "manual_mark_points_tube",
            },
        }
        with open(os.path.join(out_case_dir, "export_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(
            "exported full-mouth pseudo-GT NIfTI for case=%s -> %s | Y_review labels=%s",
            case_id,
            out_case_dir,
            label_stats,
        )
        logger.info(
            "curve consistency case=%s | equal=%s iou=%.6f dice=%.6f",
            case_id,
            curve_equal,
            curve_iou,
            curve_dice,
        )

    logger.info("done. output root: %s", out_root)


if __name__ == "__main__":
    main()
