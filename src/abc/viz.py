import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from src.abc.export import _build_three_segment_compare, _curve_summary
from src.datasets.io import load_volume, save_volume
from src.utils.config import ensure_dir, load_config
from src.utils.log import get_logger
from src.viz import _ensure_3d_viz_deps, _write_3d_index, save_3d_viewer

logger = get_logger("abc_viz")


def _load_dense_curve_points(path):
    if not os.path.exists(path):
        return None
    try:
        pts = np.load(path).astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return None
        return pts
    except Exception:
        return None


def _load_tooth_axis_from_meta(meta_path):
    if not os.path.exists(meta_path):
        return None
    try:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
    except Exception:
        return None
    axis = np.asarray(meta.get("axis", []), dtype=np.float32)
    if axis.shape != (3,):
        return None
    n = float(np.linalg.norm(axis))
    if n <= 1e-6:
        return None
    return axis / n


def _filter_points_by_curve_mask(points_vox, curve_mask, max_dist_vox=1.5):
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return pts

    curve = np.asarray(curve_mask) > 0
    curve_vox = np.array(np.where(curve)).T.astype(np.float32)
    if curve_vox.shape[0] == 0:
        return pts

    valid = np.all(np.isfinite(pts), axis=1)
    if not np.any(valid):
        return pts

    tree = cKDTree(curve_vox)
    d = np.full((pts.shape[0],), np.inf, dtype=np.float32)
    d_valid, _ = tree.query(pts[valid], k=1)
    d[valid] = d_valid.astype(np.float32)

    keep = d <= float(max_dist_vox)
    kept = pts[keep]
    if kept.shape[0] >= 3:
        return kept
    return pts


def _break_long_segments_for_view(points_vox, jump_ratio=4.0, min_jump_vox=4.0):
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
        return pts

    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    valid_seg = np.isfinite(seg) & (seg > 1e-6)
    nz = seg[valid_seg]
    if nz.size == 0:
        return pts
    thr = max(float(min_jump_vox), float(np.median(nz)) * float(jump_ratio))

    out = [pts[0]]
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            if np.all(np.isfinite(p1)):
                out.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                out.append(p1)
            else:
                out.append(p1)
            continue
        if np.isfinite(seg[i]) and float(seg[i]) > thr:
            out.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
        out.append(p1)
    return np.asarray(out, dtype=np.float32)


def _clip_axis_tail_outliers_for_view(
    points_vox,
    spacing,
    axis,
    mad_k=3.0,
    min_outliers=4,
    asymmetry_ratio=1.5,
):
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 8 or pts.shape[1] != 3:
        return pts
    if axis is None:
        return pts

    axis = np.asarray(axis, dtype=np.float32)
    if axis.shape != (3,):
        return pts
    n = float(np.linalg.norm(axis))
    if n <= 1e-6:
        return pts
    axis = axis / n

    spacing_arr = np.asarray(spacing, dtype=np.float32)
    pts_mm = pts * spacing_arr[None, :]
    proj = pts_mm @ axis

    med = float(np.median(proj))
    mad = float(np.median(np.abs(proj - med)))
    if not np.isfinite(mad):
        return pts

    if mad <= 1e-6:
        # Fallback for near-constant main band where MAD collapses to zero.
        q10 = float(np.percentile(proj, 10))
        q90 = float(np.percentile(proj, 90))
        iqr = max(0.0, q90 - q10)
        robust_sigma = max(1e-3, iqr / 2.56)
    else:
        robust_sigma = 1.4826 * mad
    lo = med - float(mad_k) * robust_sigma
    hi = med + float(mad_k) * robust_sigma
    low_mask = proj < lo
    high_mask = proj > hi
    n_low = int(np.sum(low_mask))
    n_high = int(np.sum(high_mask))
    if n_low == 0 and n_high == 0:
        return pts

    q10 = float(np.percentile(proj, 10))
    q90 = float(np.percentile(proj, 90))
    low_span = med - q10
    high_span = q90 - med

    out = pts.copy()
    if n_low >= int(min_outliers) and low_span > max(0.6, float(asymmetry_ratio) * max(1e-6, high_span)):
        out[low_mask] = np.nan
    if n_high >= int(min_outliers) and high_span > max(0.6, float(asymmetry_ratio) * max(1e-6, low_span)):
        out[high_mask] = np.nan
    return out


def _select_cases_and_teeth(infer_dir, max_cases, max_teeth):
    selected = []
    cases = sorted([d for d in os.listdir(infer_dir) if os.path.isdir(os.path.join(infer_dir, d))])
    if max_cases > 0:
        cases = cases[:max_cases]

    for case_id in cases:
        case_tooth_dirs = sorted(
            [
                d
                for d in os.listdir(os.path.join(infer_dir, case_id))
                if d.startswith("tooth_") and os.path.isdir(os.path.join(infer_dir, case_id, d))
            ]
        )
        if max_teeth > 0:
            case_tooth_dirs = case_tooth_dirs[:max_teeth]

        for tooth_dir_name in case_tooth_dirs:
            selected.append((case_id, tooth_dir_name))

    return selected


def _ensure_curve_nifti_from_points(curve_path, dense_curve_path, shape, spacing):
    if os.path.exists(curve_path):
        return True

    pts = _load_dense_curve_points(dense_curve_path)
    if pts is None or pts.shape[0] == 0:
        return False

    vox = np.rint(pts).astype(np.int32)
    valid = (
        (vox[:, 0] >= 0)
        & (vox[:, 0] < shape[0])
        & (vox[:, 1] >= 0)
        & (vox[:, 1] < shape[1])
        & (vox[:, 2] >= 0)
        & (vox[:, 2] < shape[2])
    )
    vox = vox[valid]
    if vox.shape[0] == 0:
        return False

    curve = np.zeros(shape, dtype=np.uint8)
    curve[vox[:, 0], vox[:, 1], vox[:, 2]] = 1
    save_volume(curve_path, curve, spacing=spacing, dtype=np.uint8)
    logger.info("generated missing curve NIfTI from points: %s", curve_path)
    return True


def _ensure_case_compare_nifti(case_id, cfg):
    infer_root = os.path.join(cfg["data"]["output_dir"], "infer")
    export_case_dir = ensure_dir(os.path.join(cfg["data"]["output_dir"], "pseudo_gt_review_nifti", case_id))

    medical_path = os.path.join(export_case_dir, "ABC_medical_compare_full.nii.gz")
    model_path = os.path.join(export_case_dir, "ABC_model_compare_full.nii.gz")
    meta_path = os.path.join(export_case_dir, "export_meta.json")
    if os.path.exists(medical_path) and os.path.exists(model_path) and os.path.exists(meta_path):
        return

    y_path = os.path.join(infer_root, case_id, "Y_ABC_pred.nii.gz")
    if not os.path.exists(y_path):
        logger.warning("missing Y_ABC_pred for case=%s; cannot auto-generate compare NIfTI", case_id)
        return

    Y, spacing, affine = load_volume(y_path, dtype=np.uint8)
    tooth = (Y == 1).astype(np.uint8)
    abc_curve = (Y == 2).astype(np.uint8)
    reserved_placeholder = np.zeros_like(abc_curve, dtype=np.uint8)

    medical_compare, medical_stats = _build_three_segment_compare(tooth, reserved_placeholder, abc_curve)
    model_compare, model_stats = _build_three_segment_compare(tooth, reserved_placeholder, abc_curve)

    save_volume(medical_path, medical_compare, affine=affine, spacing=spacing, dtype=np.uint8)
    save_volume(model_path, model_compare, affine=affine, spacing=spacing, dtype=np.uint8)

    meta = {
        "case_id": case_id,
        "source_y_abc_pred": y_path,
        "compare_label_definition": {
            "0": "background",
            "1": "tooth_body",
            "2": "reserved_channel",
            "3": "abc_curve",
        },
        "reserved_channel_used": False,
        "curve_summary": _curve_summary(abc_curve, spacing),
        "compare_files": {
            "ABC_medical_compare_full.nii.gz": {
                "segment_1": "tooth_body",
                "segment_2": "reserved_channel(empty)",
                "segment_3": "abc_curve_rule_extracted",
                "voxel_stats": medical_stats,
            },
            "ABC_model_compare_full.nii.gz": {
                "segment_1": "tooth_body",
                "segment_2": "reserved_channel(empty)",
                "segment_3": "abc_curve_pred",
                "voxel_stats": model_stats,
            },
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("generated missing compare NIfTI set for case=%s", case_id)


def _rewrite_abc_viewer_labels(viewer_html_path):
    if not os.path.exists(viewer_html_path):
        return

    text = open(viewer_html_path, "r", encoding="utf-8").read()
    replacements = [
        ("CEJ 3D可视化", "ABC 3D可视化"),
        ("CEJ 3D\\u53ef\\u89c6\\u5316", "ABC 3D\\u53ef\\u89c6\\u5316"),
        ("1 标注点", "1 ABC点位"),
        ("1 \\u6807\\u6ce8\\u70b9", "1 ABC\\u70b9\\u4f4d"),
        ("2 插值曲线", "2 ABC插值曲线"),
        ("2 \\u63d2\\u503c\\u66f2\\u7ebf", "2 ABC\\u63d2\\u503c\\u66f2\\u7ebf"),
        ("3 伪GT热图", "3 ABC候选热图"),
        ("3 \\u4f2aGT\\u70ed\\u56fe", "3 ABC\\u5019\\u9009\\u70ed\\u56fe"),
        ("4 伪GT骨架", "4 ABC骨架"),
        ("4 \\u4f2aGT\\u9aa8\\u67b6", "4 ABC\\u9aa8\\u67b6"),
        ("5 推理热图", "5 ABC结果热图"),
        ("5 \\u63a8\\u7406\\u70ed\\u56fe", "5 ABC\\u7ed3\\u679c\\u70ed\\u56fe"),
        ("6 推理曲线", "6 ABC结果曲线"),
        ("6 \\u63a8\\u7406\\u66f2\\u7ebf", "6 ABC\\u7ed3\\u679c\\u66f2\\u7ebf"),
        ('"name":"插值曲线"', '"name":"ABC插值曲线"'),
        ('"name":"\\u63d2\\u503c\\u66f2\\u7ebf"', '"name":"ABC\\u63d2\\u503c\\u66f2\\u7ebf"'),
        ('"name":"推理曲线"', '"name":"ABC结果曲线"'),
        ('"name":"\\u63a8\\u7406\\u66f2\\u7ebf"', '"name":"ABC\\u7ed3\\u679c\\u66f2\\u7ebf"'),
        ('"name":"伪GT骨架"', '"name":"ABC骨架"'),
        ('"name":"\\u4f2aGT\\u9aa8\\u67b6"', '"name":"ABC\\u9aa8\\u67b6"'),
        ('"name":"伪GT热图', '"name":"ABC候选热图'),
        ('"name":"\\u4f2aGT\\u70ed\\u56fe', '"name":"ABC\\u5019\\u9009\\u70ed\\u56fe'),
        ('"name":"推理热图', '"name":"ABC结果热图'),
        ('"name":"\\u63a8\\u7406\\u70ed\\u56fe', '"name":"ABC\\u7ed3\\u679c\\u70ed\\u56fe'),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)

    with open(viewer_html_path, "w", encoding="utf-8") as f:
        f.write(text)


def _rewrite_abc_index_labels(index_html_path):
    if not os.path.exists(index_html_path):
        return
    text = open(index_html_path, "r", encoding="utf-8").read()
    text = text.replace("CEJ 3D可视化索引", "ABC 3D可视化索引")
    text = text.replace("CEJ 3D\\u53ef\\u89c6\\u5316\\u7d22\\u5f15", "ABC 3D\\u53ef\\u89c6\\u5316\\u7d22\\u5f15")
    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/abc_default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Reuse existing 3D viewer engine, while forcing ABC-only legend semantics.
    viz_cfg = dict(cfg.get("viz", {}))
    viz_cfg["show_pseudo_gt_skeleton"] = False
    viz_cfg["color_pseudo_gt_skeleton"] = "#7F7F7F"
    # ABC often has partial/irregular ring candidates; avoid forcing tail-to-head connection.
    viz_cfg["dense_interp_curve_closed"] = False
    cfg_for_viz = dict(cfg)
    cfg_for_viz["viz"] = viz_cfg
    _ensure_3d_viz_deps()

    processed_dir = cfg["data"]["processed_dir"]
    infer_dir = os.path.join(cfg["data"]["output_dir"], "infer")
    viz_root = ensure_dir(os.path.join(cfg["data"]["output_dir"], "viz", "3d"))

    max_cases = int(cfg.get("abc_viz", {}).get("max_cases", 5))
    max_teeth = int(cfg.get("abc_viz", {}).get("max_teeth_per_case", 8))

    if not os.path.isdir(infer_dir):
        logger.warning("abc infer dir not found: %s", infer_dir)
        return

    selected = _select_cases_and_teeth(infer_dir, max_cases=max_cases, max_teeth=max_teeth)
    if not selected:
        logger.warning("no abc tooth folders found under %s", infer_dir)
        return

    viewer_records = []
    ensured_compare_cases = set()
    for case_id, tooth_dir_name in selected:
        tooth_id = tooth_dir_name.replace("tooth_", "")
        infer_tooth_dir = os.path.join(infer_dir, case_id, tooth_dir_name)
        processed_tooth_dir = os.path.join(processed_dir, case_id, tooth_dir_name)

        if case_id not in ensured_compare_cases:
            _ensure_case_compare_nifti(case_id, cfg)
            ensured_compare_cases.add(case_id)

        a_path = os.path.join(processed_tooth_dir, "A_t.nii.gz")
        t_path = os.path.join(processed_tooth_dir, "T_t.nii.gz")
        b_path = os.path.join(processed_tooth_dir, "B_t.nii.gz")
        c_path = os.path.join(infer_tooth_dir, "C_ABC.nii.gz")
        dense_curve_path = os.path.join(infer_tooth_dir, "abc_curve_points_vox.npy")
        meta_path = os.path.join(infer_tooth_dir, "abc_meta.json")

        if not (os.path.exists(a_path) and os.path.exists(t_path)):
            logger.warning("skip missing ABC viz inputs: case=%s tooth=%s", case_id, tooth_id)
            continue

        A_roi, spacing, _ = load_volume(a_path, dtype=np.float32)
        T_roi, _, _ = load_volume(t_path, dtype=np.uint8)
        if not os.path.exists(c_path):
            _ensure_curve_nifti_from_points(
                curve_path=c_path,
                dense_curve_path=dense_curve_path,
                shape=T_roi.shape,
                spacing=spacing,
            )
        if not os.path.exists(c_path):
            logger.warning("skip missing ABC curve NIfTI: case=%s tooth=%s", case_id, tooth_id)
            continue

        C_roi, _, _ = load_volume(c_path, dtype=np.uint8)
        B_roi = None
        if os.path.exists(b_path):
            B_roi, _, _ = load_volume(b_path, dtype=np.uint8)

        C_roi = (C_roi > 0).astype(np.uint8)
        H_curve = C_roi.astype(np.float32)
        dense_curve_points = _load_dense_curve_points(dense_curve_path)
        if dense_curve_points is not None and dense_curve_points.shape[0] > 1:
            keep_dist_vox = float(cfg.get("abc_viz", {}).get("dense_curve_keep_dist_vox", 0.0))
            if keep_dist_vox > 0:
                dense_curve_points = _filter_points_by_curve_mask(
                    dense_curve_points,
                    C_roi,
                    max_dist_vox=keep_dist_vox,
                )
            dense_curve_points = _clip_axis_tail_outliers_for_view(
                dense_curve_points,
                spacing=spacing,
                axis=_load_tooth_axis_from_meta(meta_path),
                mad_k=float(cfg.get("abc_viz", {}).get("dense_curve_axis_tail_mad_k", 3.0)),
                min_outliers=int(cfg.get("abc_viz", {}).get("dense_curve_axis_tail_min_outliers", 4)),
                asymmetry_ratio=float(cfg.get("abc_viz", {}).get("dense_curve_axis_tail_asymmetry_ratio", 1.5)),
            )
            dense_curve_points = _break_long_segments_for_view(
                dense_curve_points,
                jump_ratio=float(cfg.get("abc_viz", {}).get("dense_curve_break_jump_ratio", 4.0)),
                min_jump_vox=float(cfg.get("abc_viz", {}).get("dense_curve_break_min_jump_vox", 4.0)),
            )

        out_dir = ensure_dir(os.path.join(viz_root, case_id, tooth_dir_name))
        out_html = os.path.join(out_dir, "viewer.html")

        # Reuse shared 3D viewer renderer; labels are rewritten to ABC semantics below.
        save_3d_viewer(
            A=A_roi,
            T=T_roi,
            H_gt=H_curve,
            spacing=spacing,
            points=np.zeros((0, 3), dtype=np.float32),
            case_id=case_id,
            tooth_id=tooth_id,
            out_html=out_html,
            cfg=cfg_for_viz,
            H_pred=H_curve,
            C_pred=C_roi,
            C_gt=None,
            dense_curve_points=dense_curve_points,
            R=B_roi,
            distances=None,
        )
        _rewrite_abc_viewer_labels(out_html)

        viewer_records.append(
            {
                "case_id": case_id,
                "tooth_id": tooth_id,
                "rel_path": os.path.relpath(out_html, viz_root),
            }
        )
        logger.info("abc viz generated case=%s tooth=%s", case_id, tooth_id)

    if viewer_records:
        index_path = os.path.join(viz_root, "index.html")
        _write_3d_index(index_path, viewer_records)
        _rewrite_abc_index_labels(index_path)
        logger.info("abc viz index -> %s", index_path)


if __name__ == "__main__":
    main()
