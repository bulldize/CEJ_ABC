import argparse
import glob
import json
import os
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, zoom
from monai.inferers import sliding_window_inference
from monai.transforms import DivisiblePad

from src.datasets.heatmap import fit_curve_and_sample
from src.datasets.io import load_volume, save_volume
from src.datasets.raw_cases import collect_raw_case_map
from src.datasets.transforms import normalize_intensity
from src.models.unet3d import UNet3D
from src.postprocess.skeleton import extract_curve
from src.postprocess.priors import compute_geometric_prior
from src.postprocess.stitch import stitch_curve_to_full, build_full_output
from src.utils.geometry import tooth_surface
from src.utils.config import load_config, ensure_dir, get_device
from src.utils.log import get_logger

logger = get_logger("infer")


def map_pulp_to_tooth(B):
    B = B.copy()
    mask = B >= 100
    B[mask] = B[mask] % 100
    return B


def resize_to_shape(vol, shape):
    out = vol
    if out.shape != tuple(shape):
        scale = [shape[i] / out.shape[i] for i in range(3)]
        out = zoom(out, scale, order=0)
        # pad/crop to exact
        out = out[:shape[0], :shape[1], :shape[2]]
        pad = [shape[i] - out.shape[i] for i in range(3)]
        if any(p > 0 for p in pad):
            out = np.pad(out, [(0, pad[0]), (0, pad[1]), (0, pad[2])], mode="constant")
    return out


def _rasterize_polyline_to_mask(mask, points_xyz, close_loop=False):
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return

    shape = np.asarray(mask.shape, dtype=np.int32)

    def _mark(p):
        q = np.round(p).astype(np.int32)
        if np.all(q >= 0) and np.all(q < shape):
            mask[q[0], q[1], q[2]] = 1

    _mark(pts[0])
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        n = int(np.ceil(np.max(np.abs(p1 - p0)))) + 1
        n = max(2, n)
        for p in np.linspace(p0, p1, n):
            _mark(p)

    if close_loop and pts.shape[0] > 2:
        p0 = pts[-1]
        p1 = pts[0]
        n = int(np.ceil(np.max(np.abs(p1 - p0)))) + 1
        n = max(2, n)
        for p in np.linspace(p0, p1, n):
            _mark(p)


def _fit_pred_curve_mask(
    curve_mask,
    spacing,
    step_mm,
    closed,
    smooth,
    min_points,
):
    pts = np.argwhere(curve_mask > 0).astype(np.float32)
    if pts.shape[0] < max(2, int(min_points)):
        return curve_mask.astype(np.uint8), np.zeros((0, 3), dtype=np.float32)

    try:
        dense_pts = fit_curve_and_sample(
            pts,
            spacing=np.asarray(spacing, dtype=np.float32),
            step_mm=float(step_mm),
            closed=bool(closed),
            smooth=float(smooth),
        )
    except Exception:
        return curve_mask.astype(np.uint8), np.zeros((0, 3), dtype=np.float32)

    if dense_pts.ndim != 2 or dense_pts.shape[0] < 2:
        return curve_mask.astype(np.uint8), np.zeros((0, 3), dtype=np.float32)

    fit_mask = np.zeros_like(curve_mask, dtype=np.uint8)
    _rasterize_polyline_to_mask(fit_mask, dense_pts, close_loop=bool(closed))
    if int(fit_mask.sum()) == 0:
        return curve_mask.astype(np.uint8), dense_pts.astype(np.float32)
    return fit_mask.astype(np.uint8), dense_pts.astype(np.float32)


def _project_curve_to_tooth_surface(curve_mask, tooth_mask):
    curve = np.asarray(curve_mask) > 0
    tooth = np.asarray(tooth_mask) > 0
    if curve.sum() == 0 or tooth.sum() == 0:
        return np.asarray(curve_mask, dtype=np.uint8)

    surface = tooth_surface(tooth.astype(np.uint8))
    if surface.sum() == 0:
        return np.asarray(curve_mask, dtype=np.uint8)

    # For each curve voxel, snap to the nearest tooth-surface voxel.
    _, nearest = distance_transform_edt(~surface, return_indices=True)
    out = np.zeros_like(curve_mask, dtype=np.uint8)
    sx = nearest[0][curve]
    sy = nearest[1][curve]
    sz = nearest[2][curve]
    out[sx, sy, sz] = 1
    return out.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["project"]["device"])
    infer_cfg = cfg.get("infer", {})

    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        depth=cfg["model"]["depth"],
        num_res_units=cfg["model"].get("num_res_units", 2),
        norm=cfg["model"].get("norm", "batch"),
    ).to(device)

    ckpt_path = os.path.join(cfg["data"]["output_dir"], "train", "checkpoints", "last.pt")
    if not os.path.exists(ckpt_path):
        logger.warning("checkpoint not found: %s", ckpt_path)
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    processed_dir = cfg["data"]["processed_dir"]
    tooth_dirs = sorted(glob.glob(os.path.join(processed_dir, "*", "tooth_*")))
    if not tooth_dirs:
        logger.warning("no processed teeth found")
        return

    infer_root = ensure_dir(os.path.join(cfg["data"]["output_dir"], "infer"))
    raw_case_map = collect_raw_case_map(cfg["data"])

    # collect per-case curves/heatmaps for stitching
    curves_by_case = {}
    heatmaps_by_case = {}

    for tdir in tooth_dirs:
        roi_meta_path = os.path.join(tdir, "roi_meta.json")
        with open(roi_meta_path, "r") as f:
            roi_meta = json.load(f)
        case_id = roi_meta["case_id"]
        tooth_id = roi_meta["tooth_id"]

        fmt = cfg["data"]["processed_format"]
        a_path = os.path.join(tdir, f"A_t.{fmt}")
        t_path = os.path.join(tdir, f"T_t.{fmt}")

        A, spacing, affine = load_volume(a_path, dtype=np.float32)
        T, _, _ = load_volume(t_path, dtype=np.uint8)

        # if A already normalized, this is idempotent
        A = normalize_intensity(
            A,
            cfg["preprocess"]["intensity_clip_percentiles"],
            cfg["preprocess"]["intensity_norm"],
        )

        if cfg["data"]["use_tooth_mask_channel"]:
            x = np.stack([A, T.astype(np.float32)], axis=0)
        else:
            x = A[None, ...]

        with torch.no_grad():
            x_t = torch.from_numpy(x[None, ...]).to(device)
            pad_divisor = 2 ** max(0, int(cfg["model"]["depth"]) - 1)
            if pad_divisor > 1:
                padder = DivisiblePad(k=pad_divisor, method="end")
                x_pad = padder(x_t[0])
                x_t = x_pad.unsqueeze(0)
            if infer_cfg.get("use_sliding_window", False):
                roi_size = infer_cfg.get("sw_roi_size", None)
                if roi_size is None:
                    roi_size = list(x_t.shape[-3:])
                sw_batch = int(infer_cfg.get("sw_batch_size", 1))
                overlap = float(infer_cfg.get("sw_overlap", 0.25))
                logits = sliding_window_inference(x_t, roi_size, sw_batch, model, overlap=overlap)
            else:
                logits = model(x_t)
            H_pred = torch.sigmoid(logits).cpu().numpy()[0, 0]
            if pad_divisor > 1:
                H_pred = H_pred[:A.shape[0], :A.shape[1], :A.shape[2]]

        if bool(infer_cfg.get("use_prior_gating", False)):
            R = compute_geometric_prior(
                A,
                T,
                spacing,
                sigma_z_mm=float(infer_cfg.get("prior_sigma_z_mm", 2.0)),
                sigma_s_mm=float(infer_cfg.get("prior_sigma_s_mm", 1.0)),
                delta_s_mm=float(infer_cfg.get("prior_delta_s_mm", 1.0)),
                z_ignore_ratio=cfg["preprocess"]["z_ignore_ratio"],
                use_gradient=bool(infer_cfg.get("prior_use_gradient", False)),
                gradient_weight=float(infer_cfg.get("prior_gradient_weight", 1.0)),
            )
            H_pred = H_pred * R

        # Keep all skeleton components by default for prediction. Keeping only the
        # largest connected component tends to collapse CEJ to a few points.
        # Optionally disable tooth-mask constraint for debugging/visual analysis.
        constrain_to_tooth = bool(infer_cfg.get("constrain_curve_to_tooth_mask", False))
        curve_tooth_mask = T if constrain_to_tooth else np.ones_like(T, dtype=np.uint8)
        C_pred_raw = extract_curve(
            H_pred,
            curve_tooth_mask,
            threshold=float(infer_cfg.get("threshold_theta", 0.3)),
            keep_lcc=bool(infer_cfg.get("keep_lcc_for_curve", False)),
        )
        C_pred_fit = C_pred_raw
        pred_dense_pts = np.zeros((0, 3), dtype=np.float32)
        if bool(infer_cfg.get("fit_pred_curve", True)):
            C_pred_fit, pred_dense_pts = _fit_pred_curve_mask(
                C_pred_raw,
                spacing=spacing,
                step_mm=float(
                    infer_cfg.get(
                        "fit_pred_curve_step_mm",
                        cfg.get("preprocess", {}).get("dense_sample_step_mm", 0.2),
                    )
                ),
                closed=bool(
                    infer_cfg.get(
                        "fit_pred_curve_closed",
                        cfg.get("preprocess", {}).get("curve_closed", True),
                    )
                ),
                smooth=float(
                    infer_cfg.get(
                        "fit_pred_curve_smooth",
                        cfg.get("preprocess", {}).get("curve_smooth", 0.0),
                    )
                ),
                min_points=int(infer_cfg.get("fit_pred_curve_min_points", 8)),
            )
        if bool(infer_cfg.get("constrain_curve_to_tooth_surface", False)):
            C_pred_fit = _project_curve_to_tooth_surface(C_pred_fit, T)

        out_dir = ensure_dir(os.path.join(infer_root, case_id, f"tooth_{tooth_id}"))
        save_volume(os.path.join(out_dir, "H_pred.nii.gz"), H_pred.astype(np.float32), affine=affine, spacing=spacing)
        save_volume(os.path.join(out_dir, "C_pred.nii.gz"), C_pred_raw.astype(np.uint8), affine=affine, spacing=spacing)
        save_volume(os.path.join(out_dir, "C_pred_fit.nii.gz"), C_pred_fit.astype(np.uint8), affine=affine, spacing=spacing)
        if bool(infer_cfg.get("constrain_curve_to_tooth_surface", False)):
            save_volume(
                os.path.join(out_dir, "C_pred_surface.nii.gz"),
                C_pred_fit.astype(np.uint8),
                affine=affine,
                spacing=spacing,
            )
        np.save(os.path.join(out_dir, "curve_pred_dense_points.npy"), pred_dense_pts.astype(np.float32))

        # handle resampled ROI for stitching
        curve_roi = C_pred_fit
        heat_roi = H_pred
        if roi_meta.get("resampled", False):
            scale = np.array(roi_meta.get("resample_scale", [1.0, 1.0, 1.0]), dtype=np.float32)
            inv_scale = 1.0 / scale
            curve_roi = zoom(curve_roi, inv_scale, order=0)
            curve_roi = resize_to_shape(curve_roi, roi_meta["roi_shape_full"])
            heat_roi = zoom(heat_roi, inv_scale, order=1)
            heat_roi = resize_to_shape(heat_roi, roi_meta["roi_shape_full"])

        curves_by_case.setdefault(case_id, []).append((curve_roi, roi_meta))
        heatmaps_by_case.setdefault(case_id, []).append((heat_roi, roi_meta))

        logger.info("inferred case=%s tooth=%s", case_id, tooth_id)

    # stitch to full
    for case_id, curves in curves_by_case.items():
        case_rec = raw_case_map.get(case_id, None)
        if case_rec is None:
            logger.warning("raw case record not found for case=%s, skip stitching", case_id)
            continue
        b_path = case_rec["b_path"]
        meta_path = case_rec.get("meta_path", None)
        if meta_path and not os.path.exists(meta_path):
            meta_path = None
        if not os.path.exists(b_path):
            logger.warning("raw label not found for case=%s, skip stitching (%s)", case_id, b_path)
            continue
        B_full, spacing, affine = load_volume(b_path, meta_path=meta_path, dtype=np.int16)
        B_full = map_pulp_to_tooth(B_full)

        # full heatmap (max over teeth)
        H_full = np.zeros(B_full.shape, dtype=np.float32)
        for heat_roi, roi_meta in heatmaps_by_case.get(case_id, []):
            origin = roi_meta["roi_origin_in_full"]
            x0, y0, z0 = origin
            x1 = x0 + heat_roi.shape[0]
            y1 = y0 + heat_roi.shape[1]
            z1 = z0 + heat_roi.shape[2]
            H_full[x0:x1, y0:y1, z0:z1] = np.maximum(
                H_full[x0:x1, y0:y1, z0:z1], heat_roi
            )

        curves_full = []
        for curve_roi, roi_meta in curves:
            origin = roi_meta["roi_origin_in_full"]
            curve_full = stitch_curve_to_full(B_full.shape, curve_roi, origin)
            curves_full.append(curve_full)

        Y = build_full_output(B_full, curves_full)
        out_case_dir = ensure_dir(os.path.join(infer_root, case_id))
        save_volume(os.path.join(out_case_dir, "Y_pred.nii.gz"), Y.astype(np.uint8), affine=affine, spacing=spacing)
        save_volume(os.path.join(out_case_dir, "H_pred.nii.gz"), H_full.astype(np.float32), affine=affine, spacing=spacing)
        logger.info("stitched case=%s", case_id)


if __name__ == "__main__":
    main()
