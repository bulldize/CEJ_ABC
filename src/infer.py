import argparse
import glob
import json
import os
import numpy as np
import torch
from scipy.ndimage import zoom
from monai.inferers import sliding_window_inference
from monai.transforms import DivisiblePad

from src.datasets.io import load_volume, save_volume
from src.datasets.transforms import normalize_intensity
from src.models.unet3d import UNet3D
from src.postprocess.skeleton import extract_curve
from src.postprocess.priors import compute_geometric_prior
from src.postprocess.stitch import stitch_curve_to_full, build_full_output
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["project"]["device"])

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
                x_t = padder(x_t)
            if cfg.get("infer", {}).get("use_sliding_window", False):
                roi_size = cfg["infer"].get("sw_roi_size", None)
                if roi_size is None:
                    roi_size = list(x_t.shape[-3:])
                sw_batch = int(cfg["infer"].get("sw_batch_size", 1))
                overlap = float(cfg["infer"].get("sw_overlap", 0.25))
                logits = sliding_window_inference(x_t, roi_size, sw_batch, model, overlap=overlap)
            else:
                logits = model(x_t)
            H_pred = torch.sigmoid(logits).cpu().numpy()[0, 0]
            if pad_divisor > 1:
                H_pred = H_pred[:A.shape[0], :A.shape[1], :A.shape[2]]

        if cfg["infer"]["use_prior_gating"]:
            R = compute_geometric_prior(
                A,
                T,
                spacing,
                sigma_z_mm=cfg["infer"]["prior_sigma_z_mm"],
                sigma_s_mm=cfg["infer"]["prior_sigma_s_mm"],
                delta_s_mm=cfg["infer"]["prior_delta_s_mm"],
                z_ignore_ratio=cfg["preprocess"]["z_ignore_ratio"],
                use_gradient=cfg["infer"]["prior_use_gradient"],
                gradient_weight=cfg["infer"]["prior_gradient_weight"],
            )
            H_pred = H_pred * R

        C_pred = extract_curve(H_pred, T, threshold=cfg["infer"]["threshold_theta"])

        out_dir = ensure_dir(os.path.join(infer_root, case_id, f"tooth_{tooth_id}"))
        save_volume(os.path.join(out_dir, "H_pred.nii.gz"), H_pred.astype(np.float32), affine=affine, spacing=spacing)
        save_volume(os.path.join(out_dir, "C_pred.nii.gz"), C_pred.astype(np.uint8), affine=affine, spacing=spacing)

        # handle resampled ROI for stitching
        curve_roi = C_pred
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
        raw_case_dir = os.path.join(cfg["data"]["raw_dir"], case_id)
        b_path = os.path.join(raw_case_dir, cfg["data"]["raw_b_name"])
        meta_path = os.path.join(raw_case_dir, cfg["data"]["raw_meta_name"])
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
