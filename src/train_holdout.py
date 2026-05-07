import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree
from monai.utils import set_determinism

from src.datasets.dataset import ToothDataset, write_supervised_audit
from src.infer import _fit_pred_curve_mask
from src.models.unet3d import UNet3D
from src.postprocess.skeleton import extract_curve, extract_curve_from_heatmap_peak
from src.train import build_loss, build_train_loader, now_iso
from src.utils.config import ensure_dir, get_device, load_config
from src.utils.log import get_logger


logger = get_logger("train_holdout")


def _distance_penalty(shape, spacing):
    spacing = np.asarray(spacing, dtype=np.float32)
    diag = np.maximum(np.asarray(shape, dtype=np.float32) - 1.0, 0.0) * spacing
    penalty = float(np.linalg.norm(diag))
    if penalty > 0.0 and np.isfinite(penalty):
        return penalty
    return float(np.max(spacing)) if spacing.size else 1.0


def _points_from_mask(mask):
    return np.argwhere(np.asarray(mask) > 0).astype(np.float32)


def _directed_distances(src_pts, dst_pts, spacing, shape):
    spacing = np.asarray(spacing, dtype=np.float32)
    if src_pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if dst_pts.shape[0] == 0:
        return np.full((src_pts.shape[0],), _distance_penalty(shape, spacing), dtype=np.float32)
    tree = cKDTree(dst_pts.astype(np.float32) * spacing)
    d, _ = tree.query(src_pts.astype(np.float32) * spacing, k=1)
    return d.astype(np.float32)


def _summarize_distances(distances):
    if not distances:
        return None, None
    arr = np.concatenate([d for d in distances if d.size > 0])
    if arr.size == 0:
        return None, None
    return float(np.mean(arr)), float(np.percentile(arr, 95))


def _component_stats(binary):
    binary = np.asarray(binary) > 0
    total = int(binary.sum())
    if total == 0:
        return 0, 0.0
    labels, n_labels = ndimage.label(binary)
    if n_labels <= 0:
        return 0, 0.0
    counts = np.bincount(labels.ravel())
    lcc = int(counts[1:].max()) if counts.size > 1 else 0
    return int(n_labels), float(lcc / max(1, total))


def _surface_band_from_tooth(tooth_mask, radius_vox=2):
    tooth = (tooth_mask > 0.5).float()
    if tooth.sum() == 0:
        return torch.ones_like(tooth)
    inv = 1.0 - tooth
    eroded = 1.0 - F.max_pool3d(inv, kernel_size=3, stride=1, padding=1)
    surface = (tooth - eroded).clamp_min(0.0)
    k = int(radius_vox) * 2 + 1
    band = F.max_pool3d(surface, kernel_size=k, stride=1, padding=int(radius_vox))
    return band.clamp(0.0, 1.0)


def _surface_constraint_loss(logits, x, weight, radius_vox):
    if weight <= 0.0 or x.shape[1] < 2:
        return logits.new_tensor(0.0)
    tooth = x[:, 1:2]
    band = _surface_band_from_tooth(tooth, radius_vox=radius_vox)
    prob = torch.sigmoid(logits)
    return float(weight) * torch.mean((prob ** 2) * (1.0 - band))


def train_one_epoch(model, loader, optimizer, device, loss_fn, cfg, global_step):
    model.train()
    total = 0.0
    steps = 0
    surface_weight = float(cfg["train"].get("surface_neighborhood_weight", 0.0))
    surface_radius = int(cfg["train"].get("surface_neighborhood_radius_vox", 2))
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss = loss + _surface_constraint_loss(logits, x, surface_weight, surface_radius)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
        global_step += 1
    return total / max(1, steps), global_step


def build_model(cfg, device):
    return UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        depth=cfg["model"]["depth"],
        num_res_units=cfg["model"].get("num_res_units", 2),
        norm=cfg["model"].get("norm", "batch"),
    ).to(device)


def load_pretrained(model, ckpt_path, strict=False):
    if not ckpt_path:
        return {"loaded": False, "path": None, "missing": None, "unexpected": None}
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"pretrained checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=bool(strict))
    logger.info(
        "loaded pretrained=%s strict=%s missing=%d unexpected=%d",
        ckpt_path,
        strict,
        len(missing),
        len(unexpected),
    )
    return {
        "loaded": True,
        "path": ckpt_path,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def _item_spacing(item):
    meta = item.get("y_meta_dict") or item.get("label_meta_dict") or {}
    pixdim = meta.get("pixdim")
    if pixdim is not None:
        arr = pixdim.detach().cpu().numpy() if torch.is_tensor(pixdim) else np.asarray(pixdim)
        if arr.size >= 4:
            return tuple(float(x) for x in arr.reshape(-1)[1:4])
    affine = meta.get("affine")
    if affine is not None:
        arr = affine.detach().cpu().numpy() if torch.is_tensor(affine) else np.asarray(affine)
        if arr.shape[-2:] == (4, 4):
            return tuple(float(np.linalg.norm(arr[:3, i])) for i in range(3))
    return (1.0, 1.0, 1.0)


def _case_suffix(case_id):
    text = str(case_id)
    digits = ""
    for ch in reversed(text):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return digits or text


def _parse_hard_cases(specs):
    hard = set()
    for spec in specs or []:
        case, teeth = str(spec).split("-", 1)
        for tooth in teeth.replace(",", "/").split("/"):
            tooth = tooth.strip()
            if tooth:
                hard.add((_case_suffix(case), tooth))
    return hard


def evaluate_holdout(model, dataset, device, cfg, epoch, out_dir=None, save_prefix=None):
    model.eval()
    holdout_cfg = cfg.get("holdout_eval", {})
    infer_cfg = cfg.get("infer", {})
    threshold = float(infer_cfg.get("threshold_theta", holdout_cfg.get("threshold_theta", 0.3)))
    keep_lcc = bool(infer_cfg.get("keep_lcc_for_curve", False))
    fit_curve = bool(infer_cfg.get("fit_pred_curve", True))
    constrain_to_tooth = bool(infer_cfg.get("constrain_curve_to_tooth_mask", False))
    gt_peak = float(holdout_cfg.get("gt_peak_threshold", 0.95))
    wrap_tau = float(holdout_cfg.get("wrap_tau_mm", 1.0))
    hard_cases = _parse_hard_cases(holdout_cfg.get("hard_cases", []))

    rows = []
    hard_rows = []
    all_sym = []
    gt_to_pred_all = []
    no_curve_count = 0
    h_max_values = []
    h_p99_values = []
    vox03_values = []
    cc_values = []
    lcc_ratio_values = []
    coverage_values = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            x = item["x"].unsqueeze(0).to(device)
            y = item["y"].detach().cpu().numpy()[0]
            tooth = item["x"][1].detach().cpu().numpy() if item["x"].shape[0] > 1 else np.ones_like(y)
            spacing = _item_spacing(item)
            logits = model(x)
            h_pred = torch.sigmoid(logits).detach().cpu().numpy()[0, 0]

            binary = h_pred >= threshold
            if constrain_to_tooth:
                binary = np.logical_and(binary, tooth > 0)
            cc_count, lcc_ratio = _component_stats(binary)

            curve_mask = extract_curve(
                h_pred,
                tooth if constrain_to_tooth else np.ones_like(tooth, dtype=np.uint8),
                threshold=threshold,
                keep_lcc=keep_lcc,
            )
            if fit_curve:
                curve_mask, _ = _fit_pred_curve_mask(
                    curve_mask,
                    spacing=spacing,
                    step_mm=float(holdout_cfg.get("fit_pred_curve_step_mm", cfg.get("preprocess", {}).get("dense_sample_step_mm", 0.2))),
                    closed=bool(holdout_cfg.get("fit_pred_curve_closed", cfg.get("preprocess", {}).get("curve_closed", True))),
                    smooth=float(holdout_cfg.get("fit_pred_curve_smooth", cfg.get("preprocess", {}).get("curve_smooth", 0.0))),
                    min_points=int(holdout_cfg.get("fit_pred_curve_min_points", 8)),
                )

            gt_curve = extract_curve_from_heatmap_peak(y, peak_threshold=min(gt_peak, float(np.nanmax(y)) if y.size else gt_peak), keep_lcc=False)
            if gt_curve.sum() == 0 and y.size:
                rel_thr = float(np.nanmax(y)) * float(holdout_cfg.get("gt_rel_threshold", 0.95))
                gt_curve = (y >= rel_thr).astype(np.uint8)

            pred_pts = _points_from_mask(curve_mask)
            gt_pts = _points_from_mask(gt_curve)
            no_curve = int(pred_pts.shape[0] == 0)
            no_curve_count += no_curve

            pred_to_gt = _directed_distances(pred_pts, gt_pts, spacing, y.shape)
            gt_to_pred = _directed_distances(gt_pts, pred_pts, spacing, y.shape)
            sym = np.concatenate([pred_to_gt, gt_to_pred]).astype(np.float32)
            if sym.size > 0:
                all_sym.append(sym)
            if gt_to_pred.size > 0:
                gt_to_pred_all.append(gt_to_pred)
            coverage = float(np.mean(gt_to_pred <= wrap_tau)) if gt_to_pred.size else 0.0

            h_max = float(np.max(h_pred)) if h_pred.size else 0.0
            h_p99 = float(np.percentile(h_pred, 99)) if h_pred.size else 0.0
            vox03 = int(binary.sum())
            case_id = str(dataset.items[idx].get("case_id"))
            tooth_id = str(dataset.items[idx].get("tooth_id"))
            row = {
                "epoch": epoch,
                "case_id": case_id,
                "tooth_id": tooth_id,
                "is_hard_case": int((_case_suffix(case_id), tooth_id) in hard_cases),
                "sym_mean": float(np.mean(sym)) if sym.size else None,
                "sym_p95": float(np.percentile(sym, 95)) if sym.size else None,
                "gt_to_pred_mean": float(np.mean(gt_to_pred)) if gt_to_pred.size else None,
                "pred_to_gt_mean": float(np.mean(pred_to_gt)) if pred_to_gt.size else None,
                "no_curve": no_curve,
                "h_pred_max": h_max,
                "h_pred_p99": h_p99,
                "vox03": vox03,
                "cc_count": cc_count,
                "lcc_ratio": lcc_ratio,
                "wrap_coverage": coverage,
                "pred_curve_voxels": int(pred_pts.shape[0]),
                "gt_curve_voxels": int(gt_pts.shape[0]),
            }
            rows.append(row)
            if row["is_hard_case"]:
                hard_rows.append(row)
            h_max_values.append(h_max)
            h_p99_values.append(h_p99)
            vox03_values.append(vox03)
            cc_values.append(cc_count)
            lcc_ratio_values.append(lcc_ratio)
            coverage_values.append(coverage)

    sym_mean, sym_p95 = _summarize_distances(all_sym)
    gt_mean, gt_p95 = _summarize_distances(gt_to_pred_all)
    metrics = {
        "epoch": int(epoch),
        "holdout_tooth_count": int(len(rows)),
        "sym_mean": sym_mean,
        "sym_p95": sym_p95,
        "gt_to_pred_mean": gt_mean,
        "gt_to_pred_p95": gt_p95,
        "no_curve_count": int(no_curve_count),
        "h_pred_max": float(np.max(h_max_values)) if h_max_values else 0.0,
        "h_pred_p99": float(np.mean(h_p99_values)) if h_p99_values else 0.0,
        "vox03": float(np.mean(vox03_values)) if vox03_values else 0.0,
        "vox03_total": int(np.sum(vox03_values)) if vox03_values else 0,
        "cc_count": float(np.mean(cc_values)) if cc_values else 0.0,
        "lcc_ratio": float(np.mean(lcc_ratio_values)) if lcc_ratio_values else 0.0,
        "wrap_coverage": float(np.mean(coverage_values)) if coverage_values else 0.0,
        "hard_case_count": int(len(hard_rows)),
    }

    if out_dir and save_prefix:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        per_tooth_path = out_path / f"{save_prefix}_per_tooth.csv"
        hard_path = out_path / f"{save_prefix}_hard_cases.csv"
        fieldnames = list(rows[0].keys()) if rows else []
        if fieldnames:
            with per_tooth_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            with hard_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(hard_rows)
        with (out_path / f"{save_prefix}_summary.json").open("w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics, rows, hard_rows


def score_tuple(metrics):
    sym_p95 = metrics.get("sym_p95")
    sym_p95 = float(sym_p95) if sym_p95 is not None else float("inf")
    no_curve = int(metrics.get("no_curve_count", 10 ** 9))
    wrap = float(metrics.get("wrap_coverage", 0.0) or 0.0)
    return (sym_p95, no_curve, -wrap)


def is_better(metrics, best_metrics):
    if best_metrics is None:
        return True
    return score_tuple(metrics) < score_tuple(best_metrics)


def save_checkpoint(path, model, optimizer, epoch, global_step, metrics, cfg):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "holdout_metrics": metrics,
            "config": cfg,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_determinism(seed=cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])

    pad_divisor = 2 ** max(0, int(cfg["model"]["depth"]) - 1)
    train_ds = ToothDataset(
        cfg["data"]["processed_dir"],
        processed_format=cfg["data"]["processed_format"],
        use_mask_channel=cfg["data"]["use_tooth_mask_channel"],
        clip_percentiles=cfg["preprocess"]["intensity_clip_percentiles"],
        norm_mode=cfg["preprocess"]["intensity_norm"],
        cache_rate=cfg["train"].get("cache_rate", 0.0),
        pad_divisor=pad_divisor,
    )
    holdout_ds = ToothDataset(
        cfg["data"]["holdout_processed_dir"],
        processed_format=cfg["data"]["processed_format"],
        use_mask_channel=cfg["data"]["use_tooth_mask_channel"],
        clip_percentiles=cfg["preprocess"]["intensity_clip_percentiles"],
        norm_mode=cfg["preprocess"]["intensity_norm"],
        cache_rate=cfg["train"].get("holdout_cache_rate", 0.0),
        pad_divisor=pad_divisor,
    )
    if len(train_ds) == 0:
        raise RuntimeError("no labeled training teeth available after audit")
    if len(holdout_ds) == 0:
        raise RuntimeError("no labeled holdout teeth available after audit")

    out_dir = ensure_dir(os.path.join(cfg["data"]["output_dir"], "train"))
    ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))
    analysis_dir = ensure_dir(os.path.join(out_dir, "holdout_analysis"))
    audit_paths = {
        "train": write_supervised_audit(train_ds.audit, os.path.join(out_dir, "audit_train")),
        "holdout": write_supervised_audit(holdout_ds.audit, os.path.join(out_dir, "audit_holdout")),
    }

    loader, sampler_weights = build_train_loader(train_ds, cfg)
    model = build_model(cfg, device)
    pretrained_info = load_pretrained(
        model,
        cfg["train"].get("pretrained_ckpt"),
        strict=bool(cfg["train"].get("pretrained_strict", False)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = build_loss(cfg)

    max_epochs = int(cfg["train"].get("max_epochs", cfg["train"].get("epochs", 50)))
    patience = int(cfg["train"].get("early_stopping_patience", 5))
    best_min_epoch = int(cfg["train"].get("best_min_epoch", 1))
    metrics_path = os.path.join(out_dir, "metrics.csv")
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_metrics = None
    best_epoch = None
    epochs_without_improve = 0
    global_step = 0
    started_at = now_iso()

    fieldnames = [
        "epoch",
        "global_step",
        "train_loss",
        "holdout_sym_mean",
        "holdout_sym_p95",
        "holdout_gt_to_pred_mean",
        "holdout_gt_to_pred_p95",
        "no_curve_count",
        "h_pred_max",
        "h_pred_p99",
        "vox03",
        "vox03_total",
        "cc_count",
        "lcc_ratio",
        "wrap_coverage",
        "is_best",
    ]
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(max_epochs):
            train_loss, global_step = train_one_epoch(model, loader, optimizer, device, loss_fn, cfg, global_step)
            holdout_metrics, _, _ = evaluate_holdout(model, holdout_ds, device, cfg, epoch)
            improved = epoch >= best_min_epoch and is_better(holdout_metrics, best_metrics)
            if improved:
                best_metrics = dict(holdout_metrics)
                best_epoch = epoch
                epochs_without_improve = 0
                save_checkpoint(best_path, model, optimizer, epoch, global_step, holdout_metrics, cfg)
                evaluate_holdout(model, holdout_ds, device, cfg, epoch, out_dir=analysis_dir, save_prefix="best")
            else:
                epochs_without_improve += 1
            save_checkpoint(last_path, model, optimizer, epoch, global_step, holdout_metrics, cfg)

            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "holdout_sym_mean": holdout_metrics["sym_mean"],
                "holdout_sym_p95": holdout_metrics["sym_p95"],
                "holdout_gt_to_pred_mean": holdout_metrics["gt_to_pred_mean"],
                "holdout_gt_to_pred_p95": holdout_metrics["gt_to_pred_p95"],
                "no_curve_count": holdout_metrics["no_curve_count"],
                "h_pred_max": holdout_metrics["h_pred_max"],
                "h_pred_p99": holdout_metrics["h_pred_p99"],
                "vox03": holdout_metrics["vox03"],
                "vox03_total": holdout_metrics["vox03_total"],
                "cc_count": holdout_metrics["cc_count"],
                "lcc_ratio": holdout_metrics["lcc_ratio"],
                "wrap_coverage": holdout_metrics["wrap_coverage"],
                "is_best": int(improved),
            }
            writer.writerow(row)
            f.flush()
            logger.info(
                "epoch=%d loss=%.6f sym_p95=%s no_curve=%d wrap=%.4f best_epoch=%s patience=%d/%d",
                epoch,
                train_loss,
                holdout_metrics["sym_p95"],
                holdout_metrics["no_curve_count"],
                holdout_metrics["wrap_coverage"],
                best_epoch,
                epochs_without_improve,
                patience,
            )
            if epoch >= best_min_epoch and epochs_without_improve >= patience:
                logger.info("early stopping at epoch=%d best_epoch=%s", epoch, best_epoch)
                break

    if best_metrics is None:
        raise RuntimeError(f"no best checkpoint was selected; best_min_epoch={best_min_epoch} max_epochs={max_epochs}")

    manifest = {
        "started_at": started_at,
        "finished_at": now_iso(),
        "config_path": os.path.abspath(args.config),
        "output_dir": out_dir,
        "metrics_csv": metrics_path,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "score_order": ["sym_p95 asc", "no_curve_count asc", "wrap_coverage desc"],
        "best_min_epoch": best_min_epoch,
        "pretrained": pretrained_info,
        "dataset": {
            "train": {
                "total_tooth_dirs": train_ds.audit["total_tooth_dirs"],
                "used_labeled_tooth_dirs": train_ds.audit["used_labeled_tooth_dirs"],
                "skipped_unlabeled_tooth_dirs": train_ds.audit["skipped_unlabeled_tooth_dirs"],
                "invalid_tooth_dirs": train_ds.audit.get("invalid_tooth_dirs", 0),
            },
            "holdout": {
                "total_tooth_dirs": holdout_ds.audit["total_tooth_dirs"],
                "used_labeled_tooth_dirs": holdout_ds.audit["used_labeled_tooth_dirs"],
                "skipped_unlabeled_tooth_dirs": holdout_ds.audit["skipped_unlabeled_tooth_dirs"],
                "invalid_tooth_dirs": holdout_ds.audit.get("invalid_tooth_dirs", 0),
            },
        },
        "audit_paths": audit_paths,
        "sampler": {
            "weighted_sampler": bool(cfg["train"].get("weighted_sampler", False)),
            "weight_count": int(len(sampler_weights)) if sampler_weights is not None else 0,
            "hard_weight_count": int((sampler_weights > 1.0).sum().item()) if sampler_weights is not None else 0,
        },
    }
    with open(os.path.join(out_dir, "train_manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
