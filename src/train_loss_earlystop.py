import argparse
import csv
import html
import json
import os

import torch
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from src.datasets.dataset import ToothDataset, write_supervised_audit
from src.models.unet3d import UNet3D
from src.train import build_loss, build_train_loader, now_iso
from src.train_holdout import evaluate_holdout
from src.utils.config import ensure_dir, get_device, load_config
from src.utils.log import get_logger


logger = get_logger("train_loss_earlystop")


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
        return {"loaded": False, "path": None, "missing": [], "unexpected": []}
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"pretrained checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    skipped_mismatch = []
    loaded_tensor_count = len(state)
    if strict:
        missing, unexpected = model.load_state_dict(state, strict=True)
    else:
        model_state = model.state_dict()
        compatible_state = {}
        unexpected = []
        for key, value in state.items():
            target = model_state.get(key)
            if target is None:
                unexpected.append(key)
                continue
            if getattr(value, "shape", None) != target.shape:
                skipped_mismatch.append(
                    {
                        "key": key,
                        "checkpoint_shape": list(value.shape),
                        "model_shape": list(target.shape),
                    }
                )
                continue
            compatible_state[key] = value
        loaded_tensor_count = len(compatible_state)
        missing, unexpected_loaded = model.load_state_dict(compatible_state, strict=False)
        unexpected = list(unexpected) + list(unexpected_loaded)
    logger.info(
        "loaded pretrained=%s strict=%s loaded_tensors=%d missing=%d unexpected=%d skipped_mismatch=%d",
        ckpt_path,
        strict,
        loaded_tensor_count,
        len(missing),
        len(unexpected),
        len(skipped_mismatch),
    )
    return {
        "loaded": loaded_tensor_count > 0,
        "path": ckpt_path,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "loaded_tensor_count": int(loaded_tensor_count),
        "skipped_mismatch": skipped_mismatch,
    }


def surface_band_from_tooth(tooth_mask, radius_vox=2):
    tooth = (tooth_mask > 0.5).float()
    if tooth.sum() == 0:
        return torch.ones_like(tooth)
    inv = 1.0 - tooth
    eroded = 1.0 - torch.nn.functional.max_pool3d(inv, kernel_size=3, stride=1, padding=1)
    surface = (tooth - eroded).clamp_min(0.0)
    k = int(radius_vox) * 2 + 1
    band = torch.nn.functional.max_pool3d(surface, kernel_size=k, stride=1, padding=int(radius_vox))
    return band.clamp(0.0, 1.0)


def surface_constraint_loss(logits, x, cfg):
    weight = float(cfg["train"].get("surface_neighborhood_weight", 0.0))
    if weight <= 0.0 or x.shape[1] < 2:
        return logits.new_tensor(0.0)
    radius = int(cfg["train"].get("surface_neighborhood_radius_vox", 2))
    band = surface_band_from_tooth(x[:, 1:2], radius_vox=radius)
    prob = torch.sigmoid(logits)
    return weight * torch.mean((prob ** 2) * (1.0 - band))


def batch_loss(model, batch, device, loss_fn, cfg):
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    logits = model(x)
    loss = loss_fn(logits, y)
    return loss + surface_constraint_loss(logits, x, cfg)


def train_one_epoch(model, loader, optimizer, device, loss_fn, cfg, global_step):
    model.train()
    total = 0.0
    steps = 0
    for batch in loader:
        optimizer.zero_grad()
        loss = batch_loss(model, batch, device, loss_fn, cfg)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
        global_step += 1
    return total / max(1, steps), global_step


def evaluate_loss(model, loader, device, loss_fn, cfg):
    model.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            loss = batch_loss(model, batch, device, loss_fn, cfg)
            total += float(loss.item())
            steps += 1
    return total / max(1, steps)


def evaluate_monitor_loss(model, loader, device, loss_fn):
    model.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total += float(loss.item())
            steps += 1
    return total / max(1, steps)


def save_checkpoint(path, model, optimizer, epoch, global_step, train_loss, monitor_metrics, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": float(train_loss),
            "holdout_loss": float(monitor_metrics.get("holdout_loss", float("inf"))),
            "holdout_metrics": monitor_metrics,
            "config": cfg,
        },
        path,
    )


def _remove_file(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def update_topk_checkpoints(
    records,
    topk_dir,
    top_k,
    model,
    optimizer,
    epoch,
    global_step,
    train_loss,
    monitor_metrics,
    cfg,
):
    if top_k <= 0:
        return records
    score = float(monitor_metrics["holdout_composite_score"])
    worst_score = max([r["holdout_composite_score"] for r in records], default=float("inf"))
    if len(records) >= top_k and score >= worst_score:
        return records
    filename = f"epoch_{int(epoch):03d}_score_{score:.6f}.pt"
    path = os.path.join(topk_dir, filename)
    save_checkpoint(path, model, optimizer, epoch, global_step, train_loss, monitor_metrics, cfg)
    records = list(records) + [
        {
            "epoch": int(epoch),
            "holdout_composite_score": score,
            "holdout_loss": float(monitor_metrics.get("holdout_loss", float("inf"))),
            "holdout_sym_p95": monitor_metrics.get("holdout_sym_p95"),
            "path": path,
        }
    ]
    records.sort(key=lambda r: (r["holdout_composite_score"], r["epoch"]))
    for rec in records[top_k:]:
        _remove_file(rec["path"])
    return records[:top_k]


def _finite_float(value, default):
    if value is None:
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if value != value or value in (float("inf"), float("-inf")):
        return float(default)
    return value


def compute_bad_rate(rows, bad_sym_p95_mm):
    if not rows:
        return 1.0
    bad = 0
    threshold = float(bad_sym_p95_mm)
    for row in rows:
        sym_p95 = row.get("sym_p95")
        is_bad = int(row.get("no_curve", 0) or 0) > 0
        if sym_p95 is None:
            is_bad = True
        else:
            is_bad = is_bad or _finite_float(sym_p95, threshold + 1.0) > threshold
        bad += int(is_bad)
    return float(bad / max(1, len(rows)))


def expected_hard_case_count(cfg, observed_count):
    specs = cfg.get("holdout", {}).get("hard_cases", [])
    count = 0
    for spec in specs:
        if not spec:
            continue
        text = str(spec)
        if "-" not in text:
            count += 1
            continue
        _, teeth = text.split("-", 1)
        count += len([t for t in teeth.split("/") if t])
    return int(count or observed_count or 1)


def compute_hard_case_viewer_component(hard_rows, cfg, targets):
    target_count = max(1, int(targets.get("hard_case_count", expected_hard_case_count(cfg, len(hard_rows)))))
    hard_sym_target = max(
        1e-6,
        _finite_float(targets.get("hard_sym_p95_mm", targets.get("bad_sym_p95_mm", 5.0)), 5.0),
    )
    hard_bad_target = _finite_float(
        targets.get("hard_bad_sym_p95_mm", targets.get("bad_sym_p95_mm", 5.0)),
        hard_sym_target,
    )
    finite_sym = [
        _finite_float(row.get("sym_p95"), hard_sym_target * 10.0)
        for row in hard_rows
        if row.get("sym_p95") is not None
    ]
    hard_sym_p95 = float(max(finite_sym)) if finite_sym else hard_sym_target * 10.0
    hard_no_curve_count = sum(int(row.get("no_curve", 0) or 0) for row in hard_rows)
    hard_bad_rate = compute_bad_rate(hard_rows, hard_bad_target)
    missing_rate = max(0.0, float(target_count - len(hard_rows)) / float(target_count))
    no_curve_rate = float(hard_no_curve_count) / float(max(1, len(hard_rows)))
    component = (
        0.40 * min(hard_sym_p95 / hard_sym_target, 10.0)
        + 0.25 * no_curve_rate
        + 0.25 * hard_bad_rate
        + 0.10 * missing_rate
    )
    details = {
        "hard_case_viewer": float(component),
        "hard_case_sym_p95": float(hard_sym_p95),
        "hard_case_no_curve_count": int(hard_no_curve_count),
        "hard_case_bad_rate": float(hard_bad_rate),
        "hard_case_missing_rate": float(missing_rate),
        "hard_case_expected_count": int(target_count),
    }
    return float(component), details


def compute_composite_score(holdout_loss, holdout_metrics, holdout_rows, cfg):
    score_cfg = cfg["train"].get("composite_score", {})
    weights = score_cfg.get("weights", {})
    targets = score_cfg.get("targets", {})
    sym_p95 = _finite_float(holdout_metrics.get("sym_p95"), targets.get("sym_p95_mm", 5.0))
    manual_p95 = _finite_float(
        holdout_metrics.get("manual_point_p95"),
        targets.get("manual_point_p95_mm", targets.get("sym_p95_mm", 5.0)),
    )
    manual_sr1 = _finite_float(holdout_metrics.get("manual_point_sr1"), 0.0)
    no_curve_count = int(holdout_metrics.get("no_curve_count", 0) or 0)
    bad_rate = compute_bad_rate(holdout_rows, targets.get("bad_sym_p95_mm", 5.0))
    wrap_coverage = _finite_float(holdout_metrics.get("wrap_coverage"), 0.0)
    cc_count = _finite_float(holdout_metrics.get("cc_count"), targets.get("cc_count", 4.0))
    vox03 = _finite_float(holdout_metrics.get("vox03"), 0.0)
    vox03_target = max(1.0, _finite_float(targets.get("vox03", 100.0), 100.0))
    sym_target = max(1e-6, _finite_float(targets.get("sym_p95_mm", 5.0), 5.0))
    manual_target = max(
        1e-6,
        _finite_float(targets.get("manual_point_p95_mm", targets.get("sym_p95_mm", 5.0)), 5.0),
    )
    cc_target = max(1.0, _finite_float(targets.get("cc_count", 4.0), 4.0))
    loss_target = max(1e-6, _finite_float(targets.get("loss", 0.5), 0.5))
    hard_rows = [row for row in holdout_rows if int(row.get("is_hard_case", 0) or 0) > 0]
    hard_case_component, hard_case_details = compute_hard_case_viewer_component(hard_rows, cfg, targets)

    components = {
        "loss": _finite_float(holdout_loss, loss_target) / loss_target,
        "sym_p95": min(sym_p95 / sym_target, 10.0),
        "manual_point_p95": min(manual_p95 / manual_target, 10.0),
        "manual_point_sr1_miss": max(0.0, 1.0 - manual_sr1),
        "no_curve_count": float(no_curve_count),
        "bad_rate": bad_rate,
        "vox03_empty": max(0.0, (vox03_target - vox03) / vox03_target),
        "cc_count": max(0.0, (cc_count - 1.0) / cc_target),
        "wrap_miss": max(0.0, 1.0 - wrap_coverage),
        "hard_case_viewer": hard_case_component,
    }
    score = 0.0
    for name, value in components.items():
        score += float(weights.get(name, 0.0)) * float(value)
    components.update(hard_case_details)
    return float(score), components, bad_rate


def make_monitor_metrics(epoch, holdout_loss, holdout_metrics, composite_score, score_components, bad_rate):
    out = {
        "epoch": int(epoch),
        "holdout_loss": float(holdout_loss),
        "holdout_composite_score": float(composite_score),
        "holdout_bad_rate": float(bad_rate),
        "score_components": score_components,
        "hard_case_viewer_component": float(score_components.get("hard_case_viewer", 0.0)),
        "hard_case_sym_p95": score_components.get("hard_case_sym_p95"),
        "hard_case_no_curve_count": score_components.get("hard_case_no_curve_count"),
        "hard_case_bad_rate": score_components.get("hard_case_bad_rate"),
    }
    for src, dst in [
        ("sym_mean", "holdout_sym_mean"),
        ("sym_p95", "holdout_sym_p95"),
        ("manual_point_mean", "holdout_manual_point_mean"),
        ("manual_point_p95", "holdout_manual_point_p95"),
        ("manual_point_sr1", "holdout_manual_point_sr1"),
        ("manual_point_sr2", "holdout_manual_point_sr2"),
        ("manual_point_count", "holdout_manual_point_count"),
        ("no_curve_count", "holdout_no_curve_count"),
        ("h_pred_max", "h_pred_max"),
        ("h_pred_p99", "h_pred_p99"),
        ("vox03", "vox03"),
        ("vox03_total", "vox03_total"),
        ("cc_count", "cc_count"),
        ("lcc_ratio", "lcc_ratio"),
        ("wrap_coverage", "wrap_coverage"),
        ("hard_case_count", "hard_case_count"),
        ("holdout_tooth_count", "holdout_tooth_count"),
    ]:
        out[dst] = holdout_metrics.get(src)
    return out


def write_hard_case_viewer(path, epoch, rows, hard_rows, monitor_metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    selected_rows = hard_rows or rows
    selected_rows = sorted(
        selected_rows,
        key=lambda r: (
            int(r.get("no_curve", 0) or 0) == 0,
            -_finite_float(r.get("sym_p95"), 0.0),
        ),
    )[:32]
    cols = [
        "case_id",
        "tooth_id",
        "is_hard_case",
        "sym_p95",
        "sym_mean",
        "no_curve",
        "vox03",
        "cc_count",
        "lcc_ratio",
        "wrap_coverage",
        "h_pred_max",
        "h_pred_p99",
    ]
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Holdout Hard Cases</title>",
        "<style>body{font-family:sans-serif}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 6px}th{background:#eee}</style>",
        "</head><body>",
        f"<h1>Holdout hard-case viewer epoch {int(epoch)}</h1>",
        f"<p>score={monitor_metrics['holdout_composite_score']:.6f}, loss={monitor_metrics['holdout_loss']:.6f}, sym_p95={monitor_metrics.get('holdout_sym_p95')}</p>",
        "<table><thead><tr>",
    ]
    lines.extend(f"<th>{html.escape(c)}</th>" for c in cols)
    lines.extend(["</tr></thead><tbody>"])
    for row in selected_rows:
        lines.append("<tr>")
        for col in cols:
            lines.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody></table>", "</body></html>"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
    topk_dir = ensure_dir(os.path.join(ckpt_dir, "topk"))
    archive_dir = ensure_dir(os.path.join(ckpt_dir, "archive"))
    analysis_dir = ensure_dir(os.path.join(out_dir, "holdout_analysis"))
    viewer_dir = ensure_dir(os.path.join(analysis_dir, "hard_case_viewers"))
    audit_paths = {
        "train": write_supervised_audit(train_ds.audit, os.path.join(out_dir, "audit_train")),
        "holdout": write_supervised_audit(holdout_ds.audit, os.path.join(out_dir, "audit_holdout")),
    }

    train_loader, sampler_weights = build_train_loader(train_ds, cfg)
    holdout_loader = torch.utils.data.DataLoader(
        holdout_ds,
        batch_size=int(cfg["train"].get("holdout_batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["train"].get("holdout_num_workers", cfg["train"]["num_workers"])),
    )

    model = build_model(cfg, device)
    pretrained_info = load_pretrained(
        model,
        cfg["train"].get("pretrained_ckpt"),
        strict=bool(cfg["train"].get("pretrained_strict", False)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = build_loss(cfg)
    monitor_loss_fn = DiceCELoss(sigmoid=True)

    max_epochs = int(cfg["train"].get("max_epochs", cfg["train"].get("epochs", 50)))
    patience = int(cfg["train"].get("early_stopping_patience", 8))
    min_delta = float(cfg["train"].get("early_stopping_min_delta", 0.0))
    min_epochs = int(cfg["train"].get("min_epochs", 1))
    score_start_epoch = int(cfg["train"].get("composite_score", {}).get("start_epoch", 10))
    metrics_path = os.path.join(out_dir, "metrics.csv")
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_composite_score = None
    best_monitor_metrics = None
    best_epoch = None
    epochs_without_improve = 0
    global_step = 0
    started_at = now_iso()
    top_k = int(cfg["train"].get("checkpoint_top_k", 5))
    checkpoint_every = int(cfg["train"].get("checkpoint_every_n_epochs", 5))
    topk_records = []

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "global_step",
                "train_loss",
                "holdout_loss",
                "holdout_composite_score",
                "best_composite_score",
                "holdout_sym_mean",
                "holdout_sym_p95",
                "holdout_manual_point_mean",
                "holdout_manual_point_p95",
                "holdout_manual_point_sr1",
                "holdout_manual_point_sr2",
                "holdout_manual_point_count",
                "holdout_no_curve_count",
                "holdout_bad_rate",
                "h_pred_max",
                "h_pred_p99",
                "vox03",
                "vox03_total",
                "cc_count",
                "lcc_ratio",
                "wrap_coverage",
                "hard_case_count",
                "score_active",
                "score_loss_component",
                "score_sym_p95_component",
                "score_no_curve_count_component",
                "score_manual_point_p95_component",
                "score_manual_point_sr1_miss_component",
                "score_bad_rate_component",
                "score_vox03_empty_component",
                "score_cc_count_component",
                "score_wrap_miss_component",
                "score_hard_case_viewer_component",
                "hard_case_sym_p95",
                "hard_case_no_curve_count",
                "hard_case_bad_rate",
                "best_epoch",
                "is_best",
                "epochs_without_improve",
            ],
        )
        writer.writeheader()
        for epoch in range(max_epochs):
            train_loss, global_step = train_one_epoch(model, train_loader, optimizer, device, loss_fn, cfg, global_step)
            holdout_loss = evaluate_monitor_loss(model, holdout_loader, device, monitor_loss_fn)
            holdout_metrics, holdout_rows, hard_rows = evaluate_holdout(model, holdout_ds, device, cfg, epoch)
            composite_score, score_components, bad_rate = compute_composite_score(
                holdout_loss,
                holdout_metrics,
                holdout_rows,
                cfg,
            )
            monitor_metrics = make_monitor_metrics(
                epoch,
                holdout_loss,
                holdout_metrics,
                composite_score,
                score_components,
                bad_rate,
            )
            score_active = epoch >= score_start_epoch
            improved = score_active and (
                best_composite_score is None or composite_score < (best_composite_score - min_delta)
            )
            if improved:
                best_composite_score = float(composite_score)
                best_monitor_metrics = dict(monitor_metrics)
                best_epoch = int(epoch)
                epochs_without_improve = 0
                save_checkpoint(best_path, model, optimizer, epoch, global_step, train_loss, monitor_metrics, cfg)
                evaluate_holdout(model, holdout_ds, device, cfg, epoch, out_dir=analysis_dir, save_prefix="best")
                write_hard_case_viewer(
                    os.path.join(viewer_dir, "best_hard_case_viewer.html"),
                    epoch,
                    holdout_rows,
                    hard_rows,
                    monitor_metrics,
                )
            else:
                if score_active and best_composite_score is not None:
                    epochs_without_improve += 1
            save_checkpoint(last_path, model, optimizer, epoch, global_step, train_loss, monitor_metrics, cfg)
            if score_active:
                topk_records = update_topk_checkpoints(
                    topk_records,
                    topk_dir,
                    top_k,
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    train_loss,
                    monitor_metrics,
                    cfg,
                )
            if checkpoint_every > 0 and ((epoch + 1) % checkpoint_every == 0):
                save_checkpoint(
                    os.path.join(archive_dir, f"epoch_{int(epoch):03d}.pt"),
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    train_loss,
                    monitor_metrics,
                    cfg,
                )
                if score_active:
                    write_hard_case_viewer(
                        os.path.join(viewer_dir, f"epoch_{int(epoch):03d}_hard_case_viewer.html"),
                        epoch,
                        holdout_rows,
                        hard_rows,
                        monitor_metrics,
                    )
            writer.writerow(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "holdout_loss": holdout_loss,
                    "holdout_composite_score": composite_score,
                    "best_composite_score": best_composite_score,
                    "holdout_sym_mean": monitor_metrics["holdout_sym_mean"],
                    "holdout_sym_p95": monitor_metrics["holdout_sym_p95"],
                    "holdout_no_curve_count": monitor_metrics["holdout_no_curve_count"],
                    "holdout_manual_point_mean": monitor_metrics["holdout_manual_point_mean"],
                    "holdout_manual_point_p95": monitor_metrics["holdout_manual_point_p95"],
                    "holdout_manual_point_sr1": monitor_metrics["holdout_manual_point_sr1"],
                    "holdout_manual_point_sr2": monitor_metrics["holdout_manual_point_sr2"],
                    "holdout_manual_point_count": monitor_metrics["holdout_manual_point_count"],
                    "holdout_bad_rate": monitor_metrics["holdout_bad_rate"],
                    "h_pred_max": monitor_metrics["h_pred_max"],
                    "h_pred_p99": monitor_metrics["h_pred_p99"],
                    "vox03": monitor_metrics["vox03"],
                    "vox03_total": monitor_metrics["vox03_total"],
                    "cc_count": monitor_metrics["cc_count"],
                    "lcc_ratio": monitor_metrics["lcc_ratio"],
                    "wrap_coverage": monitor_metrics["wrap_coverage"],
                    "hard_case_count": monitor_metrics["hard_case_count"],
                    "score_active": int(score_active),
                    "score_loss_component": score_components["loss"],
                    "score_sym_p95_component": score_components["sym_p95"],
                    "score_no_curve_count_component": score_components["no_curve_count"],
                    "score_manual_point_p95_component": score_components["manual_point_p95"],
                    "score_manual_point_sr1_miss_component": score_components["manual_point_sr1_miss"],
                    "score_bad_rate_component": score_components["bad_rate"],
                    "score_vox03_empty_component": score_components["vox03_empty"],
                    "score_cc_count_component": score_components["cc_count"],
                    "score_wrap_miss_component": score_components["wrap_miss"],
                    "score_hard_case_viewer_component": score_components["hard_case_viewer"],
                    "hard_case_sym_p95": monitor_metrics["hard_case_sym_p95"],
                    "hard_case_no_curve_count": monitor_metrics["hard_case_no_curve_count"],
                    "hard_case_bad_rate": monitor_metrics["hard_case_bad_rate"],
                    "best_epoch": best_epoch,
                    "is_best": int(improved),
                    "epochs_without_improve": epochs_without_improve,
                }
            )
            f.flush()
            logger.info(
                "epoch=%d train_loss=%.6f holdout_loss=%.6f score=%.6f sym_p95=%s manual_p95=%s manual_sr1=%s no_curve=%s bad_rate=%.4f vox03=%s cc=%s wrap=%s best_score=%s best_epoch=%s patience=%d/%d active=%s",
                epoch,
                train_loss,
                holdout_loss,
                composite_score,
                monitor_metrics["holdout_sym_p95"],
                monitor_metrics["holdout_manual_point_p95"],
                monitor_metrics["holdout_manual_point_sr1"],
                monitor_metrics["holdout_no_curve_count"],
                monitor_metrics["holdout_bad_rate"],
                monitor_metrics["vox03"],
                monitor_metrics["cc_count"],
                monitor_metrics["wrap_coverage"],
                best_composite_score,
                best_epoch,
                epochs_without_improve,
                patience,
                score_active,
            )
            if score_active and epoch + 1 >= min_epochs and epochs_without_improve >= patience:
                logger.info("early stopping at epoch=%d best_epoch=%s best_composite_score=%.6f", epoch, best_epoch, best_composite_score)
                break

    if best_monitor_metrics is None:
        raise RuntimeError(
            f"no best checkpoint selected; score_start_epoch={score_start_epoch} max_epochs={max_epochs}"
        )

    manifest = {
        "started_at": started_at,
        "finished_at": now_iso(),
        "config_path": os.path.abspath(args.config),
        "output_dir": out_dir,
        "metrics_csv": metrics_path,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "topk_checkpoints": topk_records,
        "checkpoint_top_k": top_k,
        "checkpoint_every_n_epochs": checkpoint_every,
        "archive_dir": archive_dir,
        "holdout_analysis_dir": analysis_dir,
        "hard_case_viewer": os.path.join(viewer_dir, "best_hard_case_viewer.html"),
        "early_stop_metric": "holdout_composite_score",
        "early_stop_score": cfg["train"].get("composite_score", {}),
        "score_start_epoch": score_start_epoch,
        "holdout_monitor_loss": "DiceCELoss(sigmoid=True)",
        "best_epoch": best_epoch,
        "best_composite_score": best_composite_score,
        "best_monitor_metrics": best_monitor_metrics,
        "best_holdout_loss": best_monitor_metrics.get("holdout_loss"),
        "best_holdout_sym_p95": best_monitor_metrics.get("holdout_sym_p95"),
        "best_holdout_manual_point_p95": best_monitor_metrics.get("holdout_manual_point_p95"),
        "best_holdout_manual_point_sr1": best_monitor_metrics.get("holdout_manual_point_sr1"),
        "best_holdout_no_curve_count": best_monitor_metrics.get("holdout_no_curve_count"),
        "best_holdout_bad_rate": best_monitor_metrics.get("holdout_bad_rate"),
        "best_wrap_coverage": best_monitor_metrics.get("wrap_coverage"),
        "best_hard_case_viewer_component": best_monitor_metrics.get("hard_case_viewer_component"),
        "best_hard_case_sym_p95": best_monitor_metrics.get("hard_case_sym_p95"),
        "best_hard_case_no_curve_count": best_monitor_metrics.get("hard_case_no_curve_count"),
        "best_hard_case_bad_rate": best_monitor_metrics.get("hard_case_bad_rate"),
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
