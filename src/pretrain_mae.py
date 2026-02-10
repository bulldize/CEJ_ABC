import argparse
import csv
import os
import torch
from torch.utils.data import DataLoader

from src.datasets.dataset import ToothDatasetUnsupervised
from src.models.unet3d import UNet3D
from src.utils.config import load_config, ensure_dir, get_device
from src.utils.mask import create_patch_mask_batch
from src.utils.seed import set_seed
from src.utils.log import get_logger


logger = get_logger("pretrain_mae")


def masked_mse_loss(pred, target, mask, eps=1e-6):
    diff = (pred - target) ** 2
    masked = diff * mask
    denom = mask.sum().clamp_min(eps)
    return masked.sum() / denom


def build_mae_model(cfg):
    # TODO: replace with ViT/MAE encoder-decoder (external skill integration)
    mcfg = cfg["model_unsup"]
    return UNet3D(
        in_channels=mcfg["in_channels"],
        out_channels=mcfg["out_channels"],
        base_channels=mcfg["base_channels"],
        depth=mcfg["depth"],
    )


def build_unsup_loss(cfg):
    loss_name = cfg["train_unsup"].get("loss", "masked_mse")
    if loss_name == "masked_mse":
        return masked_mse_loss
    # TODO: add other losses (e.g., MAE L1, perceptual, hybrid)
    raise NotImplementedError(f"unsupported loss: {loss_name}")


def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()
    total = 0.0
    steps = 0
    loss_fn = build_unsup_loss(cfg)
    tr_cfg = cfg["train_unsup"]
    mask_ratio = float(tr_cfg.get("mask_ratio", 0.75))
    mask_patch = tr_cfg.get("mask_patch_size", [8, 8, 8])
    mask_fill = tr_cfg.get("mask_fill", 0.0)
    use_mask_channel = bool(cfg["unsup"].get("use_tooth_mask_channel", True))
    max_steps = tr_cfg.get("max_steps")

    for batch in loader:
        a = batch["a"].to(device)
        t = batch["t"].to(device)

        mask = create_patch_mask_batch(
            a.shape[0],
            a.shape[-3:],
            mask_patch,
            mask_ratio,
            device=a.device,
            dtype=a.dtype,
        )

        if isinstance(mask_fill, str) and mask_fill.lower() == "mean":
            fill = a.mean().detach()
        else:
            fill = float(mask_fill)
        a_masked = a * (1.0 - mask) + fill * mask

        if use_mask_channel:
            x = torch.cat([a_masked, t], dim=1)
        else:
            x = a_masked

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, a, mask)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        steps += 1
        if max_steps is not None and steps >= int(max_steps):
            break

    return total / max(1, steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unsup_mae.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])

    ds = ToothDatasetUnsupervised(
        cfg["unsup"]["processed_dir"],
        processed_format=cfg["unsup"].get("processed_format", "nii.gz"),
        use_mask_channel=cfg["unsup"].get("use_tooth_mask_channel", True),
        clip_percentiles=cfg["preprocess"]["intensity_clip_percentiles"],
        norm_mode=cfg["preprocess"]["intensity_norm"],
    )

    if len(ds) == 0:
        logger.warning("no processed teeth found for unsupervised pretrain")
        return

    loader = DataLoader(
        ds,
        batch_size=cfg["train_unsup"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train_unsup"]["num_workers"],
    )

    model = build_mae_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train_unsup"]["lr"])

    out_dir = ensure_dir(os.path.join(cfg["unsup"]["output_dir"], "pretrain"))
    ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for epoch in range(cfg["train_unsup"]["epochs"]):
            loss = train_one_epoch(model, loader, optimizer, device, cfg)
            writer.writerow([epoch, loss])
            logger.info("epoch %d loss %.6f", epoch, loss)

    ckpt_path = os.path.join(ckpt_dir, "last.pt")
    torch.save({"model": model.state_dict()}, ckpt_path)
    logger.info("saved checkpoint %s", ckpt_path)


if __name__ == "__main__":
    main()
