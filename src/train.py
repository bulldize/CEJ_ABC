import argparse
import csv
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.dataset import ToothDataset
from src.models.unet3d import UNet3D
from src.utils.config import load_config, ensure_dir, get_device
from src.utils.seed import set_seed
from src.utils.log import get_logger


logger = get_logger("train")


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(2, 3, 4))
    den = (pred + target).sum(dim=(2, 3, 4)) + eps
    loss = 1 - (num / den)
    return loss.mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    bce = torch.nn.BCEWithLogitsLoss()
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = bce(logits, y) + dice_loss(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--pretrained-strict", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])

    ds = ToothDataset(
        cfg["data"]["processed_dir"],
        processed_format=cfg["data"]["processed_format"],
        use_mask_channel=cfg["data"]["use_tooth_mask_channel"],
        clip_percentiles=cfg["preprocess"]["intensity_clip_percentiles"],
        norm_mode=cfg["preprocess"]["intensity_norm"],
    )

    if len(ds) == 0:
        logger.warning("no processed teeth found")
        return

    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"])

    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        depth=cfg["model"]["depth"],
    ).to(device)

    pretrained = args.pretrained or cfg["train"].get("pretrained_ckpt")
    if pretrained:
        if os.path.exists(pretrained):
            ckpt = torch.load(pretrained, map_location="cpu")
            state = ckpt.get("model", ckpt)
            strict = args.pretrained_strict or bool(cfg["train"].get("pretrained_strict", False))
            missing, unexpected = model.load_state_dict(state, strict=strict)
            logger.info("loaded pretrained=%s strict=%s missing=%d unexpected=%d",
                        pretrained, strict, len(missing), len(unexpected))
        else:
            logger.warning("pretrained checkpoint not found: %s", pretrained)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    out_dir = ensure_dir(os.path.join(cfg["data"]["output_dir"], "train"))
    ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for epoch in range(cfg["train"]["epochs"]):
            loss = train_one_epoch(model, loader, optimizer, device)
            writer.writerow([epoch, loss])
            logger.info("epoch %d loss %.4f", epoch, loss)

    ckpt_path = os.path.join(ckpt_dir, "last.pt")
    torch.save({"model": model.state_dict()}, ckpt_path)
    logger.info("saved checkpoint %s", ckpt_path)


if __name__ == "__main__":
    main()
