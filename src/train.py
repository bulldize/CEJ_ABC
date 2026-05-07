import argparse
import csv
import datetime as dt
import json
import os
import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from monai.losses import DiceCELoss, DiceLoss
from monai.utils import set_determinism
import yaml

from src.datasets.dataset import ToothDataset, write_supervised_audit
from src.models.unet3d import UNet3D
from src.utils.config import load_config, ensure_dir, get_device
from src.utils.log import get_logger


logger = get_logger("train")


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class DiceFocalBCELoss(torch.nn.Module):
    def __init__(
        self,
        lambda_dice=1.0,
        lambda_bce=1.0,
        lambda_focal=0.25,
        focal_gamma=2.0,
        focal_alpha=0.75,
    ):
        super().__init__()
        self.dice_ce = DiceCELoss(sigmoid=True, lambda_dice=float(lambda_dice), lambda_ce=float(lambda_bce))
        self.lambda_focal = float(lambda_focal)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)

    def forward(self, logits, target):
        loss = self.dice_ce(logits, target)
        if self.lambda_focal <= 0.0:
            return loss
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = self.focal_alpha * target + (1.0 - self.focal_alpha) * (1.0 - target)
        focal = alpha_t * torch.pow(1.0 - p_t, self.focal_gamma) * bce
        return loss + self.lambda_focal * focal.mean()


class DiceFocalSkeletonLoss(DiceFocalBCELoss):
    def __init__(
        self,
        lambda_dice=1.0,
        lambda_bce=1.0,
        lambda_focal=0.25,
        focal_gamma=2.0,
        focal_alpha=0.75,
        lambda_skeleton=0.2,
        skeleton_target_threshold=0.95,
        skeleton_pos_weight=8.0,
    ):
        super().__init__(
            lambda_dice=lambda_dice,
            lambda_bce=lambda_bce,
            lambda_focal=lambda_focal,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        self.lambda_skeleton = float(lambda_skeleton)
        self.skeleton_target_threshold = float(skeleton_target_threshold)
        self.skeleton_pos_weight = float(skeleton_pos_weight)

    def forward(self, logits, target):
        loss = super().forward(logits, target)
        if self.lambda_skeleton <= 0.0:
            return loss
        skeleton = (target >= self.skeleton_target_threshold).float()
        skel_bce = F.binary_cross_entropy_with_logits(logits, skeleton, reduction="none")
        weights = 1.0 + skeleton * self.skeleton_pos_weight
        return loss + self.lambda_skeleton * (skel_bce * weights).mean()


def build_loss(cfg):
    loss_name = cfg["train"].get("loss", "dice_bce")
    if loss_name in ("dice_bce", "dice_ce"):
        return DiceCELoss(sigmoid=True)
    if loss_name == "dice":
        return DiceLoss(sigmoid=True)
    if loss_name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    if loss_name in ("dice_focal_bce", "dice_focal_hard"):
        return DiceFocalBCELoss(
            lambda_dice=cfg["train"].get("loss_lambda_dice", 1.0),
            lambda_bce=cfg["train"].get("loss_lambda_bce", 1.0),
            lambda_focal=cfg["train"].get("loss_lambda_focal", 0.25),
            focal_gamma=cfg["train"].get("focal_gamma", 2.0),
            focal_alpha=cfg["train"].get("focal_alpha", 0.75),
        )
    if loss_name == "dice_focal_skeleton":
        return DiceFocalSkeletonLoss(
            lambda_dice=cfg["train"].get("loss_lambda_dice", 1.0),
            lambda_bce=cfg["train"].get("loss_lambda_bce", 1.0),
            lambda_focal=cfg["train"].get("loss_lambda_focal", 0.25),
            focal_gamma=cfg["train"].get("focal_gamma", 2.0),
            focal_alpha=cfg["train"].get("focal_alpha", 0.75),
            lambda_skeleton=cfg["train"].get("loss_lambda_skeleton", 0.2),
            skeleton_target_threshold=cfg["train"].get("skeleton_target_threshold", 0.95),
            skeleton_pos_weight=cfg["train"].get("skeleton_pos_weight", 8.0),
        )
    raise NotImplementedError(f"unsupported loss: {loss_name}")


def _case_suffix(case_id):
    text = str(case_id)
    m = re.search(r"(\d+)$", text)
    return m.group(1) if m else text


def parse_hard_case_specs(specs):
    out = set()
    for spec in specs or []:
        if isinstance(spec, dict):
            case = str(spec.get("case_id", spec.get("case", ""))).strip()
            teeth = spec.get("teeth", spec.get("tooth_ids", spec.get("tooth_id", [])))
            if isinstance(teeth, (str, int)):
                teeth = str(teeth).replace(",", "/").split("/")
            for tooth in teeth:
                if str(tooth).strip():
                    out.add((_case_suffix(case), str(tooth).strip()))
            continue
        text = str(spec).strip()
        if not text:
            continue
        if "-" not in text:
            out.add((_case_suffix(text), "*"))
            continue
        case, teeth_raw = text.split("-", 1)
        for tooth in teeth_raw.replace(",", "/").split("/"):
            tooth = tooth.strip()
            if tooth:
                out.add((_case_suffix(case.strip()), tooth))
    return out


def is_hard_case_item(item, hard_cases):
    if not hard_cases:
        return False
    case = _case_suffix(item.get("case_id", ""))
    tooth = str(item.get("tooth_id", "")).strip()
    return (case, tooth) in hard_cases or (case, "*") in hard_cases


def build_sampler_weights(dataset, cfg):
    train_cfg = cfg.get("train", {})
    if not train_cfg.get("weighted_sampler", False):
        return None
    hard_cases = parse_hard_case_specs(train_cfg.get("hard_cases", []))
    hard_weight = float(train_cfg.get("hard_case_weight", 3.0))
    weights = []
    for item in getattr(dataset, "items", []):
        weights.append(hard_weight if is_hard_case_item(item, hard_cases) else 1.0)
    return torch.as_tensor(weights, dtype=torch.double)


def build_train_loader(dataset, cfg):
    weights = build_sampler_weights(dataset, cfg)
    sampler = None
    shuffle = True
    if weights is not None:
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=bool(cfg["train"].get("sampler_replacement", True)),
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg["train"]["num_workers"],
    )
    return loader, weights


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
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
    started_at = now_iso()
    set_determinism(seed=cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])

    pad_divisor = 2 ** max(0, int(cfg["model"]["depth"]) - 1)
    ds = ToothDataset(
        cfg["data"]["processed_dir"],
        processed_format=cfg["data"]["processed_format"],
        use_mask_channel=cfg["data"]["use_tooth_mask_channel"],
        clip_percentiles=cfg["preprocess"]["intensity_clip_percentiles"],
        norm_mode=cfg["preprocess"]["intensity_norm"],
        cache_rate=cfg["train"].get("cache_rate", 0.0),
        pad_divisor=pad_divisor,
    )

    out_dir = ensure_dir(os.path.join(cfg["data"]["output_dir"], "train"))
    config_snapshot_path = os.path.join(out_dir, "config_snapshot.yaml")
    with open(config_snapshot_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    audit_paths = write_supervised_audit(ds.audit, out_dir)
    logger.info(
        "supervised dataset audit total=%d used_labeled=%d skipped_unlabeled=%d invalid=%d audit=%s",
        ds.audit["total_tooth_dirs"],
        ds.audit["used_labeled_tooth_dirs"],
        ds.audit["skipped_unlabeled_tooth_dirs"],
        ds.audit.get("invalid_tooth_dirs", 0),
        audit_paths["audit"],
    )

    if len(ds) == 0:
        raise RuntimeError("no labeled supervised teeth available after audit")

    loader, sampler_weights = build_train_loader(ds, cfg)

    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        depth=cfg["model"]["depth"],
        num_res_units=cfg["model"].get("num_res_units", 2),
        norm=cfg["model"].get("norm", "batch"),
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
    loss_fn = build_loss(cfg)

    ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for epoch in range(cfg["train"]["epochs"]):
            loss = train_one_epoch(model, loader, optimizer, device, loss_fn)
            writer.writerow([epoch, loss])
            logger.info("epoch %d loss %.4f", epoch, loss)

    ckpt_path = os.path.join(ckpt_dir, "last.pt")
    torch.save({"model": model.state_dict()}, ckpt_path)
    logger.info("saved checkpoint %s", ckpt_path)

    manifest = {
        "started_at": started_at,
        "finished_at": now_iso(),
        "config_path": os.path.abspath(args.config),
        "config_snapshot": config_snapshot_path,
        "output_dir": out_dir,
        "checkpoint": ckpt_path,
        "metrics_csv": metrics_path,
        "audit": audit_paths,
        "dataset": {
            "total_tooth_dirs": ds.audit["total_tooth_dirs"],
            "used_labeled_tooth_dirs": ds.audit["used_labeled_tooth_dirs"],
            "skipped_unlabeled_tooth_dirs": ds.audit["skipped_unlabeled_tooth_dirs"],
            "invalid_tooth_dirs": ds.audit.get("invalid_tooth_dirs", 0),
        },
        "train": {
            "epochs": int(cfg["train"]["epochs"]),
            "batch_size": int(cfg["train"]["batch_size"]),
            "loss": cfg["train"].get("loss", "dice_bce"),
            "weighted_sampler": bool(cfg["train"].get("weighted_sampler", False)),
            "sampler_weight_count": int(len(sampler_weights)) if sampler_weights is not None else 0,
            "sampler_hard_weight_count": int((sampler_weights > 1.0).sum().item()) if sampler_weights is not None else 0,
        },
    }
    manifest_path = os.path.join(out_dir, "train_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("saved train manifest %s", manifest_path)


if __name__ == "__main__":
    main()
