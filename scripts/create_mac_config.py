#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml


def expand(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Mac-local CEJ config from a server config.")
    parser.add_argument("--base-config", default="configs/server_sup_manual_inc_041055.yaml")
    parser.add_argument("--out", default="configs/mac_sup_manual_inc_041055.yaml")
    parser.add_argument("--runtime-root", default="~/cej_runtime")
    parser.add_argument("--toothfairy-root", default="~/ToothFairy3")
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--run-name", default="run_sup_inc_041055_001")
    parser.add_argument("--manual-run-name", default="run_manual_inc_041055_001")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    cfg = yaml.safe_load(base_path.read_text())

    runtime_root = Path(args.runtime_root).expanduser().resolve()
    tf_root = Path(args.toothfairy_root).expanduser().resolve()
    sup_run = runtime_root / "cej_runs" / args.run_name
    manual_run = runtime_root / "cej_runs" / args.manual_run_name

    cfg.setdefault("project", {})["device"] = args.device

    data = cfg.setdefault("data", {})
    data["raw_layout"] = "toothfairy3"
    data["raw_dir"] = str(tf_root)
    data["images_tr_dir"] = str(tf_root / "imagesTr")
    data["labels_tr_dir"] = str(tf_root / "labelsTr")
    data["points_dir"] = str(tf_root / "pointsTr")
    data["meta_dir"] = None
    # Prefer the copied symlink subset when present; fallback target is still stable.
    data["processed_dir"] = str(sup_run / "processed_manual")
    data["output_dir"] = str(sup_run / "outputs_mac")

    train = cfg.setdefault("train", {})
    train["pretrained_ckpt"] = str(sup_run / "outputs" / "train" / "checkpoints" / "last.pt")
    train["num_workers"] = min(int(train.get("num_workers", 4)), 2)

    infer = cfg.setdefault("infer", {})
    infer["constrain_curve_to_tooth_mask"] = False
    infer["constrain_curve_to_tooth_surface"] = False
    infer["keep_lcc_for_curve"] = False
    infer["fit_pred_curve"] = True

    cfg.setdefault("mac_runtime", {})["manual_usage_report"] = str(manual_run / "manual_points_usage_report.json")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
