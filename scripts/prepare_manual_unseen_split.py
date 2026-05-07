#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.datasets.dataset import build_supervised_audit, write_supervised_audit


DEFAULT_CASE_IDS = [
    "ToothFairy3F_025",
    "ToothFairy3F_026",
    "ToothFairy3F_027",
    "ToothFairy3F_033",
    "ToothFairy3F_040",
    "ToothFairy3F_041",
    "ToothFairy3F_044",
    "ToothFairy3F_050",
    "ToothFairy3F_051",
    "ToothFairy3F_052",
    "ToothFairy3F_053",
    "ToothFairy3F_054",
    "ToothFairy3F_055",
]

DEFAULT_HOLDOUT_CASE_IDS = [
    "ToothFairy3F_040",
    "ToothFairy3F_044",
    "ToothFairy3F_052",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def ensure_clean_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        remove_path(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        remove_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
    else:
        os.symlink(src, dst)


def has_direct_tooth_dirs(case_dir: Path) -> bool:
    if not case_dir.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith("tooth_") for p in case_dir.iterdir())


def index_processed_cases(source_roots: Sequence[Path]) -> Dict[str, Path]:
    out = {}
    for root in source_roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and has_direct_tooth_dirs(child):
                out.setdefault(child.name, child.resolve())
    return out


def load_reports(report_paths: Sequence[Path]) -> Dict[str, Dict]:
    cases = {}
    for path in report_paths:
        if not path.exists():
            continue
        payload = load_json(path)
        manual_root = payload.get("summary", {}).get("manual_root")
        for rec in payload.get("cases", []):
            case_id = str(rec.get("case_id", "")).strip()
            if not case_id:
                continue
            row = dict(rec)
            row["usage_report"] = str(path)
            row["manual_root"] = manual_root
            cases[case_id] = row
    return cases


def case_short_id(case_id: str) -> str:
    return case_id.rsplit("_", 1)[-1]


def manual_dir_for_case(report_cases: Dict[str, Dict], case_id: str) -> Optional[Path]:
    rec = report_cases.get(case_id)
    if not rec:
        return None
    manual_root = rec.get("manual_root")
    manual_dir = rec.get("manual_dir")
    if not manual_root or not manual_dir:
        return None
    return Path(manual_root) / str(manual_dir)


def write_usage_markers(
    case_ids: Sequence[str],
    holdout_case_ids: Sequence[str],
    report_cases: Dict[str, Dict],
    source_index: Dict[str, Path],
    merged_processed_dir: Path,
    train_dir: Path,
    test_dir: Path,
    sup_run_root: Path,
    pretrained_ckpt: str,
) -> None:
    holdout = set(holdout_case_ids)
    for case_id in case_ids:
        rec = report_cases.get(case_id, {})
        is_holdout = case_id in holdout
        payload = {
            "updated_at": now_iso(),
            "manual_dir": str(manual_dir_for_case(report_cases, case_id) or ""),
            "case_id": case_id,
            "status": str(rec.get("status", "PASS_FULLY_USED")),
            "usage_report": str(rec.get("usage_report", "")),
            "preprocess_run_root": str(source_index[case_id].parent.parent),
            "supervised_run_root": str(sup_run_root),
            "selected_for_training": not is_holdout,
            "trained": not is_holdout,
            "selected_for_test": is_holdout,
            "heldout_for_eval": is_holdout,
            "pretrained_ckpt": pretrained_ckpt,
        }

        manual_dir = manual_dir_for_case(report_cases, case_id)
        if manual_dir is not None and manual_dir.is_dir():
            write_json(manual_dir / "cej_usage_status.json", payload)

        for root in [source_index[case_id], merged_processed_dir / case_id, train_dir / case_id, test_dir / case_id]:
            if root.exists() or root.is_symlink():
                write_json(root / "cej_usage_status.json", payload)


def render_config(
    base_config: Path,
    out_config: Path,
    tf_root: Path,
    processed_dir: Path,
    output_dir: Path,
    pretrained_ckpt: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    train_ckpt_for_infer: Optional[str],
) -> Path:
    cfg = yaml.safe_load(base_config.read_text())
    cfg.setdefault("project", {})["device"] = "auto"

    data = cfg.setdefault("data", {})
    data["raw_layout"] = "toothfairy3"
    data["raw_dir"] = str(tf_root)
    data["images_tr_dir"] = str(tf_root / "imagesTr")
    data["labels_tr_dir"] = str(tf_root / "labelsTr")
    data["points_dir"] = str(tf_root / "pointsTr")
    data["meta_dir"] = None
    data["processed_dir"] = str(processed_dir)
    data["output_dir"] = str(output_dir)

    train = cfg.setdefault("train", {})
    train["epochs"] = int(epochs)
    train["batch_size"] = int(batch_size)
    train["num_workers"] = int(num_workers)
    train["loss"] = "dice_bce"
    train["pretrained_ckpt"] = str(pretrained_ckpt)
    train["pretrained_strict"] = False

    infer = cfg.setdefault("infer", {})
    infer["constrain_curve_to_tooth_mask"] = False
    infer["constrain_curve_to_tooth_surface"] = False
    infer["keep_lcc_for_curve"] = False
    infer["fit_pred_curve"] = True
    if train_ckpt_for_infer:
        infer["ckpt_path"] = str(train_ckpt_for_infer)
    else:
        infer.pop("ckpt_path", None)

    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))
    return out_config


def summarize_split(processed_dir: Path, audit_dir: Path) -> Dict:
    audit = build_supervised_audit(str(processed_dir))
    paths = write_supervised_audit(audit, str(audit_dir))
    case_ids = sorted([p.name for p in processed_dir.iterdir() if p.is_dir() or p.is_symlink()])
    return {
        "processed_dir": str(processed_dir),
        "case_ids": case_ids,
        "case_count": len(case_ids),
        "total_tooth_dirs": audit["total_tooth_dirs"],
        "used_labeled_tooth_dirs": audit["used_labeled_tooth_dirs"],
        "skipped_unlabeled_tooth_dirs": audit["skipped_unlabeled_tooth_dirs"],
        "audit_paths": paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare unseen manual train/test split from preprocessed case dirs.")
    parser.add_argument("--source-processed-roots", default="/root/cej_runs/run_manual_new_025040_001/processed,/root/cej_runs/run_manual_inc_041055_001/processed")
    parser.add_argument("--usage-report-paths", default="/root/cej_runs/run_manual_new_025040_001/manual_points_usage_report.json,/root/cej_runs/run_manual_inc_041055_001/manual_points_usage_report.json")
    parser.add_argument("--unseen-run-root", default="/root/cej_runs/run_manual_unseen_from_manual6_001")
    parser.add_argument("--sup-run-root", default="/root/cej_runs/run_sup_manual6_unseen_001")
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASE_IDS))
    parser.add_argument("--holdout-case-ids", default=",".join(DEFAULT_HOLDOUT_CASE_IDS))
    parser.add_argument("--base-sup-config", default="configs/server_sup_manual6.yaml")
    parser.add_argument("--train-config-out", default="configs/server_sup_manual6_unseen_train.yaml")
    parser.add_argument("--test-config-out", default="configs/server_sup_manual6_unseen_test.yaml")
    parser.add_argument("--toothfairy-root", default="/root/ToothFairy3")
    parser.add_argument("--pretrained-ckpt", default="/root/cej_runs/run_sup_manual6_002/outputs/train/checkpoints/last.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying case directories.")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    source_roots = [Path(p).resolve() for p in split_csv(args.source_processed_roots)]
    report_paths = [Path(p).resolve() for p in split_csv(args.usage_report_paths)]
    case_ids = split_csv(args.case_ids)
    holdout_case_ids = split_csv(args.holdout_case_ids)
    train_case_ids = [case_id for case_id in case_ids if case_id not in set(holdout_case_ids)]

    unseen_run_root = Path(args.unseen_run_root).resolve()
    sup_run_root = Path(args.sup_run_root).resolve()
    merged_processed_dir = unseen_run_root / "processed"
    train_dir = sup_run_root / "processed_manual_train"
    test_dir = sup_run_root / "processed_manual_test"
    outputs_dir = sup_run_root / "outputs"
    test_outputs_dir = sup_run_root / "outputs_test"
    audit_root = sup_run_root / "audit"

    source_index = index_processed_cases(source_roots)
    missing = [case_id for case_id in case_ids if case_id not in source_index]
    if missing:
        raise SystemExit(f"missing preprocessed case dirs: {missing}")

    ensure_clean_dir(merged_processed_dir)
    ensure_clean_dir(train_dir)
    ensure_clean_dir(test_dir)
    audit_root.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        link_or_copy(source_index[case_id], merged_processed_dir / case_id, copy=not args.symlink)
    for case_id in train_case_ids:
        link_or_copy(merged_processed_dir / case_id, train_dir / case_id, copy=not args.symlink)
    for case_id in holdout_case_ids:
        link_or_copy(merged_processed_dir / case_id, test_dir / case_id, copy=not args.symlink)

    report_cases = load_reports(report_paths)
    write_usage_markers(
        case_ids=case_ids,
        holdout_case_ids=holdout_case_ids,
        report_cases=report_cases,
        source_index=source_index,
        merged_processed_dir=merged_processed_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        sup_run_root=sup_run_root,
        pretrained_ckpt=args.pretrained_ckpt,
    )

    base_config = (repo_dir / args.base_sup_config).resolve() if not os.path.isabs(args.base_sup_config) else Path(args.base_sup_config)
    train_config = (repo_dir / args.train_config_out).resolve() if not os.path.isabs(args.train_config_out) else Path(args.train_config_out)
    test_config = (repo_dir / args.test_config_out).resolve() if not os.path.isabs(args.test_config_out) else Path(args.test_config_out)
    train_ckpt = outputs_dir / "train" / "checkpoints" / "last.pt"

    render_config(
        base_config=base_config,
        out_config=train_config,
        tf_root=Path(args.toothfairy_root).resolve(),
        processed_dir=train_dir,
        output_dir=outputs_dir,
        pretrained_ckpt=args.pretrained_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ckpt_for_infer=None,
    )
    render_config(
        base_config=base_config,
        out_config=test_config,
        tf_root=Path(args.toothfairy_root).resolve(),
        processed_dir=test_dir,
        output_dir=test_outputs_dir,
        pretrained_ckpt=args.pretrained_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ckpt_for_infer=str(train_ckpt),
    )

    summary = {
        "created_at": now_iso(),
        "case_ids": case_ids,
        "train_case_ids": train_case_ids,
        "holdout_case_ids": holdout_case_ids,
        "source_processed_roots": [str(p) for p in source_roots],
        "merged_processed_dir": str(merged_processed_dir),
        "train_config": str(train_config),
        "test_config": str(test_config),
        "pretrained_ckpt": str(args.pretrained_ckpt),
        "train": summarize_split(train_dir, audit_root / "train"),
        "test": summarize_split(test_dir, audit_root / "test"),
        "all_unseen": summarize_split(merged_processed_dir, audit_root / "all_unseen"),
    }
    write_json(sup_run_root / "split_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
