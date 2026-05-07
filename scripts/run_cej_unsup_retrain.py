#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml


DEFAULT_RUN_ROOT = Path("/root/cej_isolated_runs/C/cej_loss_earlystop_20260507_001")
DEFAULT_UNSUP_CKPT = Path("/root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt")
DEFAULT_BASE_CONFIG = Path("configs/server_sup_manual6_unseen_train.yaml")
SUPERVISED_SOURCES = [
    Path("/root/cej_runs/run_sup_manual6_002/processed_manual"),
    Path("/root/cej_runs/run_manual_new_025040_001/processed"),
    Path("/root/cej_runs/run_manual_inc_041055_001/processed"),
]
HOLDOUT_CASES = ["ToothFairy3F_040", "ToothFairy3F_044", "ToothFairy3F_052"]
TRAIN_HARD_CASES = ["041-31/41/42"]
HOLDOUT_HARD_CASES = ["052-31/32/41/42/33", "040-42/41/32", "044-31/32/36/46"]


EXPERIMENTS = [
    {
        "name": "exp00_baseline_dicece",
        "loss": "dice_bce",
        "weighted_sampler": False,
        "run_condition": "all_loss_only_comparison",
    },
    {
        "name": "exp01_dice_focal_bce",
        "loss": "dice_focal_bce",
        "weighted_sampler": False,
        "run_condition": "all_loss_only_comparison",
    },
    {
        "name": "exp02_focal_hard_sampling",
        "loss": "dice_focal_hard",
        "weighted_sampler": True,
        "run_condition": "all_loss_only_comparison",
    },
    {
        "name": "exp03_skeleton_aux_loss",
        "loss": "dice_focal_skeleton",
        "weighted_sampler": True,
        "run_condition": "all_loss_only_comparison",
    },
    {
        "name": "exp04_surface_soft_constraint",
        "loss": "dice_focal_skeleton",
        "weighted_sampler": True,
        "surface_neighborhood_weight": 0.05,
        "run_condition": "all_loss_only_comparison",
    },
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def fail_if_exists(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} exists, refusing to overwrite: {path}")


def case_id_from_path(path: Path) -> str:
    return path.name


def collect_supervised_cases(sources: Iterable[Path]) -> Dict[str, Path]:
    cases = {}
    source_order = []
    for root in sources:
        if not root.exists():
            continue
        source_order.append(str(root))
        for case_dir in sorted(root.iterdir()):
            if not case_dir.is_dir():
                continue
            if not list(case_dir.glob("tooth_*")):
                continue
            case_id = case_id_from_path(case_dir)
            cases[case_id] = case_dir
    return cases


def copy_case_dirs(cases: Dict[str, Path], dst_root: Path, selected: List[str]) -> Dict[str, str]:
    dst_root.mkdir(parents=True, exist_ok=False)
    copied = {}
    for case_id in selected:
        src = cases.get(case_id)
        if src is None:
            raise FileNotFoundError(f"required supervised case missing: {case_id}")
        dst = dst_root / case_id
        fail_if_exists(dst, "case destination")
        shutil.copytree(src, dst, symlinks=False)
        copied[case_id] = str(src)
    return copied


def count_tooth_dirs(root: Path) -> int:
    return len(glob.glob(str(root / "*" / "tooth_*")))


def setup_inputs(run_root: Path, unsup_ckpt: Path) -> Dict:
    input_dir = run_root / "input"
    train_root = input_dir / "processed_train"
    holdout_root = input_dir / "processed_holdout"
    ckpt_dst = input_dir / "checkpoints" / "unsup_last.pt"
    manifest_path = input_dir / "source_manifest.json"
    fail_if_exists(input_dir, "input directory")
    cases = collect_supervised_cases(SUPERVISED_SOURCES)
    holdout = list(HOLDOUT_CASES)
    train = sorted([case for case in cases if case not in set(holdout)])
    if not train:
        raise RuntimeError("no training cases found after holdout split")
    if not unsup_ckpt.exists():
        raise FileNotFoundError(f"unsupervised checkpoint not found: {unsup_ckpt}")

    input_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
    train_sources = copy_case_dirs(cases, train_root, train)
    holdout_sources = copy_case_dirs(cases, holdout_root, holdout)
    shutil.copy2(unsup_ckpt, ckpt_dst)
    manifest = {
        "created_at": now_iso(),
        "supervised_sources": [str(p) for p in SUPERVISED_SOURCES],
        "unsup_checkpoint_source": str(unsup_ckpt),
        "unsup_checkpoint_copy": str(ckpt_dst),
        "train_processed": str(train_root),
        "holdout_processed": str(holdout_root),
        "train_cases": train,
        "holdout_cases": holdout,
        "train_case_sources": train_sources,
        "holdout_case_sources": holdout_sources,
        "train_tooth_dirs": count_tooth_dirs(train_root),
        "holdout_tooth_dirs": count_tooth_dirs(holdout_root),
        "train_hard_cases": TRAIN_HARD_CASES,
        "holdout_hard_cases": HOLDOUT_HARD_CASES,
    }
    write_json(manifest_path, manifest)
    write_split_summary(run_root, manifest)
    return manifest


def write_split_summary(run_root: Path, manifest: Dict) -> None:
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_dir / "split_summary.json", manifest)
    with (summary_dir / "split_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "case_id", "source"])
        writer.writeheader()
        for case_id in manifest["train_cases"]:
            writer.writerow({"split": "train", "case_id": case_id, "source": manifest["train_case_sources"][case_id]})
        for case_id in manifest["holdout_cases"]:
            writer.writerow({"split": "holdout", "case_id": case_id, "source": manifest["holdout_case_sources"][case_id]})


def load_or_setup_inputs(run_root: Path, unsup_ckpt: Path, resume: bool) -> Dict:
    manifest_path = run_root / "input" / "source_manifest.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"--resume requested but input manifest is missing: {manifest_path}")
        return read_json(manifest_path)
    return setup_inputs(run_root, unsup_ckpt)


def render_config(base_cfg: Dict, exp: Dict, attempt_root: Path, source_manifest: Dict, args) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("project", {})["device"] = args.device
    data = cfg.setdefault("data", {})
    data["processed_dir"] = source_manifest["train_processed"]
    data["holdout_processed_dir"] = source_manifest["holdout_processed"]
    data["output_dir"] = str(attempt_root / "outputs")
    data["processed_format"] = "nii.gz"
    data["use_tooth_mask_channel"] = True

    train = cfg.setdefault("train", {})
    train["loss"] = exp["loss"]
    train["pretrained_ckpt"] = source_manifest["unsup_checkpoint_copy"]
    train["pretrained_strict"] = False
    train["max_epochs"] = int(args.max_epochs)
    train["early_stopping_patience"] = int(args.patience)
    train["early_stopping_min_delta"] = float(args.early_stopping_min_delta)
    train["min_epochs"] = int(args.min_epochs)
    train["epochs"] = int(args.max_epochs)
    train["weighted_sampler"] = bool(exp.get("weighted_sampler", False))
    train["hard_cases"] = TRAIN_HARD_CASES if train["weighted_sampler"] else []
    train["hard_case_weight"] = float(args.hard_case_weight)
    train.setdefault("batch_size", int(args.batch_size))
    train["batch_size"] = int(args.batch_size)
    train["num_workers"] = int(args.num_workers)
    train["holdout_batch_size"] = int(args.holdout_batch_size)
    train["holdout_num_workers"] = int(args.holdout_num_workers)
    train["checkpoint_top_k"] = int(args.checkpoint_top_k)
    train["checkpoint_every_n_epochs"] = int(args.checkpoint_every_n_epochs)
    train["composite_score"] = {
        "start_epoch": int(args.score_start_epoch),
        "weights": {
            "loss": float(args.score_weight_loss),
            "sym_p95": float(args.score_weight_sym_p95),
            "no_curve_count": float(args.score_weight_no_curve),
            "bad_rate": float(args.score_weight_bad_rate),
            "vox03_empty": float(args.score_weight_vox03_empty),
            "cc_count": float(args.score_weight_cc_count),
            "wrap_miss": float(args.score_weight_wrap_miss),
            "hard_case_viewer": float(args.score_weight_hard_case_viewer),
        },
        "targets": {
            "loss": float(args.score_target_loss),
            "sym_p95_mm": float(args.score_target_sym_p95_mm),
            "bad_sym_p95_mm": float(args.score_bad_sym_p95_mm),
            "hard_sym_p95_mm": float(args.score_hard_sym_p95_mm),
            "hard_bad_sym_p95_mm": float(args.score_hard_bad_sym_p95_mm),
            "hard_case_count": len([tooth for spec in HOLDOUT_HARD_CASES for tooth in spec.split("-", 1)[1].split("/")]),
            "vox03": float(args.score_target_vox03),
            "cc_count": float(args.score_target_cc_count),
        },
    }
    train.setdefault("cache_rate", 0.0)
    train.setdefault("holdout_cache_rate", 0.0)
    train.setdefault("loss_lambda_dice", 1.0)
    train.setdefault("loss_lambda_bce", 1.0)
    train.setdefault("loss_lambda_focal", 0.25)
    train.setdefault("focal_gamma", 2.0)
    train.setdefault("focal_alpha", 0.75)
    train.setdefault("loss_lambda_skeleton", 0.2)
    train.setdefault("skeleton_target_threshold", 0.95)
    train.setdefault("skeleton_pos_weight", 8.0)
    train["surface_neighborhood_weight"] = float(exp.get("surface_neighborhood_weight", 0.0))
    train.setdefault("surface_neighborhood_radius_vox", 2)

    infer = cfg.setdefault("infer", {})
    infer["ckpt_path"] = str(attempt_root / "outputs" / "train" / "checkpoints" / "best.pt")
    infer["fit_pred_curve"] = True
    infer["constrain_curve_to_tooth_mask"] = False
    infer["keep_lcc_for_curve"] = False

    holdout = cfg.setdefault("holdout_eval", {})
    holdout["hard_cases"] = HOLDOUT_HARD_CASES
    holdout.setdefault("wrap_tau_mm", 1.0)
    holdout.setdefault("gt_peak_threshold", 0.95)
    holdout.setdefault("gt_rel_threshold", 0.95)
    return cfg


def run_cmd(cmd: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(f"[started_at] {now_iso()}\n")
        log.write("[cmd] " + " ".join(str(x) for x in cmd) + "\n")
        log.flush()
        proc = subprocess.run(list(cmd), cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[finished_at] {now_iso()}\n")
        log.write(f"[returncode] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(str(x) for x in cmd)}")


def experiment_names(raw: str) -> List[str]:
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [e["name"] for e in EXPERIMENTS]


def selected_experiments(raw: str) -> List[Dict]:
    names = experiment_names(raw)
    by_name = {e["name"]: e for e in EXPERIMENTS}
    return [dict(by_name[name]) for name in names]


def package_experiment(exp_name: str, exp_root: Path, run_root: Path) -> Path:
    pkg_dir = exp_root / "package"
    pkg_dir.mkdir(parents=True, exist_ok=False)
    pkg = pkg_dir / f"{exp_name}_training_bundle.tar.gz"
    fail_if_exists(pkg, "package")
    include = [
        exp_root / "config" / "train.yaml",
        exp_root / "outputs" / "train" / "checkpoints" / "best.pt",
        exp_root / "outputs" / "train" / "checkpoints" / "last.pt",
        exp_root / "outputs" / "train" / "metrics.csv",
        exp_root / "outputs" / "train" / "train_manifest.json",
        exp_root / "outputs" / "train" / "holdout_analysis" / "best_summary.json",
        exp_root / "outputs" / "train" / "holdout_analysis" / "best_per_tooth.csv",
        exp_root / "outputs" / "train" / "holdout_analysis" / "best_hard_cases.csv",
        exp_root / "outputs" / "train" / "holdout_analysis" / "hard_case_viewers" / "best_hard_case_viewer.html",
        exp_root / "outputs" / "train" / "audit_train" / "supervised_dataset_audit.json",
        exp_root / "outputs" / "train" / "audit_holdout" / "supervised_dataset_audit.json",
        run_root / "summary" / "split_summary.json",
        run_root / "summary" / "split_summary.csv",
    ]
    missing = [str(p) for p in include if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing package files: {missing}")
    with tarfile.open(pkg, "w:gz") as tf:
        for path in include:
            if path.is_relative_to(exp_root):
                arcname = str(path.relative_to(exp_root))
            else:
                arcname = str(path.relative_to(run_root))
            tf.add(path, arcname=arcname)
        for ckpt_path in sorted((exp_root / "outputs" / "train" / "checkpoints" / "topk").glob("*.pt")):
            tf.add(ckpt_path, arcname=str(ckpt_path.relative_to(exp_root)))
        for ckpt_path in sorted((exp_root / "outputs" / "train" / "checkpoints" / "archive").glob("*.pt")):
            tf.add(ckpt_path, arcname=str(ckpt_path.relative_to(exp_root)))
        for log_path in sorted((exp_root / "logs").glob("*.log")):
            tf.add(log_path, arcname=str(log_path.relative_to(exp_root)))
    return pkg


def read_train_manifest(exp_root: Path) -> Dict:
    p = exp_root / "outputs" / "train" / "train_manifest.json"
    return read_json(p) if p.exists() else {}


def run_experiment(repo_dir: Path, source_manifest: Dict, base_cfg: Dict, exp: Dict, args) -> Dict:
    exp_root = Path(args.run_root).resolve() / "experiments" / exp["name"]
    if exp_root.exists():
        if not args.resume:
            raise FileExistsError(f"experiment exists without --resume: {exp_root}")
        manifest_path = exp_root / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("status") == "success":
                print(f"[SKIP] {exp['name']} already succeeded")
                return manifest
        raise FileExistsError(f"experiment exists but is incomplete; inspect before retrying: {exp_root}")
    exp_root.mkdir(parents=True, exist_ok=False)
    (exp_root / "config").mkdir()
    (exp_root / "logs").mkdir()
    cfg = render_config(base_cfg, exp, exp_root, source_manifest, args)
    cfg_path = exp_root / "config" / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))
    started_at = now_iso()
    status = "success"
    error = None
    package_path = None
    try:
        run_cmd(
            [args.python_exe, "-m", "src.train_loss_earlystop", "--config", str(cfg_path)],
            cwd=repo_dir,
            log_path=exp_root / "logs" / "train_loss_earlystop.log",
        )
    except Exception as exc:
        status = "failed"
        error = repr(exc)
        if args.stop_on_failure:
            raise
    train_manifest = read_train_manifest(exp_root)
    manifest = {
        "experiment": exp["name"],
        "status": status,
        "started_at": started_at,
        "finished_at": now_iso(),
        "run_condition": exp.get("run_condition"),
        "config_path": str(cfg_path),
        "best_checkpoint": str(exp_root / "outputs" / "train" / "checkpoints" / "best.pt"),
        "last_checkpoint": str(exp_root / "outputs" / "train" / "checkpoints" / "last.pt"),
        "metrics_csv": str(exp_root / "outputs" / "train" / "metrics.csv"),
        "package_path": str(package_path) if package_path else None,
        "train_manifest": train_manifest,
        "best_epoch": train_manifest.get("best_epoch"),
        "best_composite_score": train_manifest.get("best_composite_score"),
        "best_holdout_loss": train_manifest.get("best_holdout_loss"),
        "best_holdout_sym_p95": train_manifest.get("best_holdout_sym_p95"),
        "best_holdout_no_curve_count": train_manifest.get("best_holdout_no_curve_count"),
        "best_holdout_bad_rate": train_manifest.get("best_holdout_bad_rate"),
        "best_wrap_coverage": train_manifest.get("best_wrap_coverage"),
        "best_hard_case_viewer_component": train_manifest.get("best_hard_case_viewer_component"),
        "best_hard_case_sym_p95": train_manifest.get("best_hard_case_sym_p95"),
        "best_hard_case_no_curve_count": train_manifest.get("best_hard_case_no_curve_count"),
        "best_hard_case_bad_rate": train_manifest.get("best_hard_case_bad_rate"),
        "best_monitor_metrics": train_manifest.get("best_monitor_metrics"),
        "hard_case_viewer": train_manifest.get("hard_case_viewer"),
        "early_stop_metric": train_manifest.get("early_stop_metric"),
        "early_stop_score": train_manifest.get("early_stop_score"),
        "error": error,
    }
    write_json(exp_root / "manifest.json", manifest)
    if status == "success":
        try:
            package_path = package_experiment(exp["name"], exp_root, Path(args.run_root).resolve())
            manifest["package_path"] = str(package_path)
            write_json(exp_root / "manifest.json", manifest)
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = repr(exc)
            write_json(exp_root / "manifest.json", manifest)
            if args.stop_on_failure:
                raise
    print(
        f"[{status.upper()}] {exp['name']} best_epoch={manifest.get('best_epoch')} "
        f"best_score={manifest.get('best_composite_score')} best_holdout_loss={manifest.get('best_holdout_loss')} "
        f"package={manifest.get('package_path')}"
    )
    return manifest


def summarize(run_root: Path, manifests: List[Dict]) -> Dict:
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for m in manifests:
        row = {
            "experiment": m.get("experiment"),
            "status": m.get("status"),
            "best_epoch": m.get("best_epoch"),
            "best_composite_score": m.get("best_composite_score"),
            "best_holdout_loss": m.get("best_holdout_loss"),
            "best_holdout_sym_p95": m.get("best_holdout_sym_p95"),
            "best_holdout_no_curve_count": m.get("best_holdout_no_curve_count"),
            "best_holdout_bad_rate": m.get("best_holdout_bad_rate"),
            "best_wrap_coverage": m.get("best_wrap_coverage"),
            "best_hard_case_viewer_component": m.get("best_hard_case_viewer_component"),
            "best_hard_case_sym_p95": m.get("best_hard_case_sym_p95"),
            "best_hard_case_no_curve_count": m.get("best_hard_case_no_curve_count"),
            "best_hard_case_bad_rate": m.get("best_hard_case_bad_rate"),
            "hard_case_viewer": m.get("hard_case_viewer"),
            "early_stop_metric": m.get("early_stop_metric"),
            "best_checkpoint": m.get("best_checkpoint"),
            "package_path": m.get("package_path"),
        }
        rows.append(row)
        if m.get("status") == "success" and m.get("best_composite_score") is not None:
            candidates.append((float(m["best_composite_score"]), int(m.get("best_epoch") or 10**9), m))
    csv_path = summary_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    candidates.sort(key=lambda x: x[:2])
    best_manifest = candidates[0][2] if candidates else None
    best_ckpt = best_manifest.get("best_checkpoint") if best_manifest else None
    best_pkg = best_manifest.get("package_path") if best_manifest else None
    (summary_dir / "BEST_MODEL_PATH.txt").write_text((best_ckpt or "") + ("\n" if best_ckpt else ""))
    (summary_dir / "BEST_PACKAGE_PATH.txt").write_text((best_pkg or "") + ("\n" if best_pkg else ""))
    payload = {
        "created_at": now_iso(),
        "results_csv": str(csv_path),
        "best_model_path": best_ckpt,
        "best_package_path": best_pkg,
        "experiments": manifests,
    }
    write_json(summary_dir / "experiments_manifest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--unsup-ckpt", default=os.environ.get("UNSUP_CKPT", str(DEFAULT_UNSUP_CKPT)))
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--experiments", default=None)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--holdout-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--holdout-num-workers", type=int, default=2)
    parser.add_argument("--hard-case-weight", type=float, default=4.0)
    parser.add_argument("--checkpoint-top-k", type=int, default=5)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    parser.add_argument("--score-start-epoch", type=int, default=10)
    parser.add_argument("--score-weight-loss", type=float, default=0.35)
    parser.add_argument("--score-weight-sym-p95", type=float, default=0.35)
    parser.add_argument("--score-weight-no-curve", type=float, default=0.08)
    parser.add_argument("--score-weight-bad-rate", type=float, default=0.08)
    parser.add_argument("--score-weight-vox03-empty", type=float, default=0.04)
    parser.add_argument("--score-weight-cc-count", type=float, default=0.04)
    parser.add_argument("--score-weight-wrap-miss", type=float, default=0.06)
    parser.add_argument("--score-weight-hard-case-viewer", type=float, default=0.12)
    parser.add_argument("--score-target-loss", type=float, default=0.5)
    parser.add_argument("--score-target-sym-p95-mm", type=float, default=5.0)
    parser.add_argument("--score-bad-sym-p95-mm", type=float, default=5.0)
    parser.add_argument("--score-hard-sym-p95-mm", type=float, default=5.0)
    parser.add_argument("--score-hard-bad-sym-p95-mm", type=float, default=5.0)
    parser.add_argument("--score-target-vox03", type=float, default=100.0)
    parser.add_argument("--score-target-cc-count", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    run_root = Path(args.run_root).resolve()
    args.run_root = str(run_root)
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"run root exists; use --resume to continue: {run_root}")
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        (run_root / "experiments").mkdir()
        (run_root / "summary").mkdir()
    source_manifest = load_or_setup_inputs(run_root, Path(args.unsup_ckpt).resolve(), args.resume)
    if args.setup_only:
        print(f"[OK] setup complete: {run_root}")
        return 0
    base_cfg_path = Path(args.base_config)
    if not base_cfg_path.is_absolute():
        base_cfg_path = repo_dir / base_cfg_path
    base_cfg = yaml.safe_load(base_cfg_path.read_text())
    manifests = []
    for exp in selected_experiments(args.experiments):
        manifests.append(run_experiment(repo_dir, source_manifest, base_cfg, exp, args))
    summary = summarize(run_root, manifests)
    print(f"[SUMMARY] {summary['results_csv']}")
    print(f"[BEST_MODEL] {summary.get('best_model_path')}")
    print(f"[BEST_PACKAGE] {summary.get('best_package_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
