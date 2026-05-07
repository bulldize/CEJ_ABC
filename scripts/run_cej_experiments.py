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
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


DEFAULT_RUN_ROOT = Path("/root/cej_isolated_runs/C/cej_improve_20260507_001")
DEFAULT_SOURCE_PROCESSED = Path("/root/cej_runs/run_sup_inc_041055_001/processed_manual")
DEFAULT_SEED_CHECKPOINT = Path("/root/cej_runs/run_sup_inc_041055_001/outputs/train/checkpoints/last.pt")
HARD_CASE_SPECS = ["052-31/32/41/42/33", "041-31/41/42"]


EXPERIMENTS = [
    {
        "name": "exp00_audit_baseline",
        "loss": "dice_bce",
        "weighted_sampler": False,
        "label_cleaning": "strict_h_gt_audit",
    },
    {
        "name": "exp01_clean_dicece",
        "loss": "dice_bce",
        "weighted_sampler": False,
        "label_cleaning": "strict_h_gt_audit",
    },
    {
        "name": "exp02_dice_focal_bce",
        "loss": "dice_focal_bce",
        "weighted_sampler": False,
        "label_cleaning": "strict_h_gt_audit",
    },
    {
        "name": "exp03_focal_hard_sampling",
        "loss": "dice_focal_hard",
        "weighted_sampler": True,
        "label_cleaning": "strict_h_gt_audit",
    },
    {
        "name": "exp04_skeleton_aux_loss",
        "loss": "dice_focal_skeleton",
        "weighted_sampler": True,
        "label_cleaning": "strict_h_gt_audit",
    },
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def fail_if_exists(path: Path, what: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{what} already exists, refusing to overwrite: {path}")


def count_tooth_dirs(processed_dir: Path) -> int:
    return len(glob.glob(str(processed_dir / "*" / "tooth_*")))


def validate_isolated_inputs(manifest: Dict) -> None:
    tooth_count = count_tooth_dirs(Path(manifest["processed_copy"]))
    if tooth_count <= 0:
        raise RuntimeError(f"processed copy has no tooth directories: {manifest['processed_copy']}")
    if tooth_count != int(manifest.get("tooth_dir_count", tooth_count)):
        manifest["tooth_dir_count"] = tooth_count


def copy_inputs(run_root: Path, source_processed: Path, seed_checkpoint: Path) -> Dict:
    if not source_processed.exists():
        raise FileNotFoundError(f"source processed directory not found: {source_processed}")
    if not seed_checkpoint.exists():
        raise FileNotFoundError(f"seed checkpoint not found: {seed_checkpoint}")

    input_dir = run_root / "input"
    processed_copy = input_dir / "processed_manual_copy"
    ckpt_copy = input_dir / "checkpoints" / "seed_last.pt"
    source_manifest = input_dir / "source_manifest.json"

    fail_if_exists(processed_copy, "processed copy")
    fail_if_exists(ckpt_copy, "seed checkpoint copy")
    fail_if_exists(source_manifest, "source manifest")

    input_dir.mkdir(parents=True, exist_ok=False)
    ckpt_copy.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(source_processed, processed_copy, symlinks=False)
    shutil.copy2(seed_checkpoint, ckpt_copy)

    manifest = {
        "created_at": now_iso(),
        "source_processed": str(source_processed),
        "processed_copy": str(processed_copy),
        "source_seed_checkpoint": str(seed_checkpoint),
        "seed_checkpoint_copy": str(ckpt_copy),
        "tooth_dir_count": count_tooth_dirs(processed_copy),
    }
    write_json(source_manifest, manifest)
    return manifest


def ensure_or_resume_inputs(run_root: Path, source_processed: Path, seed_checkpoint: Path, resume: bool) -> Dict:
    source_manifest = run_root / "input" / "source_manifest.json"
    if resume:
        if not source_manifest.exists():
            raise FileNotFoundError(f"resume requested but source manifest is missing: {source_manifest}")
        manifest = read_json(source_manifest)
        for key in ("processed_copy", "seed_checkpoint_copy"):
            if not Path(manifest[key]).exists():
                raise FileNotFoundError(f"resume input missing: {manifest[key]}")
        return manifest
    return copy_inputs(run_root, source_processed, seed_checkpoint)


def experiment_by_name(names: Iterable[str]) -> List[Dict]:
    available = {exp["name"]: exp for exp in EXPERIMENTS}
    selected = []
    for name in names:
        if name not in available:
            raise KeyError(f"unknown experiment {name}; available={sorted(available)}")
        selected.append(dict(available[name]))
    return selected


def parse_experiment_names(raw: Optional[str]) -> List[str]:
    if not raw:
        return [exp["name"] for exp in EXPERIMENTS]
    return [x.strip() for x in raw.split(",") if x.strip()]


def latest_attempt(exp_root: Path) -> Optional[Path]:
    if not exp_root.exists():
        return None
    attempts = [exp_root]
    attempts.extend(sorted(exp_root.glob("attempt_*")))
    attempts = [p for p in attempts if (p / "manifest.json").exists()]
    if not attempts:
        return None
    return attempts[-1]


def next_attempt_dir(exp_root: Path) -> Path:
    attempt_nums = [1]
    for p in exp_root.glob("attempt_*"):
        try:
            attempt_nums.append(int(p.name.split("_", 1)[1]))
        except Exception:
            continue
    return exp_root / f"attempt_{max(attempt_nums) + 1:03d}"


def choose_attempt_root(run_root: Path, exp_name: str, resume: bool) -> Tuple[Path, int, bool]:
    exp_root = run_root / "experiments" / exp_name
    if not exp_root.exists():
        return exp_root, 1, False

    if not resume:
        raise FileExistsError(f"experiment directory exists without --resume: {exp_root}")

    latest = latest_attempt(exp_root)
    if latest is None:
        raise FileExistsError(f"experiment directory exists but has no manifest, refusing to reuse: {exp_root}")

    manifest = read_json(latest / "manifest.json")
    if manifest.get("status") == "success":
        return latest, int(manifest.get("attempt", 1)), True

    next_dir = next_attempt_dir(exp_root)
    fail_if_exists(next_dir, "retry attempt directory")
    attempt = int(next_dir.name.split("_", 1)[1])
    return next_dir, attempt, False


def update_config_for_experiment(
    base_cfg: Dict,
    exp: Dict,
    attempt_root: Path,
    processed_copy: Path,
    seed_checkpoint: Path,
    args: argparse.Namespace,
) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("project", {})
    if args.device:
        cfg["project"]["device"] = args.device

    data = cfg.setdefault("data", {})
    data["processed_dir"] = str(processed_copy)
    data["output_dir"] = str(attempt_root / "outputs")
    data.setdefault("processed_format", "nii.gz")
    data.setdefault("use_tooth_mask_channel", True)

    train = cfg.setdefault("train", {})
    train["loss"] = exp["loss"]
    train["pretrained_ckpt"] = str(seed_checkpoint)
    train["pretrained_strict"] = False
    train["weighted_sampler"] = bool(exp.get("weighted_sampler", False))
    train["hard_cases"] = HARD_CASE_SPECS if train["weighted_sampler"] else []
    train.setdefault("hard_case_weight", 4.0)
    train.setdefault("loss_lambda_dice", 1.0)
    train.setdefault("loss_lambda_bce", 1.0)
    train.setdefault("loss_lambda_focal", 0.25)
    train.setdefault("focal_gamma", 2.0)
    train.setdefault("focal_alpha", 0.75)
    train.setdefault("loss_lambda_skeleton", 0.2)
    train.setdefault("skeleton_target_threshold", 0.95)
    train.setdefault("skeleton_pos_weight", 8.0)
    train["label_cleaning"] = exp.get("label_cleaning", "strict_h_gt_audit")
    if args.epochs is not None:
        train["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        train["num_workers"] = int(args.num_workers)

    infer = cfg.setdefault("infer", {})
    infer["ckpt_path"] = str(attempt_root / "outputs" / "train" / "checkpoints" / "last.pt")
    infer["fit_pred_curve"] = True

    eval_cfg = cfg.setdefault("eval", {})
    eval_cfg["prediction_name"] = "C_pred_fit.nii.gz"
    eval_cfg.setdefault("eval_taus_mm", [0.5, 1.0, 1.5])
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


def read_eval_summary(path: Path) -> Dict:
    if not path.exists():
        return {}
    return read_json(path)


def read_audit_summary(path: Path) -> Dict:
    if not path.exists():
        return {}
    audit = read_json(path)
    return {
        "total_tooth_dirs": audit.get("total_tooth_dirs"),
        "used_labeled_tooth_dirs": audit.get("used_labeled_tooth_dirs"),
        "skipped_unlabeled_tooth_dirs": audit.get("skipped_unlabeled_tooth_dirs"),
        "invalid_tooth_dirs": audit.get("invalid_tooth_dirs", 0),
        "invalid_used_tooth_dirs": 0,
    }


def required_package_files(attempt_root: Path) -> List[Path]:
    outputs = attempt_root / "outputs"
    return [
        outputs / "train" / "checkpoints" / "last.pt",
        outputs / "train" / "metrics.csv",
        outputs / "train" / "supervised_dataset_audit.json",
        outputs / "eval" / "metrics_summary.json",
        outputs / "eval" / "metrics_per_tooth.csv",
        attempt_root / "config" / "train.yaml",
        attempt_root / "manifest.json",
    ]


def package_experiment(exp_name: str, attempt_root: Path, attempt: int) -> Path:
    package_dir = attempt_root / "package"
    package_dir.mkdir(parents=True, exist_ok=False)
    suffix = "" if attempt == 1 else f"_attempt_{attempt:03d}"
    package_path = package_dir / f"{exp_name}{suffix}_training_bundle.tar.gz"
    fail_if_exists(package_path, "experiment package")

    missing = [p for p in required_package_files(attempt_root) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"cannot package experiment; missing files: {missing}")

    with tarfile.open(package_path, "w:gz") as tf:
        for path in required_package_files(attempt_root):
            tf.add(path, arcname=str(path.relative_to(attempt_root)))
        logs_dir = attempt_root / "logs"
        for log_path in sorted(logs_dir.glob("*.log")):
            tf.add(log_path, arcname=str(log_path.relative_to(attempt_root)))
    return package_path


def build_manifest(
    exp_name: str,
    attempt: int,
    attempt_root: Path,
    status: str,
    started_at: str,
    config_path: Path,
    error: Optional[str] = None,
    package_path: Optional[Path] = None,
) -> Dict:
    outputs = attempt_root / "outputs"
    metrics_summary = read_eval_summary(outputs / "eval" / "metrics_summary.json")
    audit_summary = read_audit_summary(outputs / "train" / "supervised_dataset_audit.json")
    return {
        "experiment": exp_name,
        "attempt": attempt,
        "status": status,
        "started_at": started_at,
        "finished_at": now_iso(),
        "attempt_root": str(attempt_root),
        "config_path": str(config_path),
        "model_path": str(outputs / "train" / "checkpoints" / "last.pt"),
        "metrics_csv": str(outputs / "train" / "metrics.csv"),
        "audit_path": str(outputs / "train" / "supervised_dataset_audit.json"),
        "eval_summary_path": str(outputs / "eval" / "metrics_summary.json"),
        "eval_per_tooth_path": str(outputs / "eval" / "metrics_per_tooth.csv"),
        "package_path": str(package_path) if package_path else None,
        "metrics_summary": metrics_summary,
        "audit_summary": audit_summary,
        "error": error,
    }


def run_experiment(
    repo_dir: Path,
    python_exe: str,
    run_root: Path,
    base_cfg: Dict,
    exp: Dict,
    source_manifest: Dict,
    args: argparse.Namespace,
) -> Dict:
    exp_name = exp["name"]
    attempt_root, attempt, skip = choose_attempt_root(run_root, exp_name, args.resume)
    if skip:
        manifest = read_json(attempt_root / "manifest.json")
        print(f"[SKIP] {exp_name} already succeeded at {attempt_root}")
        return manifest

    fail_if_exists(attempt_root, "experiment attempt directory")
    attempt_root.mkdir(parents=True, exist_ok=False)
    (attempt_root / "config").mkdir()
    (attempt_root / "logs").mkdir()
    config_path = attempt_root / "config" / "train.yaml"
    started_at = now_iso()

    cfg = update_config_for_experiment(
        base_cfg=base_cfg,
        exp=exp,
        attempt_root=attempt_root,
        processed_copy=Path(source_manifest["processed_copy"]),
        seed_checkpoint=Path(source_manifest["seed_checkpoint_copy"]),
        args=args,
    )
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))

    commands = [
        ("train", [python_exe, "-m", "src.train", "--config", str(config_path)]),
        ("infer", [python_exe, "-m", "src.infer", "--config", str(config_path)]),
        ("eval", [python_exe, "-m", "src.eval", "--config", str(config_path)]),
    ]

    try:
        for stage, cmd in commands:
            print(f"[RUN] {exp_name} attempt={attempt:03d} stage={stage}")
            run_cmd(cmd, cwd=repo_dir, log_path=attempt_root / "logs" / f"{stage}.log")

        manifest = build_manifest(
            exp_name=exp_name,
            attempt=attempt,
            attempt_root=attempt_root,
            status="success",
            started_at=started_at,
            config_path=config_path,
        )
        write_json(attempt_root / "manifest.json", manifest)
        package_path = package_experiment(exp_name, attempt_root, attempt)
        manifest = build_manifest(
            exp_name=exp_name,
            attempt=attempt,
            attempt_root=attempt_root,
            status="success",
            started_at=started_at,
            config_path=config_path,
            package_path=package_path,
        )
        write_json(attempt_root / "manifest.json", manifest)
        print(f"[OK] {exp_name} package={package_path}")
        return manifest
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        manifest = build_manifest(
            exp_name=exp_name,
            attempt=attempt,
            attempt_root=attempt_root,
            status="failed",
            started_at=started_at,
            config_path=config_path,
            error=error,
        )
        manifest["traceback"] = traceback.format_exc()
        write_json(attempt_root / "manifest.json", manifest)
        print(f"[FAIL] {exp_name}: {error}")
        if args.stop_on_failure:
            raise
        return manifest


def case_suffix(case_id: str) -> str:
    text = str(case_id)
    digits = ""
    for ch in reversed(text):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return digits or text


def expand_hard_cases(specs: Iterable[str]) -> set:
    out = set()
    for spec in specs:
        case, teeth = str(spec).split("-", 1)
        for tooth in teeth.replace(",", "/").split("/"):
            tooth = tooth.strip()
            if tooth:
                out.add((case_suffix(case), tooth))
    return out


def load_per_tooth_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value, default=float("inf")) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def sr_at_1(summary: Dict) -> float:
    sr = summary.get("sr", {})
    for key in ("1.0", "1", 1.0, 1):
        if key in sr:
            return as_float(sr[key], default=0.0)
    return 0.0


def hard_cases_present_and_ok(rows: List[Dict], hard_cases: set) -> Tuple[bool, List[str]]:
    row_map = {
        (case_suffix(r.get("case_id", "")), str(r.get("tooth_id", "")).strip()): r
        for r in rows
    }
    missing = []
    for key in sorted(hard_cases):
        row = row_map.get(key)
        if row is None:
            missing.append(f"{key[0]}-{key[1]}")
            continue
        if row.get("status") != "ok":
            missing.append(f"{key[0]}-{key[1]}:{row.get('status')}")
    return not missing, missing


def summarize_and_select_best(run_root: Path, manifests: List[Dict]) -> Dict:
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    hard_cases = expand_hard_cases(HARD_CASE_SPECS)
    rows = []
    candidates = []

    for manifest in manifests:
        summary = manifest.get("metrics_summary") or {}
        per_tooth_path = Path(manifest.get("eval_per_tooth_path") or "")
        per_tooth_rows = load_per_tooth_rows(per_tooth_path)
        hard_ok, hard_missing = hard_cases_present_and_ok(per_tooth_rows, hard_cases)
        model_path = Path(manifest.get("model_path") or "")
        package_path = Path(manifest.get("package_path") or "")
        audit_summary = manifest.get("audit_summary") or {}
        invalid_used = int(audit_summary.get("invalid_used_tooth_dirs", 0) or 0)

        worst_p95 = max([as_float(r.get("p95_dist_mm")) for r in per_tooth_rows] or [float("inf")])
        p95 = as_float(summary.get("p95"))
        mean = as_float(summary.get("mean"))
        sr1 = sr_at_1(summary)
        eligible = (
            manifest.get("status") == "success"
            and model_path.exists()
            and Path(manifest.get("eval_summary_path") or "").exists()
            and hard_ok
            and invalid_used == 0
        )
        reason = ""
        if manifest.get("status") != "success":
            reason = "not_success"
        elif not model_path.exists():
            reason = "missing_model"
        elif not Path(manifest.get("eval_summary_path") or "").exists():
            reason = "missing_eval"
        elif not hard_ok:
            reason = "missing_or_failed_hard_cases:" + ",".join(hard_missing)
        elif invalid_used != 0:
            reason = "invalid_samples_used"

        row = {
            "experiment": manifest.get("experiment"),
            "attempt": manifest.get("attempt"),
            "status": manifest.get("status"),
            "eligible": str(bool(eligible)).lower(),
            "reason": reason,
            "p95_dist_mm": "" if p95 == float("inf") else p95,
            "mean_dist_mm": "" if mean == float("inf") else mean,
            "sr@1.0mm": sr1,
            "worst_tooth_p95_dist_mm": "" if worst_p95 == float("inf") else worst_p95,
            "model_path": str(model_path),
            "package_path": str(package_path),
        }
        rows.append(row)
        if eligible:
            candidates.append((p95, mean, -sr1, worst_p95, row, manifest))

    results_csv = summary_dir / "results.csv"
    fieldnames = [
        "experiment",
        "attempt",
        "status",
        "eligible",
        "reason",
        "p95_dist_mm",
        "mean_dist_mm",
        "sr@1.0mm",
        "worst_tooth_p95_dist_mm",
        "model_path",
        "package_path",
    ]
    with results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    candidates.sort(key=lambda x: x[:4])
    best = candidates[0] if candidates else None
    best_manifest = best[5] if best else None
    best_model = Path(best_manifest["model_path"]) if best_manifest else None
    best_package = Path(best_manifest["package_path"]) if best_manifest else None

    (summary_dir / "BEST_MODEL_PATH.txt").write_text(str(best_model) + "\n" if best_model else "")
    (summary_dir / "BEST_PACKAGE_PATH.txt").write_text(str(best_package) + "\n" if best_package else "")

    experiments_manifest = {
        "created_at": now_iso(),
        "run_root": str(run_root),
        "hard_cases": sorted([f"{c}-{t}" for c, t in hard_cases]),
        "results_csv": str(results_csv),
        "best_model_path": str(best_model) if best_model else None,
        "best_package_path": str(best_package) if best_package else None,
        "experiments": manifests,
    }
    write_json(summary_dir / "experiments_manifest.json", experiments_manifest)
    return experiments_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated CEJ training experiments without touching legacy outputs.")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-processed", default=str(DEFAULT_SOURCE_PROCESSED))
    parser.add_argument("--seed-checkpoint", default=str(DEFAULT_SEED_CHECKPOINT))
    parser.add_argument("--base-config", default="configs/server_sup_manual_inc_041055.yaml")
    parser.add_argument("--experiments", default=None, help="Comma-separated experiment names. Default: all.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    run_root = Path(args.run_root).resolve()
    source_processed = Path(args.source_processed).resolve()
    seed_checkpoint = Path(args.seed_checkpoint).resolve()
    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = repo_dir / base_config_path
    base_cfg = yaml.safe_load(base_config_path.read_text())

    if run_root.exists() and not args.resume:
        raise FileExistsError(f"run root exists; use --resume to continue without overwrite: {run_root}")
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        (run_root / "experiments").mkdir()
        (run_root / "summary").mkdir()

    source_manifest = ensure_or_resume_inputs(run_root, source_processed, seed_checkpoint, args.resume)
    validate_isolated_inputs(source_manifest)
    if args.setup_only:
        print(f"[OK] isolated inputs ready: {run_root}")
        return 0

    selected_experiments = experiment_by_name(parse_experiment_names(args.experiments))
    manifests = []
    for exp in selected_experiments:
        manifest = run_experiment(
            repo_dir=repo_dir,
            python_exe=args.python_exe,
            run_root=run_root,
            base_cfg=base_cfg,
            exp=exp,
            source_manifest=source_manifest,
            args=args,
        )
        manifests.append(manifest)

    summary = summarize_and_select_best(run_root, manifests)
    print(f"[SUMMARY] {summary['results_csv']}")
    print(f"[BEST_MODEL] {summary.get('best_model_path')}")
    print(f"[BEST_PACKAGE] {summary.get('best_package_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
