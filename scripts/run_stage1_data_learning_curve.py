#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_goal_mode_16train_3holdout as split16  # noqa: E402
import run_goal_mode_19full as goal19  # noqa: E402


DEFAULT_STAGE1_CONFIG = Path(
    "/root/cej_isolated_runs/C/unsup_to_goal_retrain_20260601_012139_stage1/"
    "experiments/exp03_skeleton_aux_loss/config/train.yaml"
)
DEFAULT_STAGE2_SPLIT_ROOT = Path("/root/cej_isolated_runs/C/unsup_to_goal_val16holdout3_20260601_030405")
DEFAULT_UNSUP_CKPT = Path("/root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt")
THRESHOLDS = [0.20, 0.25, 0.30, 0.35]
FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]
SOURCE_ROUND_ROBIN = [
    "manual_inc_041055_001",
    "manual_new_025040_001",
    "manual6_from_run_unsup_001",
]
HARD_CASE_FIRST = "ToothFairy3F_041"
STAGE1_HOLDOUT_HARD_CASES = [
    "052-31/32/41/42/33",
    "040-42/41/32",
    "044-31/32/36/46",
]
STATUS_RANK = {"PASS_STRONG": 0, "PASS_USABLE": 1, "FAIL": 2}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[started_at] {now_iso()}\n")
        log.write("[cmd] " + " ".join(str(x) for x in cmd) + "\n")
        log.flush()
        proc = subprocess.run(list(cmd), cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[finished_at] {now_iso()}\n")
        log.write(f"[returncode] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(str(x) for x in cmd)}")


def label_for_fraction(fraction: float) -> str:
    return f"pct{int(round(fraction * 100)):03d}"


def count_tooth_dirs(processed_root: Path) -> int:
    return len(list(processed_root.glob("*/tooth_*")))


def symlink_dir(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"destination exists: {dst}")
    os.symlink(str(src), str(dst), target_is_directory=True)


def load_case_manifest(path: Path) -> List[Dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_split_manifest(split_root: Path, split_manifest: Dict, case_rows: List[Dict]) -> None:
    train_rows = [r for r in case_rows if r.get("split") == "train16"]
    holdout_rows = [r for r in case_rows if r.get("split") == "holdout3"]
    checks = {
        "train_cases": [r["case_id"] for r in train_rows] == split16.TRAIN_CASES,
        "holdout_cases": [r["case_id"] for r in holdout_rows] == split16.HOLDOUT_CASES,
        "train_case_count": int(split_manifest.get("train_case_count", 0) or 0) == len(split16.TRAIN_CASES),
        "holdout_case_count": int(split_manifest.get("holdout_case_count", 0) or 0) == len(split16.HOLDOUT_CASES),
        "train_tooth_dir_count": int(split_manifest.get("train_tooth_dir_count", 0) or 0)
        == split16.EXPECTED_TRAIN_TOOTH_DIRS,
        "holdout_tooth_dir_count": int(split_manifest.get("holdout_tooth_dir_count", 0) or 0)
        == split16.EXPECTED_HOLDOUT_TOOTH_DIRS,
        "processed_train16_exists": Path(split_manifest.get("processed_train16", "")).exists(),
        "processed_holdout3_exists": Path(split_manifest.get("processed_holdout3", "")).exists(),
        "case_manifest_exists": (split_root / "input/case_manifest.csv").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"split manifest checks failed: {json.dumps(checks, ensure_ascii=False)}")


def nested_case_order(case_rows: List[Dict]) -> List[str]:
    train_rows = [r for r in case_rows if r.get("split") == "train16"]
    by_source: Dict[str, List[str]] = {}
    for row in train_rows:
        by_source.setdefault(row["source"], []).append(row["case_id"])
    for source, cases in by_source.items():
        cases.sort()
        if HARD_CASE_FIRST in cases:
            cases.remove(HARD_CASE_FIRST)
            cases.insert(0, HARD_CASE_FIRST)
    order = []
    while len(order) < len(train_rows):
        progressed = False
        for source in SOURCE_ROUND_ROBIN:
            cases = by_source.get(source, [])
            if cases:
                order.append(cases.pop(0))
                progressed = True
        for source in sorted(set(by_source) - set(SOURCE_ROUND_ROBIN)):
            cases = by_source.get(source, [])
            if cases:
                order.append(cases.pop(0))
                progressed = True
        if not progressed:
            break
    if sorted(order) != sorted([r["case_id"] for r in train_rows]):
        raise RuntimeError(f"nested subset order lost cases: {order}")
    return order


def subset_size(total_cases: int, fraction: float) -> int:
    if fraction >= 1.0:
        return total_cases
    return max(1, int(round(total_cases * fraction)))


def write_subset_case_manifest(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["split", "case_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def setup_inputs(run_root: Path, split_root: Path, unsup_ckpt: Path, resume: bool) -> Dict:
    manifest_path = run_root / "input/source_manifest.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"--resume requested but source manifest is missing: {manifest_path}")
        return read_json(manifest_path)

    input_dir = run_root / "input"
    if input_dir.exists() or input_dir.is_symlink():
        raise FileExistsError(f"input directory exists, refusing to overwrite: {input_dir}")
    if not unsup_ckpt.exists():
        raise FileNotFoundError(f"unsup checkpoint missing: {unsup_ckpt}")

    split_manifest_path = split_root / "input/source_manifest.json"
    case_manifest_path = split_root / "input/case_manifest.csv"
    split_manifest = read_json(split_manifest_path)
    case_rows = load_case_manifest(case_manifest_path)
    validate_split_manifest(split_root, split_manifest, case_rows)

    input_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir = input_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unsup_copy = ckpt_dir / "unsup_last.pt"
    shutil.copy2(unsup_ckpt, unsup_copy)
    if sha256_file(unsup_ckpt) != sha256_file(unsup_copy):
        raise RuntimeError("unsup checkpoint copy hash mismatch")

    processed_train_src = Path(split_manifest["processed_train16"])
    processed_holdout_src = Path(split_manifest["processed_holdout3"])
    processed_holdout = input_dir / "processed_holdout3"
    symlink_dir(processed_holdout_src, processed_holdout)
    shutil.copy2(case_manifest_path, input_dir / "case_manifest_full16_holdout3.csv")

    order = nested_case_order(case_rows)
    row_by_case = {r["case_id"]: r for r in case_rows}
    subsets = []
    for fraction in FRACTIONS:
        label = label_for_fraction(fraction)
        n_cases = subset_size(len(order), fraction)
        cases = order[:n_cases]
        train_dir = input_dir / "subsets" / label / "processed_train"
        train_dir.mkdir(parents=True, exist_ok=False)
        for case_id in cases:
            symlink_dir(processed_train_src / case_id, train_dir / case_id)
        tooth_count = count_tooth_dirs(train_dir)
        subset_rows = [{**row_by_case[c], "learning_curve_subset": label} for c in cases]
        write_subset_case_manifest(input_dir / "subsets" / label / "case_manifest.csv", subset_rows)
        subsets.append(
            {
                "label": label,
                "target_fraction": fraction,
                "target_percent": int(round(fraction * 100)),
                "case_count": n_cases,
                "actual_case_fraction": n_cases / len(order),
                "tooth_dir_count": tooth_count,
                "actual_tooth_fraction": tooth_count / split16.EXPECTED_TRAIN_TOOTH_DIRS,
                "cases": cases,
                "processed_train": str(train_dir),
                "case_manifest": str(input_dir / "subsets" / label / "case_manifest.csv"),
            }
        )

    manifest = {
        "created_at": now_iso(),
        "run_root": str(run_root),
        "objective": "Stage1-only data learning curve on fixed 3-case holdout; no Stage2 training.",
        "split_mode": "train16_holdout3_learning_curve",
        "stage2_split_root": str(split_root),
        "stage2_split_manifest": str(split_manifest_path),
        "source_processed_train16": str(processed_train_src),
        "source_processed_holdout3": str(processed_holdout_src),
        "processed_holdout3": str(processed_holdout),
        "train_cases_full": split16.TRAIN_CASES,
        "holdout_cases": split16.HOLDOUT_CASES,
        "train_case_count_full": len(split16.TRAIN_CASES),
        "holdout_case_count": len(split16.HOLDOUT_CASES),
        "train_tooth_dir_count_full": split16.EXPECTED_TRAIN_TOOTH_DIRS,
        "holdout_tooth_dir_count": count_tooth_dirs(processed_holdout),
        "nested_case_order": order,
        "subset_policy": (
            "case-level nested subsets; source round-robin order manual_inc -> manual_new -> manual6; "
            "ToothFairy3F_041 kept first within manual_inc so the Stage1 hard-case sampler remains represented."
        ),
        "fractions": FRACTIONS,
        "subsets": subsets,
        "unsup_checkpoint_source": str(unsup_ckpt),
        "unsup_checkpoint_copy": str(unsup_copy),
        "unsup_checkpoint_source_sha256": sha256_file(unsup_ckpt),
        "unsup_checkpoint_copy_sha256": sha256_file(unsup_copy),
        "thresholds": THRESHOLDS,
    }
    if manifest["holdout_tooth_dir_count"] != split16.EXPECTED_HOLDOUT_TOOTH_DIRS:
        raise RuntimeError(f"holdout tooth count mismatch: {manifest['holdout_tooth_dir_count']}")
    write_json(manifest_path, manifest)
    return manifest


def render_stage1_config(stage1_cfg: Dict, subset: Dict, source_manifest: Dict, exp_root: Path, device: str) -> Dict:
    cfg = json.loads(json.dumps(stage1_cfg))
    cfg.setdefault("project", {})["device"] = device
    data = cfg.setdefault("data", {})
    data["processed_dir"] = subset["processed_train"]
    data["holdout_processed_dir"] = source_manifest["processed_holdout3"]
    data["output_dir"] = str(exp_root / "outputs")
    data["processed_format"] = "nii.gz"
    data["use_tooth_mask_channel"] = True

    train = cfg.setdefault("train", {})
    train["pretrained_ckpt"] = source_manifest["unsup_checkpoint_copy"]
    train["pretrained_strict"] = False
    train["learning_curve_label"] = subset["label"]
    train["learning_curve_target_fraction"] = subset["target_fraction"]
    train["learning_curve_case_count"] = subset["case_count"]
    train["learning_curve_tooth_dir_count"] = subset["tooth_dir_count"]
    train["learning_curve_train_monitor_fit_pred_curve"] = False

    infer = cfg.setdefault("infer", {})
    infer["ckpt_path"] = str(exp_root / "outputs/train/checkpoints/best.pt")
    infer["threshold_theta"] = 0.30
    infer["constrain_curve_to_tooth_mask"] = False
    infer["constrain_curve_to_tooth_surface"] = False
    infer["keep_lcc_for_curve"] = False
    # Keep training monitor lightweight. Final threshold sweep below forces
    # fit_pred_curve=True, matching the formal evaluation path.
    infer["fit_pred_curve"] = False
    infer["fit_pred_curve_step_mm"] = 0.2
    infer["fit_pred_curve_closed"] = True
    infer["fit_pred_curve_smooth"] = 0.0
    infer["fit_pred_curve_min_points"] = 8
    infer["use_prior_gating"] = False

    holdout = cfg.setdefault("holdout_eval", {})
    holdout["hard_cases"] = STAGE1_HOLDOUT_HARD_CASES
    holdout["threshold_theta"] = 0.30
    holdout["fit_pred_curve_step_mm"] = 0.2
    holdout["fit_pred_curve_closed"] = True
    holdout["fit_pred_curve_smooth"] = 0.0
    holdout["fit_pred_curve_min_points"] = 8
    holdout["wrap_tau_mm"] = 1.0
    holdout["gt_peak_threshold"] = 0.95
    holdout["gt_rel_threshold"] = 0.95

    eval_cfg = cfg.setdefault("eval", {})
    eval_cfg["eval_taus_mm"] = [0.5, 1.0, 1.5, 2.0]
    eval_cfg["prediction_name"] = "C_pred_fit.nii.gz"

    viz = cfg.setdefault("viz", {})
    viz["enable_2d"] = False
    viz["enable_3d"] = False
    viz["max_cases"] = 3
    viz["max_teeth_per_case"] = 64
    viz["use_error_colormap_for_gt_points"] = True
    return cfg


def create_eval_config(train_cfg: Dict, checkpoint: Path, threshold: float, out_dir: Path, source_manifest: Dict) -> Dict:
    cfg = json.loads(json.dumps(train_cfg))
    cfg["data"]["processed_dir"] = source_manifest["processed_holdout3"]
    cfg["data"]["holdout_processed_dir"] = source_manifest["processed_holdout3"]
    cfg["data"]["output_dir"] = str(out_dir)
    cfg.setdefault("infer", {})["ckpt_path"] = str(checkpoint)
    cfg["infer"]["threshold_theta"] = float(threshold)
    cfg["infer"]["fit_pred_curve"] = True
    cfg.setdefault("holdout_eval", {})["threshold_theta"] = float(threshold)
    return cfg


def holdout_source_manifest(source_manifest: Dict) -> Dict:
    return {
        "train_cases": source_manifest["train_cases_full"],
        "holdout_cases": source_manifest["holdout_cases"],
        "train_case_count": source_manifest["train_case_count_full"],
        "holdout_case_count": source_manifest["holdout_case_count"],
        "train_tooth_dir_count": source_manifest["train_tooth_dir_count_full"],
        "holdout_tooth_dir_count": source_manifest["holdout_tooth_dir_count"],
        "pretrained_source": source_manifest["unsup_checkpoint_source"],
        "pretrained_copy": source_manifest["unsup_checkpoint_copy"],
        "processed_holdout3": source_manifest["processed_holdout3"],
        "processed_train16": source_manifest["source_processed_train16"],
        "tooth_dir_count": source_manifest["train_tooth_dir_count_full"],
    }


def learning_curve_validation_checks(summary: Dict, per_tooth_rows: List[Dict], failed_or_bad_teeth_count: int) -> Dict:
    holdout_case_count = split16.holdout_case_count_from_rows(per_tooth_rows)
    holdout_tooth_count = int(summary.get("tooth_count", 0) or 0)
    missing_prediction_count = int(summary.get("missing_prediction_count", 0) or 0)
    no_curve_count = int(summary.get("no_curve_count", 0) or 0)
    failed_tooth_count = int(summary.get("failed_tooth_count", 0) or 0)
    mean_dist_mm = float(summary.get("mean_dist_mm", summary.get("mean", float("inf"))))
    p95_dist_mm = goal19.summary_p95(summary)
    sr1 = goal19.summary_sr1(summary)
    base = {
        "holdout_case_count_is_3": holdout_case_count == 3,
        "holdout_tooth_count_is_88": holdout_tooth_count == split16.EXPECTED_HOLDOUT_TOOTH_DIRS,
        "missing_prediction_count_is_0": missing_prediction_count == 0,
        "no_curve_count_is_0": no_curve_count == 0,
    }
    strong = {
        **base,
        "failed_tooth_count_is_0": failed_tooth_count == 0,
        "failed_or_bad_teeth_count_is_0": int(failed_or_bad_teeth_count) == 0,
        "mean_dist_mm_lte_0_50": mean_dist_mm <= 0.50,
        "p95_dist_mm_lte_1_00": p95_dist_mm <= 1.00,
        "sr_at_1mm_gte_0_90": sr1 >= 0.90,
    }
    usable = {
        **base,
        "mean_dist_mm_lte_0_75": mean_dist_mm <= 0.75,
        "p95_dist_mm_lte_1_50": p95_dist_mm <= 1.50,
        "sr_at_1mm_gte_0_80": sr1 >= 0.80,
    }
    fail_triggers = {
        "holdout_case_count_wrong": holdout_case_count != 3,
        "holdout_tooth_count_wrong": holdout_tooth_count != split16.EXPECTED_HOLDOUT_TOOTH_DIRS,
        "missing_prediction_count_gt_0": missing_prediction_count > 0,
        "no_curve_count_gt_0": no_curve_count > 0,
        "p95_dist_mm_gt_1_50": p95_dist_mm > 1.50,
        "sr_at_1mm_lt_0_80": sr1 < 0.80,
    }
    if all(strong.values()):
        status = "PASS_STRONG"
    elif all(usable.values()) and not any(fail_triggers.values()):
        status = "PASS_USABLE"
    else:
        status = "FAIL"
    return {
        "status": status,
        "strong_criteria": strong,
        "usable_criteria": usable,
        "fail_triggers": fail_triggers,
        "holdout_case_count": holdout_case_count,
        "holdout_tooth_count": holdout_tooth_count,
        "missing_prediction_count": missing_prediction_count,
        "no_curve_count": no_curve_count,
        "failed_tooth_count": failed_tooth_count,
        "failed_or_bad_teeth_count": int(failed_or_bad_teeth_count),
        "mean_dist_mm": mean_dist_mm,
        "p95_dist_mm": p95_dist_mm,
        "sr@1.0mm": sr1,
    }


def sweep_thresholds(
    repo_dir: Path,
    python_exe: str,
    run_root: Path,
    exp_root: Path,
    cfg: Dict,
    source_manifest: Dict,
    force_eval: bool,
) -> Dict:
    ckpt_path = exp_root / "outputs/train/checkpoints/best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best checkpoint missing before sweep: {ckpt_path}")
    sweep_root = exp_root / "threshold_sweep_best"
    sweep_root.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates: List[Tuple] = []
    for threshold in THRESHOLDS:
        label = f"theta_{threshold:.2f}"
        out_dir = sweep_root / label
        if force_eval and out_dir.exists():
            shutil.rmtree(out_dir)
        cfg_path = exp_root / "config" / f"best_{label}.yaml"
        eval_cfg = create_eval_config(cfg, ckpt_path, threshold, out_dir, source_manifest)
        write_yaml(cfg_path, eval_cfg)
        metrics_path = out_dir / "eval/metrics_summary.json"
        if not metrics_path.exists():
            run_cmd([python_exe, "-m", "src.infer", "--config", str(cfg_path)], repo_dir, exp_root / "logs" / f"infer_{label}.log")
            run_cmd([python_exe, "-m", "src.eval", "--config", str(cfg_path)], repo_dir, exp_root / "logs" / f"eval_{label}.log")
        eval_dir = out_dir / "eval"
        summary = goal19.load_summary(eval_dir)
        per_tooth = eval_dir / "metrics_per_tooth.csv"
        per_tooth_rows = split16.load_csv_rows(per_tooth)
        bad_rows = goal19.load_bad_rows(per_tooth)
        failed_csv = eval_dir / "failed_or_bad_teeth.csv"
        goal19.write_failed_rows(failed_csv, bad_rows)
        checks = learning_curve_validation_checks(summary, per_tooth_rows, len(bad_rows))
        row = {
            "threshold": threshold,
            "output_dir": str(out_dir),
            "config": str(cfg_path),
            "metrics_summary": str(metrics_path),
            "metrics_per_tooth": str(per_tooth),
            "failed_or_bad_teeth": str(failed_csv),
            "mean_dist_mm": checks["mean_dist_mm"],
            "p95_dist_mm": checks["p95_dist_mm"],
            "sr@1.0mm": checks["sr@1.0mm"],
            "missing_prediction_count": checks["missing_prediction_count"],
            "no_curve_count": checks["no_curve_count"],
            "failed_tooth_count": checks["failed_tooth_count"],
            "failed_or_bad_teeth_count": checks["failed_or_bad_teeth_count"],
            "validation_status": checks["status"],
        }
        rows.append(row)
        candidates.append(
            (
                STATUS_RANK.get(checks["status"], 9),
                0 if checks["missing_prediction_count"] == 0 else 1,
                0 if checks["no_curve_count"] == 0 else 1,
                0 if checks["failed_or_bad_teeth_count"] == 0 else 1,
                checks["p95_dist_mm"],
                -checks["sr@1.0mm"],
                threshold,
                row,
                checks,
            )
        )
    candidates.sort(key=lambda item: item[:7])
    selected = candidates[0]
    sweep_csv = exp_root / "summary/threshold_sweep_best.csv"
    sweep_csv.parent.mkdir(parents=True, exist_ok=True)
    with sweep_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    final_root = exp_root / "final_best"
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)
    selected_out = Path(selected[7]["output_dir"])
    shutil.copytree(selected_out / "infer", final_root / "infer")
    shutil.copytree(selected_out / "eval", final_root / "eval")
    return {
        "sweep_csv": str(sweep_csv),
        "rows": rows,
        "selected_row": selected[7],
        "selected_validation": selected[8],
        "final_root": str(final_root),
    }


def read_train_manifest(exp_root: Path) -> Dict:
    path = exp_root / "outputs/train/train_manifest.json"
    return read_json(path) if path.exists() else {}


def run_subset(
    repo_dir: Path,
    python_exe: str,
    run_root: Path,
    stage1_cfg: Dict,
    source_manifest: Dict,
    subset: Dict,
    device: str,
    skip_train: bool,
    train_only: bool,
    force_eval: bool,
) -> Dict:
    exp_root = run_root / "experiments" / subset["label"]
    exp_root.mkdir(parents=True, exist_ok=True)
    (exp_root / "config").mkdir(exist_ok=True)
    (exp_root / "logs").mkdir(exist_ok=True)
    (exp_root / "summary").mkdir(exist_ok=True)
    cfg = render_stage1_config(stage1_cfg, subset, source_manifest, exp_root, device)
    cfg_path = exp_root / "config/train.yaml"
    write_yaml(cfg_path, cfg)
    best_ckpt = exp_root / "outputs/train/checkpoints/best.pt"
    if not skip_train and not best_ckpt.exists():
        run_cmd([python_exe, "-m", "src.train_loss_earlystop", "--config", str(cfg_path)], repo_dir, exp_root / "logs/train_loss_earlystop.log")
    train_manifest = read_train_manifest(exp_root)
    result = {
        "label": subset["label"],
        "target_fraction": subset["target_fraction"],
        "actual_case_fraction": subset["actual_case_fraction"],
        "actual_tooth_fraction": subset["actual_tooth_fraction"],
        "case_count": subset["case_count"],
        "tooth_dir_count": subset["tooth_dir_count"],
        "cases": subset["cases"],
        "config": str(cfg_path),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(exp_root / "outputs/train/checkpoints/last.pt"),
        "metrics_csv": str(exp_root / "outputs/train/metrics.csv"),
        "train_manifest": str(exp_root / "outputs/train/train_manifest.json"),
        "best_epoch": train_manifest.get("best_epoch"),
        "best_composite_score": train_manifest.get("best_composite_score"),
        "best_holdout_loss": train_manifest.get("best_holdout_loss"),
        "best_holdout_manual_point_p95": train_manifest.get("best_holdout_manual_point_p95"),
        "best_holdout_manual_point_sr1": train_manifest.get("best_holdout_manual_point_sr1"),
    }
    if train_only:
        write_json(exp_root / "summary/subset_result.json", result)
        return result
    sweep = sweep_thresholds(repo_dir, python_exe, run_root, exp_root, cfg, source_manifest, force_eval)
    selected = sweep["selected_validation"]
    result.update(
        {
            "threshold_sweep": sweep["sweep_csv"],
            "selected_threshold": sweep["selected_row"]["threshold"],
            "validation_status": selected["status"],
            "mean_dist_mm": selected["mean_dist_mm"],
            "p95_dist_mm": selected["p95_dist_mm"],
            "sr@1.0mm": selected["sr@1.0mm"],
            "missing_prediction_count": selected["missing_prediction_count"],
            "no_curve_count": selected["no_curve_count"],
            "failed_tooth_count": selected["failed_tooth_count"],
            "failed_or_bad_teeth_count": selected["failed_or_bad_teeth_count"],
            "final_root": sweep["final_root"],
        }
    )
    write_json(exp_root / "summary/subset_result.json", result)
    return result


def write_curve_csv(run_root: Path, results: List[Dict]) -> Path:
    csv_path = run_root / "summary/learning_curve.csv"
    fields = [
        "label",
        "target_fraction",
        "actual_case_fraction",
        "actual_tooth_fraction",
        "case_count",
        "tooth_dir_count",
        "selected_threshold",
        "validation_status",
        "mean_dist_mm",
        "p95_dist_mm",
        "sr@1.0mm",
        "missing_prediction_count",
        "no_curve_count",
        "failed_tooth_count",
        "failed_or_bad_teeth_count",
        "best_epoch",
        "best_composite_score",
        "best_holdout_loss",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return csv_path


def _by_fraction(results: List[Dict]) -> Dict[float, Dict]:
    return {round(float(r["target_fraction"]), 2): r for r in results if r.get("p95_dist_mm") is not None}


def analyze_learning_curve(results: List[Dict]) -> Dict:
    by_fraction = _by_fraction(results)
    required = [0.60, 0.80, 1.00]
    if not all(f in by_fraction for f in required):
        return {
            "status": "INCOMPLETE",
            "conclusion": "Learning curve is incomplete; need 60%, 80%, and 100% results before judging data sufficiency.",
        }
    r60 = by_fraction[0.60]
    r80 = by_fraction[0.80]
    r100 = by_fraction[1.00]
    p95_gain_60_to_100 = float(r60["p95_dist_mm"]) - float(r100["p95_dist_mm"])
    p95_gain_80_to_100 = float(r80["p95_dist_mm"]) - float(r100["p95_dist_mm"])
    sr1_gain_60_to_100 = float(r100["sr@1.0mm"]) - float(r60["sr@1.0mm"])
    sr1_gain_80_to_100 = float(r100["sr@1.0mm"]) - float(r80["sr@1.0mm"])
    plateau = (
        abs(p95_gain_60_to_100) < 0.10
        and abs(p95_gain_80_to_100) < 0.05
        and abs(sr1_gain_60_to_100) < 0.02
        and abs(sr1_gain_80_to_100) < 0.01
    )
    still_improving = p95_gain_80_to_100 >= 0.10 or sr1_gain_80_to_100 >= 0.02
    if plateau:
        status = "PLATEAU"
        conclusion = (
            "Data curve is flat from 60% to 100%; additional data is unlikely to be the main lever before improving model/postprocess."
        )
    elif still_improving:
        status = "DATA_LIMITED"
        conclusion = (
            "The curve is still improving from 80% to 100%; more labeled data or stronger augmentation is likely useful."
        )
    else:
        status = "BORDERLINE_OR_NOISY"
        conclusion = (
            "The curve does not show a clean plateau or a strong 80%-to-100% gain; repeat subsets/seeds before making a hard data sufficiency call."
        )
    return {
        "status": status,
        "heuristic": (
            "Plateau if p95 gains are <0.10mm from 60->100 and <0.05mm from 80->100, "
            "and sr@1 gains are <0.02 from 60->100 and <0.01 from 80->100. "
            "Data-limited if 80->100 gains >=0.10mm p95 or >=0.02 sr@1."
        ),
        "p95_gain_60_to_100_mm": p95_gain_60_to_100,
        "p95_gain_80_to_100_mm": p95_gain_80_to_100,
        "sr1_gain_60_to_100": sr1_gain_60_to_100,
        "sr1_gain_80_to_100": sr1_gain_80_to_100,
        "conclusion": conclusion,
    }


def write_report(run_root: Path, source_manifest: Dict, results: List[Dict], curve_csv: Path) -> Dict:
    analysis = analyze_learning_curve(results)
    report = {
        "created_at": now_iso(),
        "run_root": str(run_root),
        "objective": "Stage1-only data learning curve to judge whether adding data still improves fixed-holdout performance.",
        "source_manifest": str(run_root / "input/source_manifest.json"),
        "learning_curve_csv": str(curve_csv),
        "subset_policy": source_manifest["subset_policy"],
        "holdout_cases": source_manifest["holdout_cases"],
        "thresholds": THRESHOLDS,
        "results": results,
        "analysis": analysis,
        "data_sufficiency_conclusion": analysis["conclusion"],
    }
    write_json(run_root / "summary/data_learning_curve_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage1-only data learning curve on fixed 3-case holdout.")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--stage1-config", default=str(DEFAULT_STAGE1_CONFIG))
    parser.add_argument("--stage2-split-root", default=str(DEFAULT_STAGE2_SPLIT_ROOT))
    parser.add_argument("--unsup-ckpt", default=str(DEFAULT_UNSUP_CKPT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--only-label", action="append", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    run_root = (
        Path(args.run_root).resolve()
        if args.run_root
        else Path(f"/root/cej_isolated_runs/C/stage1_data_learning_curve_{timestamp()}").resolve()
    )
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"run root exists; use --resume to continue: {run_root}")
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        (run_root / "summary").mkdir()

    source_manifest = setup_inputs(
        run_root,
        Path(args.stage2_split_root).resolve(),
        Path(args.unsup_ckpt).resolve(),
        args.resume,
    )
    write_json(
        run_root / "summary/setup_audit.json",
        {
            "train_case_count_full_is_16": source_manifest["train_case_count_full"] == 16,
            "holdout_case_count_is_3": source_manifest["holdout_case_count"] == 3,
            "train_tooth_count_full_is_497": source_manifest["train_tooth_dir_count_full"] == 497,
            "holdout_tooth_count_is_88": source_manifest["holdout_tooth_dir_count"] == 88,
            "unsup_hash_match": source_manifest["unsup_checkpoint_source_sha256"]
            == source_manifest["unsup_checkpoint_copy_sha256"],
            "subset_labels": [s["label"] for s in source_manifest["subsets"]],
            "subset_case_counts": {s["label"]: s["case_count"] for s in source_manifest["subsets"]},
            "subset_tooth_counts": {s["label"]: s["tooth_dir_count"] for s in source_manifest["subsets"]},
        },
    )
    if args.setup_only:
        print(f"[OK] setup complete: {run_root}")
        print(f"[AUDIT] {run_root / 'summary/setup_audit.json'}")
        return 0

    stage1_cfg_path = Path(args.stage1_config).resolve()
    stage1_cfg = yaml.safe_load(stage1_cfg_path.read_text(encoding="utf-8"))
    wanted = set(args.only_label or [])
    subsets = [s for s in source_manifest["subsets"] if not wanted or s["label"] in wanted]
    results = []
    for subset in subsets:
        print(
            f"[RUN_SUBSET] {subset['label']} cases={subset['case_count']} "
            f"teeth={subset['tooth_dir_count']} fraction={subset['actual_tooth_fraction']:.3f}",
            flush=True,
        )
        result = run_subset(
            repo_dir,
            args.python_exe,
            run_root,
            stage1_cfg,
            source_manifest,
            subset,
            args.device,
            args.skip_train,
            args.train_only,
            args.force_eval,
        )
        results.append(result)
        if result.get("p95_dist_mm") is not None:
            print(
                f"[RESULT] {subset['label']} theta={result['selected_threshold']} "
                f"mean={result['mean_dist_mm']} p95={result['p95_dist_mm']} "
                f"sr1={result['sr@1.0mm']} status={result['validation_status']}",
                flush=True,
            )

    if args.train_only:
        write_json(run_root / "summary/train_only_results.json", {"results": results})
        print(f"[TRAIN_ONLY_RESULTS] {run_root / 'summary/train_only_results.json'}")
        return 0

    all_result_paths = sorted((run_root / "experiments").glob("*/summary/subset_result.json"))
    merged = [read_json(path) for path in all_result_paths]
    merged.sort(key=lambda r: float(r["target_fraction"]))
    curve_csv = write_curve_csv(run_root, merged)
    report = write_report(run_root, source_manifest, merged, curve_csv)
    print(f"[CURVE_CSV] {curve_csv}")
    print(f"[CONCLUSION_STATUS] {report['analysis']['status']}")
    print(f"[CONCLUSION] {report['data_sufficiency_conclusion']}")
    print(f"[REPORT] {run_root / 'summary/data_learning_curve_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
