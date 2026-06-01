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


DEFAULT_STAGE1_RUN_ROOT = Path("/root/cej_isolated_runs/C/unsup_to_goal_retrain_20260601_012139_stage1")
DEFAULT_STAGE1_CONFIG = (
    DEFAULT_STAGE1_RUN_ROOT / "experiments/exp03_skeleton_aux_loss/config/train.yaml"
)
DEFAULT_STAGE1_LAST = (
    DEFAULT_STAGE1_RUN_ROOT
    / "experiments/exp03_skeleton_aux_loss/outputs/train/checkpoints/last.pt"
)
DEFAULT_STAGE2_SPLIT_ROOT = Path("/root/cej_isolated_runs/C/unsup_to_goal_val16holdout3_20260601_030405")
DEFAULT_STAGE2_BASELINES = [
    Path("/root/cej_isolated_runs/C/unsup_to_goal_val16holdout3_20260601_030405"),
    Path("/root/cej_isolated_runs/C/unsup_to_goal_val16holdout3_max20_min7_20260601_031934"),
]
THRESHOLDS = [0.20, 0.25, 0.30, 0.35]
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


def count_tooth_dirs(processed_root: Path) -> int:
    return len(list(processed_root.glob("*/tooth_*")))


def symlink_dir(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"destination exists: {dst}")
    os.symlink(str(src), str(dst), target_is_directory=True)


def validate_split_manifest(split_root: Path, split_manifest: Dict) -> None:
    checks = {
        "train_cases": list(split_manifest.get("train_cases", [])) == split16.TRAIN_CASES,
        "holdout_cases": list(split_manifest.get("holdout_cases", [])) == split16.HOLDOUT_CASES,
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


def setup_inputs(
    run_root: Path,
    split_root: Path,
    stage1_best: Path,
    stage1_last: Path,
    resume: bool,
) -> Dict:
    manifest_path = run_root / "input/source_manifest.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"--resume requested but source manifest is missing: {manifest_path}")
        return read_json(manifest_path)

    input_dir = run_root / "input"
    if input_dir.exists() or input_dir.is_symlink():
        raise FileExistsError(f"input directory exists, refusing to overwrite: {input_dir}")
    if not stage1_best.exists():
        raise FileNotFoundError(f"Stage1 best checkpoint missing: {stage1_best}")
    if not stage1_last.exists():
        raise FileNotFoundError(f"Stage1 last checkpoint missing: {stage1_last}")

    split_manifest_path = split_root / "input/source_manifest.json"
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"split source manifest missing: {split_manifest_path}")
    split_manifest = read_json(split_manifest_path)
    validate_split_manifest(split_root, split_manifest)

    processed_train_src = Path(split_manifest["processed_train16"])
    processed_holdout_src = Path(split_manifest["processed_holdout3"])
    input_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir = input_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    processed_train = input_dir / "processed_train16"
    processed_holdout = input_dir / "processed_holdout3"
    symlink_dir(processed_train_src, processed_train)
    symlink_dir(processed_holdout_src, processed_holdout)

    case_manifest_src = split_root / "input/case_manifest.csv"
    shutil.copy2(case_manifest_src, input_dir / "case_manifest.csv")
    best_copy = ckpt_dir / "stage1_best.pt"
    last_copy = ckpt_dir / "stage1_last.pt"
    shutil.copy2(stage1_best, best_copy)
    shutil.copy2(stage1_last, last_copy)

    train_tooth_count = count_tooth_dirs(processed_train)
    holdout_tooth_count = count_tooth_dirs(processed_holdout)
    manifest = {
        "created_at": now_iso(),
        "run_root": str(run_root),
        "objective": "Stage1-only direct eval on the 3 holdout cases using the same 16 train / 3 holdout split as Stage2 validation.",
        "split_mode": "train16_holdout3",
        "stage2_split_root": str(split_root),
        "stage2_split_manifest": str(split_manifest_path),
        "processed_train16": str(processed_train),
        "processed_holdout3": str(processed_holdout),
        "source_processed_train16": str(processed_train_src),
        "source_processed_holdout3": str(processed_holdout_src),
        "train_cases": split16.TRAIN_CASES,
        "holdout_cases": split16.HOLDOUT_CASES,
        "train_case_count": len(split16.TRAIN_CASES),
        "holdout_case_count": len(split16.HOLDOUT_CASES),
        "train_tooth_dir_count": train_tooth_count,
        "holdout_tooth_dir_count": holdout_tooth_count,
        "total_tooth_dir_count": train_tooth_count + holdout_tooth_count,
        "expected_train_tooth_dir_count": split16.EXPECTED_TRAIN_TOOTH_DIRS,
        "expected_holdout_tooth_dir_count": split16.EXPECTED_HOLDOUT_TOOTH_DIRS,
        "tooth_dir_count": train_tooth_count,
        "case_count": len(split16.TRAIN_CASES),
        "pretrained_source": str(stage1_best),
        "pretrained_copy": str(best_copy),
        "pretrained_source_sha256": sha256_file(stage1_best),
        "pretrained_copy_sha256": sha256_file(best_copy),
        "stage1_best_source": str(stage1_best),
        "stage1_best_copy": str(best_copy),
        "stage1_best_source_sha256": sha256_file(stage1_best),
        "stage1_best_copy_sha256": sha256_file(best_copy),
        "stage1_last_source": str(stage1_last),
        "stage1_last_copy": str(last_copy),
        "stage1_last_source_sha256": sha256_file(stage1_last),
        "stage1_last_copy_sha256": sha256_file(last_copy),
        "thresholds": THRESHOLDS,
    }
    mismatches = {
        "train_tooth_count": train_tooth_count == split16.EXPECTED_TRAIN_TOOTH_DIRS,
        "holdout_tooth_count": holdout_tooth_count == split16.EXPECTED_HOLDOUT_TOOTH_DIRS,
        "stage1_best_hash": manifest["stage1_best_source_sha256"] == manifest["stage1_best_copy_sha256"],
        "stage1_last_hash": manifest["stage1_last_source_sha256"] == manifest["stage1_last_copy_sha256"],
    }
    if not all(mismatches.values()):
        raise RuntimeError(f"input setup checks failed: {json.dumps(mismatches, ensure_ascii=False)}")
    write_json(manifest_path, manifest)
    return manifest


def render_eval_base_config(stage1_cfg: Dict, source_manifest: Dict, run_root: Path, device: str) -> Dict:
    cfg = json.loads(json.dumps(stage1_cfg))
    cfg.setdefault("project", {})["device"] = device
    data = cfg.setdefault("data", {})
    data["processed_dir"] = source_manifest["processed_holdout3"]
    data["holdout_processed_dir"] = source_manifest["processed_holdout3"]
    data["output_dir"] = str(run_root / "outputs_direct")
    data["processed_format"] = "nii.gz"
    data["use_tooth_mask_channel"] = True

    infer = cfg.setdefault("infer", {})
    infer["threshold_theta"] = 0.30
    infer["constrain_curve_to_tooth_mask"] = False
    infer["constrain_curve_to_tooth_surface"] = False
    infer["keep_lcc_for_curve"] = False
    infer["fit_pred_curve"] = True
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


def create_eval_config(base_cfg: Dict, checkpoint: Path, threshold: float, out_dir: Path) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["data"]["processed_dir"] = base_cfg["data"]["processed_dir"]
    cfg["data"]["holdout_processed_dir"] = base_cfg["data"]["processed_dir"]
    cfg["data"]["output_dir"] = str(out_dir)
    cfg.setdefault("infer", {})["ckpt_path"] = str(checkpoint)
    cfg["infer"]["threshold_theta"] = float(threshold)
    cfg.setdefault("holdout_eval", {})["threshold_theta"] = float(threshold)
    return cfg


def select_checkpoint(rows: List[Tuple]) -> Tuple:
    rows.sort(key=lambda item: item[:7])
    return rows[0]


def summarize_checkpoint_eval(
    repo_dir: Path,
    python_exe: str,
    run_root: Path,
    base_cfg: Dict,
    source_manifest: Dict,
    checkpoint_label: str,
    checkpoint: Path,
    force: bool,
) -> Dict:
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint_label} checkpoint missing: {checkpoint}")
    sweep_root = run_root / f"threshold_sweep_{checkpoint_label}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for threshold in THRESHOLDS:
        label = f"theta_{threshold:.2f}"
        out_dir = sweep_root / label
        if force and out_dir.exists():
            shutil.rmtree(out_dir)
        cfg_path = run_root / "config" / f"{checkpoint_label}_{label}.yaml"
        cfg = create_eval_config(base_cfg, checkpoint, threshold, out_dir)
        write_yaml(cfg_path, cfg)
        metrics_path = out_dir / "eval/metrics_summary.json"
        if not metrics_path.exists():
            run_cmd([python_exe, "-m", "src.infer", "--config", str(cfg_path)], repo_dir, run_root / "logs" / f"infer_{checkpoint_label}_{label}.log")
            run_cmd([python_exe, "-m", "src.eval", "--config", str(cfg_path)], repo_dir, run_root / "logs" / f"eval_{checkpoint_label}_{label}.log")

        eval_dir = out_dir / "eval"
        summary = goal19.load_summary(eval_dir)
        per_tooth = eval_dir / "metrics_per_tooth.csv"
        per_tooth_rows = split16.load_csv_rows(per_tooth)
        bad_rows = goal19.load_bad_rows(per_tooth)
        failed_csv = eval_dir / "failed_or_bad_teeth.csv"
        goal19.write_failed_rows(failed_csv, bad_rows)
        checks = split16.validation_checks(source_manifest, summary, per_tooth_rows, len(bad_rows))
        row = {
            "checkpoint_label": checkpoint_label,
            "checkpoint": str(checkpoint),
            "threshold": threshold,
            "output_dir": str(out_dir),
            "config": str(cfg_path),
            "metrics_summary": str(eval_dir / "metrics_summary.json"),
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

    selected = select_checkpoint(candidates)
    csv_path = run_root / "summary" / f"threshold_sweep_{checkpoint_label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "checkpoint_label": checkpoint_label,
        "checkpoint": str(checkpoint),
        "sweep_root": str(sweep_root),
        "sweep_csv": str(csv_path),
        "rows": rows,
        "selected_row": selected[7],
        "selected_validation": selected[8],
    }


def copy_final_selected(run_root: Path, checkpoint_eval: Dict) -> Dict:
    label = checkpoint_eval["checkpoint_label"]
    selected = checkpoint_eval["selected_row"]
    final_root = run_root / f"final_{label}"
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)
    selected_out = Path(selected["output_dir"])
    if (selected_out / "infer").exists():
        shutil.copytree(selected_out / "infer", final_root / "infer")
    if (selected_out / "eval").exists():
        shutil.copytree(selected_out / "eval", final_root / "eval")
    return {
        "final_root": str(final_root),
        "metrics_summary": str(final_root / "eval/metrics_summary.json"),
        "metrics_per_tooth": str(final_root / "eval/metrics_per_tooth.csv"),
        "failed_or_bad_teeth": str(final_root / "eval/failed_or_bad_teeth.csv"),
    }


def load_stage2_baseline(root: Path) -> Dict:
    report_path = root / "summary/validation_report.json"
    sweep_path = root / "summary/threshold_sweep.csv"
    if not report_path.exists():
        return {
            "run_root": str(root),
            "available": False,
            "report_path": str(report_path),
        }
    report = read_json(report_path)
    metrics = report.get("metrics_summary") or {}
    return {
        "run_root": str(root),
        "available": True,
        "report_path": str(report_path),
        "sweep_csv": str(sweep_path) if sweep_path.exists() else None,
        "validation_status": report.get("validation_status"),
        "selected_threshold": report.get("selected_threshold"),
        "stage2_validation_best_epoch": report.get("stage2_validation_best_epoch"),
        "mean_dist_mm": float(report.get("mean_dist_mm", metrics.get("mean_dist_mm", metrics.get("mean", float("inf"))))),
        "p95_dist_mm": float(report.get("p95_dist_mm", metrics.get("p95_dist_mm", metrics.get("p95", float("inf"))))),
        "sr@1.0mm": float(report.get("sr@1.0mm", goal19.summary_sr1(metrics))),
        "missing_prediction_count": int(report.get("missing_prediction_count", 0) or 0),
        "no_curve_count": int(report.get("no_curve_count", 0) or 0),
        "failed_tooth_count": int(report.get("failed_tooth_count", 0) or 0),
        "failed_or_bad_teeth_count": int(report.get("failed_or_bad_teeth_count", 0) or 0),
    }


def choose_primary_stage2(baselines: List[Dict]) -> Dict:
    available = [b for b in baselines if b.get("available")]
    if not available:
        return {"available": False}
    available.sort(
        key=lambda b: (
            STATUS_RANK.get(str(b.get("validation_status")), 9),
            b.get("p95_dist_mm", float("inf")),
            -b.get("sr@1.0mm", 0.0),
        )
    )
    return available[0]


def compare_stage1_to_stage2(stage1_eval: Dict, primary_stage2: Dict) -> Dict:
    selected = stage1_eval["selected_validation"]
    if not primary_stage2.get("available"):
        return {
            "primary_stage2_available": False,
            "stage2_usefulness_conclusion": "No completed Stage2 16/3 baseline report was available; Stage2 usefulness is inconclusive.",
        }
    delta_p95 = selected["p95_dist_mm"] - primary_stage2["p95_dist_mm"]
    delta_sr1 = selected["sr@1.0mm"] - primary_stage2["sr@1.0mm"]
    delta_mean = selected["mean_dist_mm"] - primary_stage2["mean_dist_mm"]
    close = delta_p95 <= 0.15 and delta_sr1 >= -0.03 and delta_mean <= 0.05
    clearly_worse = delta_p95 > 0.25 or delta_sr1 < -0.05
    if close:
        conclusion = (
            "Stage1-only is close to or better than the Stage2 16/3 baseline; this split does not show an independent Stage2 benefit."
        )
    elif clearly_worse:
        conclusion = (
            "Stage1-only is clearly worse than the Stage2 16/3 baseline; Stage2 has measurable value on this split, although the Stage2 holdout run may still fail validation."
        )
    else:
        conclusion = (
            "Stage1-only differs from Stage2 but not decisively under the configured closeness rule; Stage2 usefulness remains borderline on this split."
        )
    return {
        "primary_stage2_available": True,
        "primary_stage2_run_root": primary_stage2["run_root"],
        "delta_mean_stage1_minus_stage2": delta_mean,
        "delta_p95_stage1_minus_stage2": delta_p95,
        "delta_sr1_stage1_minus_stage2": delta_sr1,
        "closeness_rule": "close if delta_p95 <= 0.15, delta_sr1 >= -0.03, and delta_mean <= 0.05",
        "stage1_close_to_stage2": close,
        "stage1_clearly_worse_than_stage2": clearly_worse,
        "stage2_usefulness_conclusion": conclusion,
    }


def audit_artifacts(run_root: Path, source_manifest: Dict, stage1_cfg_path: Path) -> Dict:
    return {
        "stage1_config_exists": stage1_cfg_path.exists(),
        "input_manifest_exists": (run_root / "input/source_manifest.json").exists(),
        "case_manifest_exists": (run_root / "input/case_manifest.csv").exists(),
        "train_case_count_is_16": source_manifest.get("train_case_count") == 16,
        "holdout_case_count_is_3": source_manifest.get("holdout_case_count") == 3,
        "train_tooth_count_is_497": source_manifest.get("train_tooth_dir_count") == 497,
        "holdout_tooth_count_is_88": source_manifest.get("holdout_tooth_dir_count") == 88,
        "stage1_best_hash_match": source_manifest.get("stage1_best_source_sha256")
        == source_manifest.get("stage1_best_copy_sha256"),
        "stage1_last_hash_match": source_manifest.get("stage1_last_source_sha256")
        == source_manifest.get("stage1_last_copy_sha256"),
    }


def write_direct_report(
    run_root: Path,
    source_manifest: Dict,
    stage1_cfg_path: Path,
    best_eval: Dict,
    last_eval: Dict,
    stage2_baselines: List[Dict],
) -> Dict:
    best_final = copy_final_selected(run_root, best_eval)
    last_final = copy_final_selected(run_root, last_eval)
    primary_stage2 = choose_primary_stage2(stage2_baselines)
    comparison = compare_stage1_to_stage2(best_eval, primary_stage2)
    report = {
        "created_at": now_iso(),
        "run_root": str(run_root),
        "objective": "Evaluate the existing Stage1 exp03 16-train checkpoint directly on the same 3 holdout cases, without Stage2 training.",
        "source_manifest": str(run_root / "input/source_manifest.json"),
        "case_manifest": str(run_root / "input/case_manifest.csv"),
        "stage1_train_config_source": str(stage1_cfg_path),
        "thresholds": THRESHOLDS,
        "stage1_best": {
            **best_eval,
            "final": best_final,
        },
        "stage1_last": {
            **last_eval,
            "final": last_final,
        },
        "best_vs_last": {
            "best_selected_threshold": best_eval["selected_row"]["threshold"],
            "last_selected_threshold": last_eval["selected_row"]["threshold"],
            "best_status": best_eval["selected_validation"]["status"],
            "last_status": last_eval["selected_validation"]["status"],
            "best_p95_dist_mm": best_eval["selected_validation"]["p95_dist_mm"],
            "last_p95_dist_mm": last_eval["selected_validation"]["p95_dist_mm"],
            "best_sr@1.0mm": best_eval["selected_validation"]["sr@1.0mm"],
            "last_sr@1.0mm": last_eval["selected_validation"]["sr@1.0mm"],
        },
        "stage2_16_3_baselines": stage2_baselines,
        "primary_stage2_16_3_baseline": primary_stage2,
        "stage1_only_vs_stage2_16_3": comparison,
        "precheck_and_artifact_audit": audit_artifacts(run_root, source_manifest, stage1_cfg_path),
        "stage2_usefulness_conclusion": comparison["stage2_usefulness_conclusion"],
    }
    write_json(run_root / "summary/direct_eval_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage1-only direct eval on the 16 train / 3 holdout split.")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--stage1-config", default=str(DEFAULT_STAGE1_CONFIG))
    parser.add_argument("--stage1-best", default=str(split16.DEFAULT_STAGE1_BEST))
    parser.add_argument("--stage1-last", default=str(DEFAULT_STAGE1_LAST))
    parser.add_argument("--stage2-split-root", default=str(DEFAULT_STAGE2_SPLIT_ROOT))
    parser.add_argument("--stage2-baseline-root", action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    run_root = (
        Path(args.run_root).resolve()
        if args.run_root
        else Path(f"/root/cej_isolated_runs/C/stage1_16train_direct_eval_{timestamp()}").resolve()
    )
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"run root exists; use --resume to continue: {run_root}")
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        (run_root / "config").mkdir()
        (run_root / "logs").mkdir()
        (run_root / "summary").mkdir()

    source_manifest = setup_inputs(
        run_root,
        Path(args.stage2_split_root).resolve(),
        Path(args.stage1_best).resolve(),
        Path(args.stage1_last).resolve(),
        args.resume,
    )
    stage1_cfg_path = Path(args.stage1_config).resolve()
    if not stage1_cfg_path.exists():
        raise FileNotFoundError(f"Stage1 config missing: {stage1_cfg_path}")
    stage1_cfg = yaml.safe_load(stage1_cfg_path.read_text(encoding="utf-8"))
    base_cfg = render_eval_base_config(stage1_cfg, source_manifest, run_root, args.device)
    base_eval_cfg_path = run_root / "config/stage1_16train_holdout3_direct_eval_base.yaml"
    write_yaml(base_eval_cfg_path, base_cfg)
    write_json(run_root / "summary/setup_audit.json", audit_artifacts(run_root, source_manifest, stage1_cfg_path))

    if args.setup_only:
        print(f"[OK] setup complete: {run_root}")
        print(f"[CONFIG] {base_eval_cfg_path}")
        print(f"[AUDIT] {run_root / 'summary/setup_audit.json'}")
        return 0

    best_eval = summarize_checkpoint_eval(
        repo_dir,
        args.python_exe,
        run_root,
        base_cfg,
        source_manifest,
        "best",
        Path(source_manifest["stage1_best_copy"]),
        args.force_eval,
    )
    last_eval = summarize_checkpoint_eval(
        repo_dir,
        args.python_exe,
        run_root,
        base_cfg,
        source_manifest,
        "last",
        Path(source_manifest["stage1_last_copy"]),
        args.force_eval,
    )
    baseline_roots = [Path(p).resolve() for p in args.stage2_baseline_root] if args.stage2_baseline_root else DEFAULT_STAGE2_BASELINES
    stage2_baselines = [load_stage2_baseline(root) for root in baseline_roots]
    report = write_direct_report(run_root, source_manifest, stage1_cfg_path, best_eval, last_eval, stage2_baselines)
    selected = best_eval["selected_validation"]
    print(f"[STAGE1_BEST_STATUS] {selected['status']}")
    print(f"[STAGE1_BEST_THRESHOLD] {best_eval['selected_row']['threshold']}")
    print(f"[STAGE1_BEST_MEAN] {selected['mean_dist_mm']}")
    print(f"[STAGE1_BEST_P95] {selected['p95_dist_mm']}")
    print(f"[STAGE1_BEST_SR1] {selected['sr@1.0mm']}")
    print(f"[CONCLUSION] {report['stage2_usefulness_conclusion']}")
    print(f"[REPORT] {run_root / 'summary/direct_eval_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
