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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_goal_mode_19full as goal19


DEFAULT_UNSUP_CKPT = Path("/root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt")
DEFAULT_BASE_CONFIG = Path("configs/server_sup_manual6_unseen_train.yaml")
DEFAULT_STAGE2_CURRENT = Path("/root/cej_isolated_runs/C/unsup_to_goal_retrain_20260601_012139_stage2_goal19")
DEFAULT_STAGE2_HISTORY = Path("/root/cej_isolated_runs/C/goal模式_19full_001")
THRESHOLDS = [0.20, 0.25, 0.30, 0.35]
STAGE1_TRAIN_HARD_CASES = ["041-31/41/42"]
STAGE1_HOLDOUT_HARD_CASES = ["052-31/32/41/42/33", "040-42/41/32", "044-31/32/36/46"]


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


def setup_inputs(run_root: Path, unsup_ckpt: Path) -> Dict:
    input_dir = run_root / "input"
    processed_all = input_dir / "processed_all19"
    ckpt_dir = input_dir / "checkpoints"
    if input_dir.exists() or input_dir.is_symlink():
        raise FileExistsError(f"input directory exists, refusing to overwrite: {input_dir}")
    if not unsup_ckpt.exists():
        raise FileNotFoundError(f"unsup checkpoint not found: {unsup_ckpt}")

    source_rows = goal19.validate_sources()
    input_dir.mkdir(parents=True, exist_ok=False)
    copied_cases = goal19.copy_case_dirs(source_rows, processed_all)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unsup_copy = ckpt_dir / "unsup_last.pt"
    shutil.copy2(unsup_ckpt, unsup_copy)
    source_hash = sha256_file(unsup_ckpt)
    copy_hash = sha256_file(unsup_copy)
    if source_hash != copy_hash:
        raise RuntimeError(f"unsup checkpoint hash mismatch: {source_hash} != {copy_hash}")

    manifest = {
        "created_at": now_iso(),
        "run_root": str(run_root),
        "processed_all19": str(processed_all),
        "train_processed": str(processed_all),
        "holdout_processed": str(processed_all),
        "case_count": len(source_rows),
        "expected_case_count": 19,
        "case_ids": [row["case_id"] for row in source_rows],
        "source_rows": source_rows,
        "copied_cases": copied_cases,
        "tooth_dir_count": goal19.count_tooth_dirs(processed_all),
        "expected_tooth_count": 585,
        "unsup_checkpoint_source": str(unsup_ckpt),
        "unsup_checkpoint_copy": str(unsup_copy),
        "unsup_checkpoint_source_sha256": source_hash,
        "unsup_checkpoint_copy_sha256": copy_hash,
        "stage1_train_hard_cases": STAGE1_TRAIN_HARD_CASES,
        "stage1_holdout_hard_cases": STAGE1_HOLDOUT_HARD_CASES,
        "thresholds": THRESHOLDS,
    }
    if manifest["case_count"] != manifest["expected_case_count"]:
        raise RuntimeError(f"expected 19 cases, got {manifest['case_count']}")
    if manifest["tooth_dir_count"] != manifest["expected_tooth_count"]:
        raise RuntimeError(f"expected 585 tooth dirs, got {manifest['tooth_dir_count']}")
    write_json(input_dir / "source_manifest.json", manifest)
    goal19.write_case_manifest(input_dir / "case_manifest.csv", source_rows)
    return manifest


def load_or_setup_inputs(run_root: Path, unsup_ckpt: Path, resume: bool) -> Dict:
    manifest_path = run_root / "input" / "source_manifest.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"--resume requested but input manifest is missing: {manifest_path}")
        return read_json(manifest_path)
    return setup_inputs(run_root, unsup_ckpt)


def render_stage1_config(base_cfg: Dict, source_manifest: Dict, run_root: Path, args: argparse.Namespace) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("project", {})["device"] = args.device

    data = cfg.setdefault("data", {})
    data["processed_dir"] = source_manifest["processed_all19"]
    data["holdout_processed_dir"] = source_manifest["processed_all19"]
    data["output_dir"] = str(run_root / "outputs_stage1")
    data["processed_format"] = "nii.gz"
    data["use_tooth_mask_channel"] = True

    model = cfg.setdefault("model", {})
    model["in_channels"] = 2
    model["out_channels"] = 1
    model["base_channels"] = int(args.base_channels)
    model["depth"] = 4
    model.setdefault("num_res_units", 2)
    model.setdefault("norm", "batch")

    train = cfg.setdefault("train", {})
    train["epochs"] = int(args.max_epochs)
    train["max_epochs"] = int(args.max_epochs)
    train["min_epochs"] = int(args.min_epochs)
    train["early_stopping_patience"] = int(args.patience)
    train["early_stopping_min_delta"] = float(args.early_stopping_min_delta)
    train["batch_size"] = int(args.batch_size)
    train["holdout_batch_size"] = int(args.holdout_batch_size)
    train["num_workers"] = int(args.num_workers)
    train["holdout_num_workers"] = int(args.holdout_num_workers)
    train["lr"] = float(args.lr)
    train["cache_rate"] = float(args.cache_rate)
    train["holdout_cache_rate"] = float(args.holdout_cache_rate)
    train["loss"] = "dice_focal_skeleton"
    train["pretrained_ckpt"] = source_manifest["unsup_checkpoint_copy"]
    train["pretrained_strict"] = False
    train["weighted_sampler"] = True
    train["hard_cases"] = STAGE1_TRAIN_HARD_CASES
    train["hard_case_weight"] = float(args.hard_case_weight)
    train["checkpoint_top_k"] = int(args.checkpoint_top_k)
    train["checkpoint_every_n_epochs"] = int(args.checkpoint_every_n_epochs)
    train["loss_lambda_dice"] = 1.0
    train["loss_lambda_bce"] = 1.0
    train["loss_lambda_focal"] = 0.25
    train["focal_gamma"] = 2.0
    train["focal_alpha"] = 0.75
    train["loss_lambda_skeleton"] = 0.2
    train["skeleton_target_threshold"] = 0.95
    train["skeleton_pos_weight"] = 8.0
    train["surface_neighborhood_weight"] = 0.0
    train["surface_neighborhood_radius_vox"] = 2
    train["composite_score"] = {
        "start_epoch": int(args.score_start_epoch),
        "weights": {
            "loss": 0.30,
            "sym_p95": 0.30,
            "no_curve_count": 0.07,
            "bad_rate": 0.07,
            "vox03_empty": 0.03,
            "cc_count": 0.03,
            "wrap_miss": 0.05,
            "hard_case_viewer": 0.15,
        },
        "targets": {
            "loss": 0.5,
            "sym_p95_mm": 5.0,
            "bad_sym_p95_mm": 5.0,
            "hard_sym_p95_mm": 5.0,
            "hard_bad_sym_p95_mm": 5.0,
            "hard_case_count": 12,
            "vox03": 100.0,
            "cc_count": 4.0,
        },
    }

    infer = cfg.setdefault("infer", {})
    infer["ckpt_path"] = str(run_root / "outputs_stage1" / "train" / "checkpoints" / "best.pt")
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
    eval_cfg["prediction_name"] = "C_pred_fit.nii.gz"
    eval_cfg["eval_taus_mm"] = [0.5, 1.0, 1.5, 2.0]

    viz = cfg.setdefault("viz", {})
    viz["enable_2d"] = False
    viz["enable_3d"] = bool(args.enable_viz)
    viz["max_cases"] = 19
    viz["max_teeth_per_case"] = 64
    viz["show_dense_interp_curve"] = True
    viz["dense_interp_curve_closed"] = True
    viz["show_pseudo_gt_skeleton"] = True
    viz["pseudo_gt_skeleton_from_interp"] = False
    viz["pseudo_gt_skeleton_from_heatmap_peak"] = True
    viz["use_error_colormap_for_gt_points"] = True
    return cfg


def create_eval_config(base_cfg: Dict, checkpoint: Path, threshold: float, out_dir: Path) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["data"]["output_dir"] = str(out_dir)
    cfg.setdefault("infer", {})["ckpt_path"] = str(checkpoint)
    cfg["infer"]["threshold_theta"] = float(threshold)
    cfg.setdefault("holdout_eval", {})["threshold_theta"] = float(threshold)
    return cfg


def write_failed_rows(path: Path, rows: List[Dict]) -> None:
    goal19.write_failed_rows(path, rows)


def summarize_checkpoint_eval(
    repo_dir: Path,
    python_exe: str,
    run_root: Path,
    stage1_cfg: Dict,
    source_manifest: Dict,
    checkpoint_label: str,
    checkpoint: Path,
    thresholds: List[float],
    force: bool,
) -> Dict:
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint_label} checkpoint missing: {checkpoint}")
    sweep_root = run_root / f"threshold_sweep_{checkpoint_label}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates: List[Tuple] = []
    for threshold in thresholds:
        label = f"theta_{threshold:.2f}"
        out_dir = sweep_root / label
        cfg_path = run_root / "config" / f"{checkpoint_label}_{label}.yaml"
        cfg = create_eval_config(stage1_cfg, checkpoint, threshold, out_dir)
        write_yaml(cfg_path, cfg)
        metrics_path = out_dir / "eval" / "metrics_summary.json"
        if force and out_dir.exists():
            shutil.rmtree(out_dir)
        if not metrics_path.exists():
            run_cmd([python_exe, "-m", "src.infer", "--config", str(cfg_path)], repo_dir, run_root / "logs" / f"infer_{checkpoint_label}_{label}.log")
            run_cmd([python_exe, "-m", "src.eval", "--config", str(cfg_path)], repo_dir, run_root / "logs" / f"eval_{checkpoint_label}_{label}.log")
        summary = goal19.load_summary(out_dir / "eval")
        bad_rows = goal19.load_bad_rows(out_dir / "eval" / "metrics_per_tooth.csv")
        failed_csv = out_dir / "eval" / "failed_or_bad_teeth.csv"
        write_failed_rows(failed_csv, bad_rows)
        strict = goal19.strict_acceptance_checks(source_manifest, summary, len(bad_rows))
        row = {
            "checkpoint_label": checkpoint_label,
            "checkpoint": str(checkpoint),
            "threshold": threshold,
            "output_dir": str(out_dir),
            "config": str(cfg_path),
            "metrics_summary": str(out_dir / "eval" / "metrics_summary.json"),
            "metrics_per_tooth": str(out_dir / "eval" / "metrics_per_tooth.csv"),
            "failed_or_bad_teeth": str(failed_csv),
            "mean_dist_mm": strict["mean_dist_mm"],
            "p95_dist_mm": strict["p95_dist_mm"],
            "sr@1.0mm": strict["sr@1.0mm"],
            "missing_prediction_count": strict["missing_prediction_count"],
            "no_curve_count": strict["no_curve_count"],
            "failed_tooth_count": strict["failed_tooth_count"],
            "failed_or_bad_teeth_count": strict["failed_or_bad_teeth_count"],
            "strict_accepted": strict["accepted"],
        }
        rows.append(row)
        candidates.append(
            (
                0 if strict["accepted"] else 1,
                0 if strict["missing_prediction_count"] == 0 and strict["no_curve_count"] == 0 else 1,
                0 if strict["failed_or_bad_teeth_count"] == 0 else 1,
                strict["p95_dist_mm"],
                -strict["sr@1.0mm"],
                threshold,
                row,
                strict,
            )
        )
    candidates.sort(key=lambda x: x[:6])
    selected_row = candidates[0][6]
    selected_strict = candidates[0][7]
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
        "selected_row": selected_row,
        "selected_acceptance": selected_strict,
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
        "final_eval": str(final_root / "eval"),
        "metrics_summary": str(final_root / "eval" / "metrics_summary.json"),
        "metrics_per_tooth": str(final_root / "eval" / "metrics_per_tooth.csv"),
        "failed_or_bad_teeth": str(final_root / "eval" / "failed_or_bad_teeth.csv"),
    }


def load_stage2_baseline(current_root: Path, history_root: Path) -> Dict:
    for kind, root in (("current_same_lineage", current_root), ("historical_accepted", history_root)):
        report_path = root / "summary" / "acceptance_report.json"
        if not report_path.exists():
            continue
        report = read_json(report_path)
        accepted = bool(report.get("accepted", False))
        if kind == "current_same_lineage" or accepted:
            summary = report.get("metrics_summary") or {}
            return {
                "kind": kind,
                "run_root": str(root),
                "report_path": str(report_path),
                "accepted": accepted,
                "selected_threshold": report.get("selected_threshold"),
                "mean_dist_mm": float(summary.get("mean_dist_mm", summary.get("mean", float("inf")))),
                "p95_dist_mm": float(summary.get("p95_dist_mm", summary.get("p95", float("inf")))),
                "sr@1.0mm": goal19.summary_sr1(summary),
                "missing_prediction_count": int(summary.get("missing_prediction_count", report.get("missing_prediction_count", 0)) or 0),
                "no_curve_count": int(summary.get("no_curve_count", report.get("no_curve_count", 0)) or 0),
                "failed_tooth_count": int(summary.get("failed_tooth_count", report.get("failed_tooth_count", 0)) or 0),
                "failed_or_bad_teeth_count": int(report.get("failed_or_bad_teeth_count", 0) or 0),
            }
    return {
        "kind": "missing",
        "run_root": None,
        "report_path": None,
        "accepted": False,
        "mean_dist_mm": None,
        "p95_dist_mm": None,
        "sr@1.0mm": None,
    }


def stage2_conclusion(best_acceptance: Dict, baseline: Dict) -> str:
    if best_acceptance.get("accepted"):
        return "Stage1-only strict accepted; Stage2 independent gain is not demonstrated by this direct-eval ablation."
    if not baseline.get("accepted"):
        return "Stage1-only did not strict accept, but no accepted Stage2 baseline was available; Stage2 usefulness is inconclusive."
    p95 = float(best_acceptance.get("p95_dist_mm", float("inf")))
    sr1 = float(best_acceptance.get("sr@1.0mm", 0.0))
    if p95 > 0.65 or sr1 < 0.99:
        return "Stage1-only is clearly worse than accepted Stage2 full19 under strict criteria; Stage2 training has independent value."
    return "Stage1-only missed strict acceptance only narrowly; Stage2 value appears limited or threshold/postprocess-dependent."


def audit_inputs_and_training(run_root: Path, source_manifest: Dict) -> Dict:
    train_manifest_path = run_root / "outputs_stage1" / "train" / "train_manifest.json"
    train_manifest = read_json(train_manifest_path) if train_manifest_path.exists() else {}
    return {
        "input_case_count_ok": source_manifest.get("case_count") == 19,
        "input_tooth_dir_count_ok": source_manifest.get("tooth_dir_count") == 585,
        "train_equals_holdout_all19": source_manifest.get("train_processed") == source_manifest.get("holdout_processed") == source_manifest.get("processed_all19"),
        "unsup_hash_match": source_manifest.get("unsup_checkpoint_source_sha256") == source_manifest.get("unsup_checkpoint_copy_sha256"),
        "stage1_best_exists": (run_root / "outputs_stage1" / "train" / "checkpoints" / "best.pt").exists(),
        "stage1_last_exists": (run_root / "outputs_stage1" / "train" / "checkpoints" / "last.pt").exists(),
        "stage1_metrics_exists": (run_root / "outputs_stage1" / "train" / "metrics.csv").exists(),
        "stage1_train_manifest_exists": train_manifest_path.exists(),
        "train_manifest_dataset": train_manifest.get("dataset"),
        "train_manifest_pretrained": train_manifest.get("pretrained"),
        "train_manifest_best_epoch": train_manifest.get("best_epoch"),
        "train_manifest_best_composite_score": train_manifest.get("best_composite_score"),
    }


def write_direct_report(
    run_root: Path,
    source_manifest: Dict,
    stage1_cfg_path: Path,
    best_eval: Dict,
    last_eval: Dict,
    baseline: Dict,
) -> Dict:
    best_final = copy_final_selected(run_root, best_eval)
    last_final = copy_final_selected(run_root, last_eval)
    report = {
        "created_at": now_iso(),
        "run_root": str(run_root),
        "objective": "Stage1 exp03_skeleton_aux_loss trained on all 19 PASS cases from unsup_last.pt, directly evaluated with Stage2 goal19 sweep/acceptance without Stage2 training.",
        "source_manifest": str(run_root / "input" / "source_manifest.json"),
        "case_manifest": str(run_root / "input" / "case_manifest.csv"),
        "stage1_train_config": str(stage1_cfg_path),
        "thresholds": THRESHOLDS,
        "stage1_best": {
            **best_eval,
            "final": best_final,
        },
        "stage1_last": {
            **last_eval,
            "final": last_final,
        },
        "stage2_baseline": baseline,
        "best_vs_last": {
            "best_selected_threshold": best_eval["selected_row"]["threshold"],
            "last_selected_threshold": last_eval["selected_row"]["threshold"],
            "best_p95_dist_mm": best_eval["selected_acceptance"]["p95_dist_mm"],
            "last_p95_dist_mm": last_eval["selected_acceptance"]["p95_dist_mm"],
            "best_sr@1.0mm": best_eval["selected_acceptance"]["sr@1.0mm"],
            "last_sr@1.0mm": last_eval["selected_acceptance"]["sr@1.0mm"],
            "best_strict_accepted": best_eval["selected_acceptance"]["accepted"],
            "last_strict_accepted": last_eval["selected_acceptance"]["accepted"],
        },
        "stage1_only_vs_stage2": {
            "stage1_best_acceptance": best_eval["selected_acceptance"],
            "stage2_baseline": baseline,
            "delta_p95_stage1_minus_stage2": (
                best_eval["selected_acceptance"]["p95_dist_mm"] - baseline["p95_dist_mm"]
                if baseline.get("p95_dist_mm") is not None
                else None
            ),
            "delta_sr1_stage1_minus_stage2": (
                best_eval["selected_acceptance"]["sr@1.0mm"] - baseline["sr@1.0mm"]
                if baseline.get("sr@1.0mm") is not None
                else None
            ),
        },
        "precheck_and_artifact_audit": audit_inputs_and_training(run_root, source_manifest),
        "stage2_usefulness_conclusion": stage2_conclusion(best_eval["selected_acceptance"], baseline),
    }
    write_json(run_root / "summary" / "direct_eval_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage1 exp03 on all 19 PASS cases and directly evaluate best/last with Stage2 goal19 criteria.")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--unsup-ckpt", default=str(DEFAULT_UNSUP_CKPT))
    parser.add_argument("--stage2-current-root", default=str(DEFAULT_STAGE2_CURRENT))
    parser.add_argument("--stage2-history-root", default=str(DEFAULT_STAGE2_HISTORY))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=5e-4)
    parser.add_argument("--score-start-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--holdout-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--holdout-num-workers", type=int, default=2)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--holdout-cache-rate", type=float, default=0.0)
    parser.add_argument("--hard-case-weight", type=float, default=4.0)
    parser.add_argument("--checkpoint-top-k", type=int, default=5)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    parser.add_argument("--enable-viz", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    run_root = Path(args.run_root).resolve() if args.run_root else Path(f"/root/cej_isolated_runs/C/stage1_full19_direct_eval_{timestamp()}").resolve()
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"run root exists; use --resume to continue: {run_root}")
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        (run_root / "config").mkdir()
        (run_root / "logs").mkdir()
        (run_root / "summary").mkdir()

    source_manifest = load_or_setup_inputs(run_root, Path(args.unsup_ckpt).resolve(), args.resume)
    base_config = Path(args.base_config)
    if not base_config.is_absolute():
        base_config = repo_dir / base_config
    base_cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    stage1_cfg = render_stage1_config(base_cfg, source_manifest, run_root, args)
    stage1_cfg_path = run_root / "config" / "stage1_full19_train.yaml"
    write_yaml(stage1_cfg_path, stage1_cfg)

    if args.setup_only:
        audit = audit_inputs_and_training(run_root, source_manifest)
        write_json(run_root / "summary" / "setup_audit.json", audit)
        print(f"[OK] setup complete: {run_root}")
        print(f"[CONFIG] {stage1_cfg_path}")
        print(f"[AUDIT] {run_root / 'summary' / 'setup_audit.json'}")
        return 0

    best_ckpt = run_root / "outputs_stage1" / "train" / "checkpoints" / "best.pt"
    last_ckpt = run_root / "outputs_stage1" / "train" / "checkpoints" / "last.pt"
    if not args.skip_train and not best_ckpt.exists():
        run_cmd(
            [args.python_exe, "-m", "src.train_loss_earlystop", "--config", str(stage1_cfg_path)],
            repo_dir,
            run_root / "logs" / "train_stage1_full19.log",
        )

    if args.train_only:
        print(f"[OK] training complete: {best_ckpt}")
        return 0

    best_eval = summarize_checkpoint_eval(
        repo_dir,
        args.python_exe,
        run_root,
        stage1_cfg,
        source_manifest,
        "best",
        best_ckpt,
        THRESHOLDS,
        args.force_eval,
    )
    last_eval = summarize_checkpoint_eval(
        repo_dir,
        args.python_exe,
        run_root,
        stage1_cfg,
        source_manifest,
        "last",
        last_ckpt,
        THRESHOLDS,
        args.force_eval,
    )
    baseline = load_stage2_baseline(Path(args.stage2_current_root), Path(args.stage2_history_root))
    report = write_direct_report(run_root, source_manifest, stage1_cfg_path, best_eval, last_eval, baseline)
    print(f"[STAGE1_BEST_ACCEPTED] {best_eval['selected_acceptance']['accepted']}")
    print(f"[STAGE1_BEST_P95] {best_eval['selected_acceptance']['p95_dist_mm']}")
    print(f"[STAGE1_BEST_SR1] {best_eval['selected_acceptance']['sr@1.0mm']}")
    print(f"[STAGE2_BASELINE] {baseline.get('kind')} {baseline.get('run_root')}")
    print(f"[CONCLUSION] {report['stage2_usefulness_conclusion']}")
    print(f"[REPORT] {run_root / 'summary' / 'direct_eval_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
