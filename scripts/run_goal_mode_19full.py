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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


DEFAULT_RUN_ROOT = Path("/root/cej_isolated_runs/C/goal模式_19full_001")
DEFAULT_BASE_CONFIG = Path("configs/server_sup_manual_inc_041055.yaml")
DEFAULT_BEST_CKPT = Path(
    "/root/cej_isolated_runs/C/cej_loss_earlystop_20260507_004/"
    "experiments/exp03_skeleton_aux_loss/outputs/train/checkpoints/best.pt"
)
DEFAULT_UNSUP_CKPT = Path("/root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt")

PASS_SOURCES = [
    {
        "name": "manual6_from_run_unsup_001",
        "processed_root": Path("/root/cej_runs/run_unsup_001/processed"),
        "usage_report": Path("/root/cej_runs/run_unsup_001/manual_points_usage_report.json"),
        "cases": [
            "ToothFairy3F_008",
            "ToothFairy3F_009",
            "ToothFairy3F_010",
            "ToothFairy3F_018",
            "ToothFairy3F_021",
            "ToothFairy3F_023",
        ],
    },
    {
        "name": "manual_new_025040_001",
        "processed_root": Path("/root/cej_runs/run_manual_new_025040_001/processed"),
        "usage_report": Path("/root/cej_runs/run_manual_new_025040_001/manual_points_usage_report.json"),
        "cases": [
            "ToothFairy3F_025",
            "ToothFairy3F_026",
            "ToothFairy3F_027",
            "ToothFairy3F_033",
            "ToothFairy3F_040",
        ],
    },
    {
        "name": "manual_inc_041055_001",
        "processed_root": Path("/root/cej_runs/run_manual_inc_041055_001/processed"),
        "usage_report": Path("/root/cej_runs/run_manual_inc_041055_001/manual_points_usage_report.json"),
        "cases": [
            "ToothFairy3F_041",
            "ToothFairy3F_044",
            "ToothFairy3F_050",
            "ToothFairy3F_051",
            "ToothFairy3F_052",
            "ToothFairy3F_053",
            "ToothFairy3F_054",
            "ToothFairy3F_055",
        ],
    },
]

HARD_CASES = ["052-31/32/33/41/42", "040-32/41/42", "044-31/32/36/46"]
THRESHOLDS = [0.20, 0.25, 0.30, 0.35]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fail_if_exists(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} exists, refusing to overwrite: {path}")


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


def case_suffix(case_id: str) -> str:
    text = str(case_id)
    digits = ""
    for ch in reversed(text):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return digits or text


def load_pass_cases(report_path: Path) -> List[str]:
    data = read_json(report_path)
    return [str(c.get("case_id")) for c in data.get("cases", []) if c.get("status") == "PASS_FULLY_USED"]


def validate_sources() -> List[Dict]:
    seen = set()
    rows = []
    for src in PASS_SOURCES:
        processed_root = src["processed_root"]
        usage_report = src["usage_report"]
        if not processed_root.exists():
            raise FileNotFoundError(f"processed source missing: {processed_root}")
        if not usage_report.exists():
            raise FileNotFoundError(f"usage report missing: {usage_report}")
        pass_cases = set(load_pass_cases(usage_report))
        missing_from_report = [case for case in src["cases"] if case not in pass_cases]
        if missing_from_report:
            raise RuntimeError(f"source {src['name']} has non-PASS cases: {missing_from_report}")
        for case in src["cases"]:
            if case in seen:
                raise RuntimeError(f"duplicate source case: {case}")
            seen.add(case)
            case_dir = processed_root / case
            if not case_dir.exists():
                raise FileNotFoundError(f"case directory missing: {case_dir}")
            rows.append(
                {
                    "source": src["name"],
                    "case_id": case,
                    "source_case_dir": str(case_dir),
                    "usage_report": str(usage_report),
                }
            )
    if len(rows) != 19:
        raise RuntimeError(f"expected 19 PASS cases, found {len(rows)}")
    return rows


def count_tooth_dirs(root: Path) -> int:
    return len(glob.glob(str(root / "*" / "tooth_*")))


def copy_case_dirs(rows: Iterable[Dict], dst_root: Path) -> Dict[str, str]:
    dst_root.mkdir(parents=True, exist_ok=False)
    copied = {}
    for row in rows:
        case_id = row["case_id"]
        src = Path(row["source_case_dir"])
        dst = dst_root / case_id
        fail_if_exists(dst, "case destination")
        shutil.copytree(src, dst, symlinks=False)
        copied[case_id] = str(src)
    return copied


def choose_pretrained(best_ckpt: Path, unsup_ckpt: Path) -> Tuple[Path, str]:
    if best_ckpt.exists():
        return best_ckpt, "exp03_skeleton_aux_loss_best"
    if unsup_ckpt.exists():
        return unsup_ckpt, "unsup_pretrain_fallback"
    raise FileNotFoundError(f"neither pretrained checkpoint exists: {best_ckpt} / {unsup_ckpt}")


def setup_inputs(run_root: Path, best_ckpt: Path, unsup_ckpt: Path) -> Dict:
    input_dir = run_root / "input"
    processed_all = input_dir / "processed_all19"
    ckpt_dir = input_dir / "checkpoints"
    manifest_path = input_dir / "source_manifest.json"
    fail_if_exists(input_dir, "input directory")
    source_rows = validate_sources()
    pretrained, pretrained_kind = choose_pretrained(best_ckpt, unsup_ckpt)

    input_dir.mkdir(parents=True, exist_ok=False)
    copied_cases = copy_case_dirs(source_rows, processed_all)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pretrained_copy = ckpt_dir / pretrained.name
    shutil.copy2(pretrained, pretrained_copy)
    unsup_copy = None
    if unsup_ckpt.exists() and unsup_ckpt.resolve() != pretrained.resolve():
        unsup_copy = ckpt_dir / "unsup_last.pt"
        shutil.copy2(unsup_ckpt, unsup_copy)

    manifest = {
        "created_at": now_iso(),
        "run_root": str(run_root),
        "processed_all19": str(processed_all),
        "case_count": len(source_rows),
        "case_ids": [row["case_id"] for row in source_rows],
        "source_rows": source_rows,
        "copied_cases": copied_cases,
        "tooth_dir_count": count_tooth_dirs(processed_all),
        "pretrained_source": str(pretrained),
        "pretrained_copy": str(pretrained_copy),
        "pretrained_kind": pretrained_kind,
        "unsup_checkpoint_source": str(unsup_ckpt),
        "unsup_checkpoint_copy": str(unsup_copy) if unsup_copy else str(pretrained_copy),
        "hard_cases": HARD_CASES,
    }
    write_json(manifest_path, manifest)
    write_case_manifest(input_dir / "case_manifest.csv", source_rows)
    return manifest


def load_or_setup_inputs(run_root: Path, best_ckpt: Path, unsup_ckpt: Path, resume: bool) -> Dict:
    manifest_path = run_root / "input" / "source_manifest.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"--resume requested but input manifest is missing: {manifest_path}")
        return read_json(manifest_path)
    return setup_inputs(run_root, best_ckpt, unsup_ckpt)


def write_case_manifest(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "case_id", "source_case_dir", "usage_report"])
        writer.writeheader()
        writer.writerows(rows)


def update_train_cfg(cfg: Dict, args: argparse.Namespace, source_manifest: Dict, run_root: Path) -> Dict:
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("project", {})
    cfg["project"]["device"] = args.device

    data = cfg.setdefault("data", {})
    data["processed_dir"] = source_manifest["processed_all19"]
    data["holdout_processed_dir"] = source_manifest["processed_all19"]
    data["output_dir"] = str(run_root / "outputs_final")
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
    train["loss"] = "dice_focal_skeleton"
    train["pretrained_ckpt"] = source_manifest["pretrained_copy"]
    train["pretrained_strict"] = False
    train["max_epochs"] = int(args.max_epochs)
    train["epochs"] = int(args.max_epochs)
    train["min_epochs"] = int(args.min_epochs)
    train["lr"] = float(args.lr)
    train["early_stopping_patience"] = int(args.patience)
    train["early_stopping_min_delta"] = float(args.early_stopping_min_delta)
    train["batch_size"] = int(args.batch_size)
    train["num_workers"] = int(args.num_workers)
    train["holdout_batch_size"] = int(args.holdout_batch_size)
    train["holdout_num_workers"] = int(args.holdout_num_workers)
    train["cache_rate"] = float(args.cache_rate)
    train["holdout_cache_rate"] = float(args.holdout_cache_rate)
    train["weighted_sampler"] = True
    train["hard_cases"] = HARD_CASES
    train["hard_case_weight"] = float(args.hard_case_weight)
    train["checkpoint_top_k"] = int(args.checkpoint_top_k)
    train["checkpoint_every_n_epochs"] = int(args.checkpoint_every_n_epochs)
    train["loss_lambda_dice"] = float(args.loss_lambda_dice)
    train["loss_lambda_bce"] = float(args.loss_lambda_bce)
    train["loss_lambda_focal"] = float(args.loss_lambda_focal)
    train["focal_gamma"] = float(args.focal_gamma)
    train["focal_alpha"] = float(args.focal_alpha)
    train["loss_lambda_skeleton"] = float(args.loss_lambda_skeleton)
    train["skeleton_target_threshold"] = float(args.skeleton_target_threshold)
    train["skeleton_pos_weight"] = float(args.skeleton_pos_weight)
    train["surface_neighborhood_weight"] = float(args.surface_neighborhood_weight)
    train["surface_neighborhood_radius_vox"] = int(args.surface_neighborhood_radius_vox)
    train["composite_score"] = {
        "start_epoch": int(args.score_start_epoch),
        "weights": {
            "loss": 0.08,
            "sym_p95": 0.16,
            "manual_point_p95": 0.58,
            "manual_point_sr1_miss": 0.16,
            "no_curve_count": 0.10,
            "bad_rate": 0.05,
            "vox03_empty": 0.02,
            "cc_count": 0.02,
            "wrap_miss": 0.02,
            "hard_case_viewer": 0.05,
        },
        "targets": {
            "loss": 0.35,
            "sym_p95_mm": 2.0,
            "manual_point_p95_mm": 2.0,
            "bad_sym_p95_mm": 2.0,
            "hard_sym_p95_mm": 2.0,
            "hard_bad_sym_p95_mm": 2.0,
            "hard_case_count": 12,
            "vox03": 100.0,
            "cc_count": 4.0,
        },
    }

    infer = cfg.setdefault("infer", {})
    infer["ckpt_path"] = str(Path(data["output_dir"]) / "train" / "checkpoints" / "best.pt")
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
    holdout["hard_cases"] = HARD_CASES
    holdout["threshold_theta"] = 0.30
    holdout["fit_pred_curve_step_mm"] = 0.2
    holdout["fit_pred_curve_closed"] = True
    holdout["fit_pred_curve_smooth"] = 0.0
    holdout["fit_pred_curve_min_points"] = 8
    holdout.setdefault("wrap_tau_mm", 1.0)
    holdout.setdefault("gt_peak_threshold", 0.95)
    holdout.setdefault("gt_rel_threshold", 0.95)

    eval_cfg = cfg.setdefault("eval", {})
    eval_cfg["prediction_name"] = "C_pred_fit.nii.gz"
    eval_cfg["eval_taus_mm"] = [0.5, 1.0, 1.5, 2.0]

    viz = cfg.setdefault("viz", {})
    viz["enable_2d"] = False
    viz["enable_3d"] = True
    viz["max_cases"] = 19
    viz["max_teeth_per_case"] = 64
    viz["show_dense_interp_curve"] = True
    viz["dense_interp_curve_closed"] = True
    viz["show_pseudo_gt_skeleton"] = True
    viz["pseudo_gt_skeleton_from_interp"] = False
    viz["pseudo_gt_skeleton_from_heatmap_peak"] = True
    viz["use_error_colormap_for_gt_points"] = True
    return cfg


def write_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def create_threshold_config(base_cfg: Dict, threshold: float, out_dir: Path, ckpt_path: Path) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["data"]["output_dir"] = str(out_dir)
    cfg.setdefault("infer", {})["ckpt_path"] = str(ckpt_path)
    cfg["infer"]["threshold_theta"] = float(threshold)
    cfg.setdefault("holdout_eval", {})["threshold_theta"] = float(threshold)
    return cfg


def summary_p95(summary: Dict) -> float:
    value = summary.get("p95_dist_mm", summary.get("p95"))
    if value is None:
        return float("inf")
    return float(value)


def summary_sr1(summary: Dict) -> float:
    sr = summary.get("sr", {})
    for key in ("1.0", "1", 1.0, 1):
        if key in sr:
            return float(sr[key])
    value = summary.get("sr@1.0mm")
    return float(value) if value is not None else 0.0


def load_summary(eval_dir: Path) -> Dict:
    path = eval_dir / "metrics_summary.json"
    return read_json(path) if path.exists() else {}


def load_bad_rows(per_tooth_csv: Path) -> List[Dict]:
    if not per_tooth_csv.exists():
        return []
    rows = []
    with per_tooth_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_points = int(row.get("n_points") or 0)
            if n_points <= 0:
                continue
            status = row.get("status", "")
            try:
                p95 = float(row.get("p95_dist_mm") or "nan")
            except ValueError:
                p95 = float("nan")
            if status != "ok" or not (p95 <= 2.0):
                rows.append(row)
    return rows


def write_failed_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "tooth_id",
        "n_points",
        "status",
        "mean_dist_mm",
        "p95_dist_mm",
        "sr@1.0",
        "sr@2.0",
        "pred_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sweep_thresholds(repo_dir: Path, python_exe: str, run_root: Path, cfg: Dict, thresholds: List[float]) -> Dict:
    ckpt_path = Path(cfg["data"]["output_dir"]) / "train" / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best checkpoint missing before sweep: {ckpt_path}")
    sweep_root = run_root / "threshold_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for threshold in thresholds:
        label = f"theta_{threshold:.2f}"
        out_dir = sweep_root / label
        cfg_path = run_root / "config" / f"{label}.yaml"
        threshold_cfg = create_threshold_config(cfg, threshold, out_dir, ckpt_path)
        write_yaml(cfg_path, threshold_cfg)
        if not (out_dir / "eval" / "metrics_summary.json").exists():
            run_cmd([python_exe, "-m", "src.infer", "--config", str(cfg_path)], repo_dir, run_root / "logs" / f"infer_{label}.log")
            run_cmd([python_exe, "-m", "src.eval", "--config", str(cfg_path)], repo_dir, run_root / "logs" / f"eval_{label}.log")
        summary = load_summary(out_dir / "eval")
        missing = int(summary.get("missing_prediction_count", 0) or 0)
        no_curve = int(summary.get("no_curve_count", 0) or 0)
        failed = int(summary.get("failed_tooth_count", 0) or 0)
        p95 = summary_p95(summary)
        mean = summary.get("mean_dist_mm", summary.get("mean"))
        sr1 = summary_sr1(summary)
        bad_rows = load_bad_rows(out_dir / "eval" / "metrics_per_tooth.csv")
        bad_count = len(bad_rows)
        row = {
            "threshold": threshold,
            "output_dir": str(out_dir),
            "metrics_summary": str(out_dir / "eval" / "metrics_summary.json"),
            "metrics_per_tooth": str(out_dir / "eval" / "metrics_per_tooth.csv"),
            "mean_dist_mm": mean,
            "p95_dist_mm": p95,
            "sr@1.0mm": sr1,
            "missing_prediction_count": missing,
            "no_curve_count": no_curve,
            "failed_tooth_count": failed,
            "failed_or_bad_teeth_count": bad_count,
        }
        rows.append(row)
        no_empty_curve = missing == 0 and no_curve == 0
        no_bad_teeth = bad_count == 0
        candidates.append((0 if no_empty_curve else 1, 0 if no_bad_teeth else 1, p95, -sr1, threshold, row, threshold_cfg))

    sweep_csv = run_root / "summary" / "threshold_sweep.csv"
    sweep_csv.parent.mkdir(parents=True, exist_ok=True)
    with sweep_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    candidates.sort(key=lambda item: item[:4])
    best = candidates[0]
    return {
        "sweep_csv": str(sweep_csv),
        "rows": rows,
        "best_row": best[5],
        "best_config": best[6],
    }


def copy_final_outputs(run_root: Path, best_row: Dict, best_cfg: Dict) -> Dict:
    final_root = run_root / "final"
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)
    best_out = Path(best_row["output_dir"])
    final_infer = final_root / "infer"
    final_eval = final_root / "eval"
    if (best_out / "infer").exists():
        shutil.copytree(best_out / "infer", final_infer)
    if (best_out / "eval").exists():
        shutil.copytree(best_out / "eval", final_eval)
    final_cfg = json.loads(json.dumps(best_cfg))
    final_cfg["data"]["output_dir"] = str(final_root)
    final_cfg["infer"]["ckpt_path"] = str(Path(run_root / "outputs_final" / "train" / "checkpoints" / "best.pt"))
    cfg_path = run_root / "config" / "final_selected.yaml"
    write_yaml(cfg_path, final_cfg)
    return {"final_root": str(final_root), "final_config": str(cfg_path)}


def run_final_viz(repo_dir: Path, python_exe: str, run_root: Path, final_config: Path) -> None:
    run_cmd([python_exe, "-m", "src.viz", "--config", str(final_config)], repo_dir, run_root / "logs" / "viz_final.log")


def acceptance_report(run_root: Path, source_manifest: Dict, best_row: Dict, final_info: Dict) -> Dict:
    final_eval = Path(final_info["final_root"]) / "eval"
    summary = load_summary(final_eval)
    per_tooth = final_eval / "metrics_per_tooth.csv"
    bad_rows = load_bad_rows(per_tooth)
    failed_csv = final_eval / "failed_or_bad_teeth.csv"
    write_failed_rows(failed_csv, bad_rows)
    criteria = {
        "case_count_is_19": int(source_manifest.get("case_count", 0)) == 19,
        "missing_prediction_count_is_0": int(summary.get("missing_prediction_count", 0) or 0) == 0,
        "no_curve_count_is_0": int(summary.get("no_curve_count", 0) or 0) == 0,
        "p95_dist_mm_lte_2": summary_p95(summary) <= 2.0,
        "mean_dist_mm_lte_1": float(summary.get("mean_dist_mm", summary.get("mean", float("inf")))) <= 1.0,
        "sr_at_1mm_gte_90pct": summary_sr1(summary) >= 0.90,
        "failed_or_bad_teeth_empty": len(bad_rows) == 0,
    }
    accepted = all(criteria.values())
    report = {
        "created_at": now_iso(),
        "accepted": bool(accepted),
        "criteria": criteria,
        "metrics_summary": summary,
        "metrics_summary_path": str(final_eval / "metrics_summary.json"),
        "metrics_per_tooth_path": str(per_tooth),
        "failed_or_bad_teeth_path": str(failed_csv),
        "failed_or_bad_teeth_count": len(bad_rows),
        "selected_threshold": best_row["threshold"],
        "selected_threshold_row": best_row,
        "case_manifest": str(run_root / "input" / "case_manifest.csv"),
        "source_manifest": str(run_root / "input" / "source_manifest.json"),
        "best_checkpoint": str(run_root / "outputs_final" / "train" / "checkpoints" / "best.pt"),
        "last_checkpoint": str(run_root / "outputs_final" / "train" / "checkpoints" / "last.pt"),
        "topk_dir": str(run_root / "outputs_final" / "train" / "checkpoints" / "topk"),
        "viewer_index": str(Path(final_info["final_root"]) / "viz" / "3d" / "index.html"),
        "final_config": final_info["final_config"],
    }
    write_json(run_root / "summary" / "acceptance_report.json", report)
    return report


def package_final(run_root: Path, report: Dict) -> Path:
    package_dir = run_root / "package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    package_path = package_dir / "goal模式_19full_training_bundle.tar.gz"
    include = [
        run_root / "config",
        run_root / "input" / "source_manifest.json",
        run_root / "input" / "case_manifest.csv",
        run_root / "outputs_final" / "train" / "checkpoints" / "best.pt",
        run_root / "outputs_final" / "train" / "checkpoints" / "last.pt",
        run_root / "outputs_final" / "train" / "checkpoints" / "topk",
        run_root / "outputs_final" / "train" / "metrics.csv",
        run_root / "outputs_final" / "train" / "train_manifest.json",
        run_root / "outputs_final" / "train" / "audit_train",
        run_root / "outputs_final" / "train" / "audit_holdout",
        run_root / "summary",
        Path(report["metrics_summary_path"]),
        Path(report["metrics_per_tooth_path"]),
        Path(report["failed_or_bad_teeth_path"]),
        Path(report["viewer_index"]),
    ]
    with tarfile.open(package_path, "w:gz") as tf:
        for path in include:
            if not path.exists():
                continue
            tf.add(path, arcname=str(path.relative_to(run_root)))
    return package_path


def audit_dataset(repo_dir: Path, python_exe: str, run_root: Path, cfg_path: Path) -> None:
    code = (
        "import json; "
        "from src.datasets.dataset import build_supervised_audit, write_supervised_audit; "
        "from src.utils.config import load_config; "
        f"cfg=load_config({str(cfg_path)!r}); "
        "audit=build_supervised_audit(cfg['data']['processed_dir'], cfg['data'].get('processed_format','nii.gz')); "
        f"paths=write_supervised_audit(audit, {str(run_root / 'input' / 'audit_all19')!r}); "
        "print(json.dumps({'audit': audit, 'paths': paths}, ensure_ascii=False))"
    )
    run_cmd([python_exe, "-c", code], repo_dir, run_root / "logs" / "audit_all19.log")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run goal-mode CEJ training on all 19 PASS manual cases.")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--best-ckpt", default=str(DEFAULT_BEST_CKPT))
    parser.add_argument("--unsup-ckpt", default=str(DEFAULT_UNSUP_CKPT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--score-start-epoch", type=int, default=5)
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
    parser.add_argument("--loss-lambda-dice", type=float, default=1.0)
    parser.add_argument("--loss-lambda-bce", type=float, default=1.0)
    parser.add_argument("--loss-lambda-focal", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--loss-lambda-skeleton", type=float, default=0.4)
    parser.add_argument("--skeleton-target-threshold", type=float, default=0.95)
    parser.add_argument("--skeleton-pos-weight", type=float, default=10.0)
    parser.add_argument("--surface-neighborhood-weight", type=float, default=0.05)
    parser.add_argument("--surface-neighborhood-radius-vox", type=int, default=2)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    run_root = Path(args.run_root).resolve()
    base_config = Path(args.base_config)
    if not base_config.is_absolute():
        base_config = repo_dir / base_config

    if run_root.exists() and not args.resume:
        raise FileExistsError(f"run root exists; use --resume to continue: {run_root}")
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        (run_root / "config").mkdir()
        (run_root / "logs").mkdir()
        (run_root / "summary").mkdir()

    source_manifest = load_or_setup_inputs(
        run_root,
        Path(args.best_ckpt).resolve(),
        Path(args.unsup_ckpt).resolve(),
        args.resume,
    )
    base_cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg = update_train_cfg(base_cfg, args, source_manifest, run_root)
    train_cfg_path = run_root / "config" / "goal_mode_train.yaml"
    write_yaml(train_cfg_path, cfg)
    audit_dataset(repo_dir, args.python_exe, run_root, train_cfg_path)

    if args.setup_only:
        print(f"[OK] setup complete: {run_root}")
        return 0

    best_ckpt = run_root / "outputs_final" / "train" / "checkpoints" / "best.pt"
    if not args.skip_train and not best_ckpt.exists():
        run_cmd(
            [args.python_exe, "-m", "src.train_loss_earlystop", "--config", str(train_cfg_path)],
            repo_dir,
            run_root / "logs" / "train_goal_mode.log",
        )

    if args.train_only:
        print(f"[OK] training complete: {best_ckpt}")
        return 0

    sweep = sweep_thresholds(repo_dir, args.python_exe, run_root, cfg, list(THRESHOLDS))
    final_info = copy_final_outputs(run_root, sweep["best_row"], sweep["best_config"])
    if not args.skip_viz:
        run_final_viz(repo_dir, args.python_exe, run_root, Path(final_info["final_config"]))
    report = acceptance_report(run_root, source_manifest, sweep["best_row"], final_info)
    package_path = package_final(run_root, report)
    report["package_path"] = str(package_path)
    write_json(run_root / "summary" / "acceptance_report.json", report)

    print(f"[ACCEPTED] {report['accepted']}")
    print(f"[P95] {report['metrics_summary'].get('p95_dist_mm', report['metrics_summary'].get('p95'))}")
    print(f"[SR@1.0] {summary_sr1(report['metrics_summary'])}")
    print(f"[BEST_CHECKPOINT] {report['best_checkpoint']}")
    print(f"[REPORT] {run_root / 'summary' / 'acceptance_report.json'}")
    print(f"[PACKAGE] {package_path}")
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
