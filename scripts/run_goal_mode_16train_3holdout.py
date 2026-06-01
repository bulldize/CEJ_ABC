#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_goal_mode_19full import (  # noqa: E402
    DEFAULT_BASE_CONFIG,
    DEFAULT_UNSUP_CKPT,
    HARD_CASES,
    PASS_SOURCES,
    THRESHOLDS,
    copy_final_outputs,
    count_tooth_dirs,
    fail_if_exists,
    load_bad_rows,
    load_summary,
    read_json,
    run_cmd,
    run_final_viz,
    summary_p95,
    summary_sr1,
    update_train_cfg as update_full19_train_cfg,
    validate_sources,
    write_failed_rows,
    write_json,
    write_yaml,
)


DEFAULT_STAGE1_BEST = Path(
    "/root/cej_isolated_runs/C/unsup_to_goal_retrain_20260601_012139_stage1/"
    "experiments/exp03_skeleton_aux_loss/outputs/train/checkpoints/best.pt"
)
DEFAULT_RUN_ROOT = Path("/root/cej_isolated_runs/C/goal模式_16train_3holdout_validation_001")
DEFAULT_REPRO_FULL19 = Path("/root/cej_isolated_runs/C/unsup_to_goal_retrain_20260601_012139_stage2_goal19")
DEFAULT_HISTORICAL_FULL19 = Path("/root/cej_isolated_runs/C/goal模式_19full_001")

TRAIN_CASES = [
    "ToothFairy3F_008",
    "ToothFairy3F_009",
    "ToothFairy3F_010",
    "ToothFairy3F_018",
    "ToothFairy3F_021",
    "ToothFairy3F_023",
    "ToothFairy3F_025",
    "ToothFairy3F_026",
    "ToothFairy3F_027",
    "ToothFairy3F_033",
    "ToothFairy3F_041",
    "ToothFairy3F_050",
    "ToothFairy3F_051",
    "ToothFairy3F_053",
    "ToothFairy3F_054",
    "ToothFairy3F_055",
]
HOLDOUT_CASES = [
    "ToothFairy3F_040",
    "ToothFairy3F_044",
    "ToothFairy3F_052",
]

EXPECTED_TRAIN_TOOTH_DIRS = 497
EXPECTED_HOLDOUT_TOOTH_DIRS = 88
EXPECTED_TOTAL_TOOTH_DIRS = 585


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def case_tooth_count(processed_root: Path, case_id: str) -> int:
    return len(glob.glob(str(processed_root / case_id / "tooth_*")))


def partition_source_rows(source_rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    row_by_case = {row["case_id"]: row for row in source_rows}
    required = set(TRAIN_CASES) | set(HOLDOUT_CASES)
    missing = sorted(required - set(row_by_case.keys()))
    if missing:
        raise RuntimeError(f"split cases missing from PASS sources: {missing}")
    extra = sorted(set(row_by_case.keys()) - required)
    if extra:
        raise RuntimeError(f"PASS sources contain cases outside the 16/3 split: {extra}")
    return [row_by_case[c] for c in TRAIN_CASES], [row_by_case[c] for c in HOLDOUT_CASES]


def write_case_manifest(path: Path, train_rows: List[Dict], holdout_rows: List[Dict], manifest: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "source",
        "case_id",
        "source_case_dir",
        "copied_case_dir",
        "usage_report",
        "tooth_dir_count",
    ]
    processed_train = Path(manifest["processed_train16"])
    processed_holdout = Path(manifest["processed_holdout3"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split_name, rows, root in (
            ("train16", train_rows, processed_train),
            ("holdout3", holdout_rows, processed_holdout),
        ):
            for row in rows:
                case_id = row["case_id"]
                writer.writerow(
                    {
                        "split": split_name,
                        "source": row["source"],
                        "case_id": case_id,
                        "source_case_dir": row["source_case_dir"],
                        "copied_case_dir": str(root / case_id),
                        "usage_report": row["usage_report"],
                        "tooth_dir_count": case_tooth_count(root, case_id),
                    }
                )


def choose_pretrained(best_ckpt: Path, unsup_ckpt: Path) -> Tuple[Path, str]:
    if best_ckpt.exists():
        return best_ckpt, "stage1_exp03_skeleton_aux_loss_best"
    if unsup_ckpt.exists():
        return unsup_ckpt, "unsup_pretrain_fallback"
    raise FileNotFoundError(f"neither pretrained checkpoint exists: {best_ckpt} / {unsup_ckpt}")


def validate_split_counts(manifest: Dict) -> None:
    expected = {
        "train_case_count": len(TRAIN_CASES),
        "holdout_case_count": len(HOLDOUT_CASES),
        "train_tooth_dir_count": EXPECTED_TRAIN_TOOTH_DIRS,
        "holdout_tooth_dir_count": EXPECTED_HOLDOUT_TOOTH_DIRS,
        "total_case_count": len(TRAIN_CASES) + len(HOLDOUT_CASES),
        "total_tooth_dir_count": EXPECTED_TOTAL_TOOTH_DIRS,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if list(manifest.get("train_cases", [])) != TRAIN_CASES:
        mismatches["train_cases"] = {"expected": TRAIN_CASES, "actual": manifest.get("train_cases")}
    if list(manifest.get("holdout_cases", [])) != HOLDOUT_CASES:
        mismatches["holdout_cases"] = {"expected": HOLDOUT_CASES, "actual": manifest.get("holdout_cases")}
    if mismatches:
        raise RuntimeError(f"16/3 split manifest does not match task spec: {json.dumps(mismatches, ensure_ascii=False)}")


def setup_inputs(run_root: Path, best_ckpt: Path, unsup_ckpt: Path) -> Dict:
    input_dir = run_root / "input"
    processed_train = input_dir / "processed_train16"
    processed_holdout = input_dir / "processed_holdout3"
    ckpt_dir = input_dir / "checkpoints"
    manifest_path = input_dir / "source_manifest.json"
    fail_if_exists(input_dir, "input directory")

    source_rows = validate_sources()
    train_rows, holdout_rows = partition_source_rows(source_rows)
    pretrained, pretrained_kind = choose_pretrained(best_ckpt, unsup_ckpt)

    input_dir.mkdir(parents=True, exist_ok=False)
    copied_train = copy_case_dirs(train_rows, processed_train)
    copied_holdout = copy_case_dirs(holdout_rows, processed_holdout)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pretrained_copy = ckpt_dir / "stage1_best.pt"
    shutil.copy2(pretrained, pretrained_copy)
    unsup_copy = None
    if unsup_ckpt.exists() and unsup_ckpt.resolve() != pretrained.resolve():
        unsup_copy = ckpt_dir / "unsup_last.pt"
        shutil.copy2(unsup_ckpt, unsup_copy)

    train_tooth_count = count_tooth_dirs(processed_train)
    holdout_tooth_count = count_tooth_dirs(processed_holdout)
    manifest = {
        "created_at": now_iso(),
        "run_root": str(run_root),
        "split_mode": "train16_holdout3",
        "processed_train16": str(processed_train),
        "processed_holdout3": str(processed_holdout),
        "train_case_count": len(train_rows),
        "holdout_case_count": len(holdout_rows),
        "total_case_count": len(train_rows) + len(holdout_rows),
        "train_tooth_dir_count": train_tooth_count,
        "holdout_tooth_dir_count": holdout_tooth_count,
        "total_tooth_dir_count": train_tooth_count + holdout_tooth_count,
        "tooth_dir_count": train_tooth_count,
        "tooth_count_for_schedule": train_tooth_count,
        "case_count": len(train_rows),
        "train_cases": TRAIN_CASES,
        "holdout_cases": HOLDOUT_CASES,
        "source_rows": source_rows,
        "train_source_rows": train_rows,
        "holdout_source_rows": holdout_rows,
        "copied_train_cases": copied_train,
        "copied_holdout_cases": copied_holdout,
        "pretrained_source": str(pretrained),
        "pretrained_copy": str(pretrained_copy),
        "pretrained_kind": pretrained_kind,
        "pretrained_source_sha256": sha256_file(pretrained),
        "pretrained_copy_sha256": sha256_file(pretrained_copy),
        "unsup_checkpoint_source": str(unsup_ckpt),
        "unsup_checkpoint_copy": str(unsup_copy) if unsup_copy else None,
        "hard_cases": HARD_CASES,
        "pass_sources": [
            {
                "name": src["name"],
                "processed_root": str(src["processed_root"]),
                "usage_report": str(src["usage_report"]),
                "cases": list(src["cases"]),
            }
            for src in PASS_SOURCES
        ],
    }
    validate_split_counts(manifest)
    write_json(manifest_path, manifest)
    write_case_manifest(input_dir / "case_manifest.csv", train_rows, holdout_rows, manifest)
    return manifest


def load_or_setup_inputs(run_root: Path, best_ckpt: Path, unsup_ckpt: Path, resume: bool) -> Dict:
    manifest_path = run_root / "input" / "source_manifest.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"--resume requested but input manifest is missing: {manifest_path}")
        manifest = read_json(manifest_path)
        validate_split_counts(manifest)
        return manifest
    return setup_inputs(run_root, best_ckpt, unsup_ckpt)


def check_lineage(manifest: Dict) -> Dict:
    src = Path(manifest.get("pretrained_source", ""))
    dst = Path(manifest.get("pretrained_copy", ""))
    checks = {
        "pretrained_source_exists": src.exists(),
        "pretrained_copy_exists": dst.exists(),
        "pretrained_source_is_stage1_best": str(src) == str(DEFAULT_STAGE1_BEST),
        "train_cases_match_spec": list(manifest.get("train_cases", [])) == TRAIN_CASES,
        "holdout_cases_match_spec": list(manifest.get("holdout_cases", [])) == HOLDOUT_CASES,
        "train_tooth_count_is_497": int(manifest.get("train_tooth_dir_count", 0) or 0) == EXPECTED_TRAIN_TOOTH_DIRS,
        "holdout_tooth_count_is_88": int(manifest.get("holdout_tooth_dir_count", 0) or 0) == EXPECTED_HOLDOUT_TOOTH_DIRS,
        "schedule_tooth_count_is_train_split": int(manifest.get("tooth_dir_count", 0) or 0)
        == int(manifest.get("train_tooth_dir_count", -1) or -1),
    }
    source_sha = None
    copy_sha = None
    if src.exists():
        source_sha = sha256_file(src)
    if dst.exists():
        copy_sha = sha256_file(dst)
    checks["pretrained_copy_sha256_matches_source"] = bool(source_sha and copy_sha and source_sha == copy_sha)
    return {
        "checks": checks,
        "ok": all(checks.values()),
        "pretrained_source_sha256": source_sha,
        "pretrained_copy_sha256": copy_sha,
    }


def choose_stage2_schedule(args: argparse.Namespace, source_manifest: Dict) -> Dict:
    train_tooth_count = int(source_manifest.get("train_tooth_dir_count", source_manifest.get("tooth_dir_count", 0)) or 0)
    tooth_count = max(1, train_tooth_count)
    batch_size = max(1, int(args.batch_size))
    steps_per_epoch = max(1, math.ceil(tooth_count / batch_size))
    target_steps = max(1, int(args.target_train_steps))

    if args.score_start_epoch is None:
        score_start_epoch = math.ceil(0.85 * target_steps / steps_per_epoch)
        score_start_epoch = max(2, min(int(args.auto_max_score_start_epoch), score_start_epoch))
    else:
        score_start_epoch = int(args.score_start_epoch)

    if args.max_epochs is None:
        max_epochs = math.ceil(target_steps / steps_per_epoch)
        max_epochs = max(max_epochs, int(args.auto_min_epochs))
        max_epochs = min(max_epochs, int(args.auto_max_epochs))
    else:
        max_epochs = int(args.max_epochs)

    if args.min_epochs is None:
        min_epochs = max_epochs
    else:
        min_epochs = int(args.min_epochs)

    args.score_start_epoch = score_start_epoch
    args.max_epochs = max_epochs
    args.min_epochs = min_epochs
    return {
        "train_tooth_count": train_tooth_count,
        "tooth_count_for_schedule": tooth_count,
        "tooth_count": tooth_count,
        "batch_size": batch_size,
        "estimated_steps_per_epoch": steps_per_epoch,
        "target_train_steps": target_steps,
        "score_start_epoch": score_start_epoch,
        "max_epochs": max_epochs,
        "min_epochs": min_epochs,
        "formula": "ceil(target_train_steps / ceil(train_tooth_count / batch_size)), bounded by auto_min/auto_max",
    }


def update_train_cfg(cfg: Dict, args: argparse.Namespace, source_manifest: Dict, run_root: Path) -> Dict:
    full19_compatible_manifest = dict(source_manifest)
    full19_compatible_manifest["processed_all19"] = source_manifest["processed_train16"]
    cfg = update_full19_train_cfg(cfg, args, full19_compatible_manifest, run_root)
    cfg["data"]["processed_dir"] = source_manifest["processed_train16"]
    cfg["data"]["holdout_processed_dir"] = source_manifest["processed_holdout3"]
    cfg["data"]["output_dir"] = str(run_root / "outputs_final")
    cfg["train"]["pretrained_ckpt"] = source_manifest["pretrained_copy"]
    cfg["train"]["split_mode"] = "train16_holdout3"
    cfg["train"]["train_case_count"] = source_manifest["train_case_count"]
    cfg["train"]["holdout_case_count"] = source_manifest["holdout_case_count"]
    cfg["train"]["train_tooth_dir_count"] = source_manifest["train_tooth_dir_count"]
    cfg["train"]["holdout_tooth_dir_count"] = source_manifest["holdout_tooth_dir_count"]
    cfg["viz"]["max_cases"] = 3
    return cfg


def audit_split_dataset(repo_dir: Path, python_exe: str, run_root: Path, cfg_path: Path) -> None:
    code = (
        "import json; "
        "from pathlib import Path; "
        "from src.datasets.dataset import build_supervised_audit, write_supervised_audit; "
        "from src.utils.config import load_config; "
        f"cfg=load_config({str(cfg_path)!r}); "
        f"root=Path({str(run_root / 'input')!r}); "
        "out={}; "
        "specs=[('train16', cfg['data']['processed_dir']), ('holdout3', cfg['data']['holdout_processed_dir'])]; "
        "\nfor name, p in specs:\n"
        "    audit=build_supervised_audit(p, cfg['data'].get('processed_format','nii.gz'))\n"
        "    paths=write_supervised_audit(audit, str(root / ('audit_' + name)))\n"
        "    out[name]={'audit': audit, 'paths': paths}\n"
        "print(json.dumps(out, ensure_ascii=False))"
    )
    run_cmd([python_exe, "-c", code], repo_dir, run_root / "logs" / "audit_split.log")


def create_holdout_threshold_config(base_cfg: Dict, source_manifest: Dict, threshold: float, out_dir: Path, ckpt_path: Path) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["data"]["processed_dir"] = source_manifest["processed_holdout3"]
    cfg["data"]["holdout_processed_dir"] = source_manifest["processed_holdout3"]
    cfg["data"]["output_dir"] = str(out_dir)
    cfg.setdefault("infer", {})["ckpt_path"] = str(ckpt_path)
    cfg["infer"]["threshold_theta"] = float(threshold)
    cfg.setdefault("holdout_eval", {})["threshold_theta"] = float(threshold)
    cfg.setdefault("viz", {})["max_cases"] = 3
    return cfg


def load_csv_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def holdout_case_count_from_rows(rows: List[Dict]) -> int:
    return len({row.get("case_id") for row in rows if row.get("case_id")})


def validation_checks(source_manifest: Dict, summary: Dict, per_tooth_rows: List[Dict], failed_or_bad_teeth_count: int) -> Dict:
    holdout_case_count = holdout_case_count_from_rows(per_tooth_rows)
    holdout_tooth_count = int(summary.get("tooth_count", 0) or 0)
    missing_prediction_count = int(summary.get("missing_prediction_count", 0) or 0)
    no_curve_count = int(summary.get("no_curve_count", 0) or 0)
    failed_tooth_count = int(summary.get("failed_tooth_count", 0) or 0)
    mean_dist_mm = summary.get("mean_dist_mm", summary.get("mean"))
    mean_dist_mm = float(mean_dist_mm) if mean_dist_mm is not None else float("inf")
    p95_dist_mm = summary_p95(summary)
    sr1 = summary_sr1(summary)
    lineage = check_lineage(source_manifest)

    base = {
        "holdout_case_count_is_3": holdout_case_count == 3,
        "holdout_tooth_count_is_88": holdout_tooth_count == 88,
        "lineage_ok": bool(lineage["ok"]),
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
        "holdout_tooth_count_wrong": holdout_tooth_count != 88,
        "lineage_not_ok": not bool(lineage["ok"]),
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
        "lineage": lineage,
        "holdout_case_count": holdout_case_count,
        "holdout_tooth_count": holdout_tooth_count,
        "expected_holdout_case_count": int(source_manifest.get("holdout_case_count", 0) or 0),
        "expected_holdout_tooth_count": int(source_manifest.get("holdout_tooth_dir_count", 0) or 0),
        "missing_prediction_count": missing_prediction_count,
        "no_curve_count": no_curve_count,
        "failed_tooth_count": failed_tooth_count,
        "failed_or_bad_teeth_count": int(failed_or_bad_teeth_count),
        "mean_dist_mm": mean_dist_mm,
        "p95_dist_mm": p95_dist_mm,
        "sr@1.0mm": sr1,
    }


def sweep_thresholds_holdout(
    repo_dir: Path,
    python_exe: str,
    run_root: Path,
    cfg: Dict,
    thresholds: List[float],
    source_manifest: Dict,
    sweep_all_thresholds: bool,
) -> Dict:
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
        threshold_cfg = create_holdout_threshold_config(cfg, source_manifest, threshold, out_dir, ckpt_path)
        write_yaml(cfg_path, threshold_cfg)
        if not (out_dir / "eval" / "metrics_summary.json").exists():
            run_cmd([python_exe, "-m", "src.infer", "--config", str(cfg_path)], repo_dir, run_root / "logs" / f"infer_{label}.log")
            run_cmd([python_exe, "-m", "src.eval", "--config", str(cfg_path)], repo_dir, run_root / "logs" / f"eval_{label}.log")
        eval_dir = out_dir / "eval"
        summary = load_summary(eval_dir)
        per_tooth = eval_dir / "metrics_per_tooth.csv"
        per_tooth_rows = load_csv_rows(per_tooth)
        bad_rows = load_bad_rows(per_tooth)
        checks = validation_checks(source_manifest, summary, per_tooth_rows, len(bad_rows))
        row = {
            "threshold": threshold,
            "output_dir": str(out_dir),
            "metrics_summary": str(eval_dir / "metrics_summary.json"),
            "metrics_per_tooth": str(per_tooth),
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
        status_rank = {"PASS_STRONG": 0, "PASS_USABLE": 1, "FAIL": 2}.get(checks["status"], 2)
        candidates.append(
            (
                status_rank,
                0 if checks["missing_prediction_count"] == 0 else 1,
                0 if checks["no_curve_count"] == 0 else 1,
                checks["p95_dist_mm"],
                -checks["sr@1.0mm"],
                checks["failed_or_bad_teeth_count"],
                threshold,
                row,
                threshold_cfg,
            )
        )
        if not sweep_all_thresholds and checks["status"] == "PASS_STRONG":
            break

    sweep_csv = run_root / "summary" / "threshold_sweep.csv"
    sweep_csv.parent.mkdir(parents=True, exist_ok=True)
    with sweep_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    candidates.sort(key=lambda item: item[:7])
    best = candidates[0]
    return {
        "sweep_csv": str(sweep_csv),
        "rows": rows,
        "best_row": best[7],
        "best_config": best[8],
    }


def load_train_manifest(run_root: Path) -> Dict:
    path = run_root / "outputs_final" / "train" / "train_manifest.json"
    return read_json(path) if path.exists() else {}


def validation_report(run_root: Path, source_manifest: Dict, best_row: Dict, final_info: Dict) -> Dict:
    final_eval = Path(final_info["final_root"]) / "eval"
    viewer_index = Path(final_info["final_root"]) / "viz" / "3d" / "index.html"
    summary = load_summary(final_eval)
    per_tooth = final_eval / "metrics_per_tooth.csv"
    per_tooth_rows = load_csv_rows(per_tooth)
    bad_rows = load_bad_rows(per_tooth)
    failed_csv = final_eval / "failed_or_bad_teeth.csv"
    write_failed_rows(failed_csv, bad_rows)
    checks = validation_checks(source_manifest, summary, per_tooth_rows, len(bad_rows))
    train_manifest = load_train_manifest(run_root)
    report = {
        "created_at": now_iso(),
        "validation_status": checks["status"],
        "accepted_level": checks["status"],
        "criteria": {
            "pass_strong": checks["strong_criteria"],
            "pass_usable": checks["usable_criteria"],
            "fail_triggers": checks["fail_triggers"],
        },
        "lineage": checks["lineage"],
        "train_case_count": source_manifest["train_case_count"],
        "train_tooth_count": source_manifest["train_tooth_dir_count"],
        "holdout_case_count": checks["holdout_case_count"],
        "holdout_tooth_count": checks["holdout_tooth_count"],
        "expected_holdout_case_count": checks["expected_holdout_case_count"],
        "expected_holdout_tooth_count": checks["expected_holdout_tooth_count"],
        "missing_prediction_count": checks["missing_prediction_count"],
        "no_curve_count": checks["no_curve_count"],
        "failed_tooth_count": checks["failed_tooth_count"],
        "failed_or_bad_teeth_count": checks["failed_or_bad_teeth_count"],
        "mean_dist_mm": checks["mean_dist_mm"],
        "p95_dist_mm": checks["p95_dist_mm"],
        "sr@0.5mm": summary.get("sr@0.5mm"),
        "sr@1.0mm": checks["sr@1.0mm"],
        "sr@1.5mm": summary.get("sr@1.5mm"),
        "sr@2.0mm": summary.get("sr@2.0mm"),
        "selected_threshold": best_row["threshold"],
        "selected_threshold_row": best_row,
        "bad_teeth": bad_rows,
        "metrics_summary": summary,
        "metrics_summary_path": str(final_eval / "metrics_summary.json"),
        "metrics_per_tooth_path": str(per_tooth),
        "failed_or_bad_teeth_path": str(failed_csv),
        "case_manifest": str(run_root / "input" / "case_manifest.csv"),
        "source_manifest": str(run_root / "input" / "source_manifest.json"),
        "stage2_schedule": str(run_root / "summary" / "stage2_schedule.json"),
        "train_manifest": str(run_root / "outputs_final" / "train" / "train_manifest.json"),
        "stage2_validation_best_epoch": train_manifest.get("best_epoch"),
        "best_checkpoint": str(run_root / "outputs_final" / "train" / "checkpoints" / "best.pt"),
        "last_checkpoint": str(run_root / "outputs_final" / "train" / "checkpoints" / "last.pt"),
        "topk_dir": str(run_root / "outputs_final" / "train" / "checkpoints" / "topk"),
        "viewer_index": str(viewer_index) if viewer_index.exists() else None,
        "final_config": final_info["final_config"],
    }
    write_json(run_root / "summary" / "validation_report.json", report)
    return report


def select_full19_baseline(repro_full19: Path, historical_full19: Path) -> Tuple[Path, str]:
    repro_report = repro_full19 / "summary" / "acceptance_report.json"
    if repro_report.exists():
        try:
            report = read_json(repro_report)
            if bool(report.get("accepted")):
                return repro_full19, "current_repro_accepted"
        except Exception:
            pass
    return historical_full19, "historical_success_baseline"


def load_baseline_metrics(baseline_root: Path) -> Dict:
    report_candidates = [
        baseline_root / "summary" / "acceptance_report.json",
        baseline_root / "summary" / "reproduction_lineage_acceptance_report.json",
    ]
    report = {}
    report_path = None
    for path in report_candidates:
        if path.exists():
            report = read_json(path)
            report_path = path
            break
    metrics_path = baseline_root / "final" / "eval" / "metrics_summary.json"
    metrics = report.get("metrics_summary") if isinstance(report.get("metrics_summary"), dict) else {}
    if not metrics and metrics_path.exists():
        metrics = read_json(metrics_path)
    return {
        "report_path": str(report_path) if report_path else None,
        "metrics_summary_path": str(metrics_path) if metrics_path.exists() else None,
        "accepted": report.get("accepted"),
        "validation_status": report.get("validation_status"),
        "selected_threshold": report.get("selected_threshold"),
        "metrics_summary": metrics,
    }


def repo_commit(repo_dir: Path) -> Optional[str]:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    except Exception:
        return None


def make_recommendation(status: str, report: Dict) -> Dict:
    if status == "PASS_STRONG":
        return {
            "recommendation": "continue_to_stage1_full19",
            "reason": "holdout3 met PASS_STRONG with clean predictions and strong final eval metrics.",
        }
    if status == "PASS_USABLE":
        return {
            "recommendation": "tune_stage2",
            "reason": "holdout3 met usable generalization but missed the stronger target; inspect bad teeth and tune Stage 2 before a Stage 1 full19 push.",
        }
    triggers = [k for k, v in report.get("criteria", {}).get("fail_triggers", {}).items() if v]
    return {
        "recommendation": "inspect_holdout_failures",
        "reason": "holdout3 failed validation triggers: " + ", ".join(triggers),
    }


def comparison_report(
    repo_dir: Path,
    run_root: Path,
    source_manifest: Dict,
    validation: Dict,
    baseline_root: Path,
    baseline_reason: str,
) -> Dict:
    stage1_source = Path(source_manifest["pretrained_source"])
    train_manifest = load_train_manifest(run_root)
    baseline = load_baseline_metrics(baseline_root)
    stage1_run_root = stage1_source.parents[5] if len(stage1_source.parents) > 5 else stage1_source.parent
    report = {
        "created_at": now_iso(),
        "repo_commit": repo_commit(repo_dir),
        "stage1_run_root": str(stage1_run_root),
        "stage1_best_checkpoint": {
            "path": str(stage1_source),
            "size_bytes": stage1_source.stat().st_size if stage1_source.exists() else None,
            "sha256": validation.get("lineage", {}).get("pretrained_source_sha256"),
        },
        "stage2_validation_run_root": str(run_root),
        "stage2_full19_baseline_run_root": str(baseline_root),
        "stage2_full19_baseline_selection_reason": baseline_reason,
        "train_cases": source_manifest["train_cases"],
        "holdout_cases": source_manifest["holdout_cases"],
        "train_tooth_count": source_manifest["train_tooth_dir_count"],
        "holdout_tooth_count": source_manifest["holdout_tooth_dir_count"],
        "stage2_validation_best_epoch": train_manifest.get("best_epoch"),
        "stage2_validation_final_selected_threshold": validation.get("selected_threshold"),
        "holdout_final_metrics": validation.get("metrics_summary"),
        "bad_teeth_list": validation.get("bad_teeth", []),
        "full19_baseline": baseline,
    }
    report["comparison_conclusion"] = make_recommendation(validation["validation_status"], validation)
    write_json(run_root / "summary" / "stage2_16train_3holdout_comparison_report.json", report)
    return report


def package_final(run_root: Path) -> Path:
    package_dir = run_root / "package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    package_path = package_dir / "goal模式_16train_3holdout_validation_bundle.tar.gz"
    include = [
        run_root / "config",
        run_root / "input" / "source_manifest.json",
        run_root / "input" / "case_manifest.csv",
        run_root / "input" / "audit_train16",
        run_root / "input" / "audit_holdout3",
        run_root / "outputs_final" / "train" / "checkpoints" / "best.pt",
        run_root / "outputs_final" / "train" / "checkpoints" / "last.pt",
        run_root / "outputs_final" / "train" / "checkpoints" / "topk",
        run_root / "outputs_final" / "train" / "metrics.csv",
        run_root / "outputs_final" / "train" / "train_manifest.json",
        run_root / "outputs_final" / "train" / "audit_train",
        run_root / "outputs_final" / "train" / "audit_holdout",
        run_root / "threshold_sweep",
        run_root / "final",
        run_root / "summary",
        run_root / "logs",
    ]
    with tarfile.open(package_path, "w:gz") as tf:
        for path in include:
            if not path.exists():
                continue
            tf.add(path, arcname=str(path.relative_to(run_root)))
    return package_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run goal-mode CEJ Stage 2 with 16 train cases and 3 holdout cases.")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--best-ckpt", default=str(DEFAULT_STAGE1_BEST))
    parser.add_argument("--unsup-ckpt", default=str(DEFAULT_UNSUP_CKPT))
    parser.add_argument("--full19-repro-root", default=str(DEFAULT_REPRO_FULL19))
    parser.add_argument("--full19-historical-root", default=str(DEFAULT_HISTORICAL_FULL19))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--sweep-all-thresholds", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--score-start-epoch", type=int, default=None)
    parser.add_argument("--target-train-steps", type=int, default=3437)
    parser.add_argument("--auto-min-epochs", type=int, default=4)
    parser.add_argument("--auto-max-epochs", type=int, default=30)
    parser.add_argument("--auto-max-score-start-epoch", type=int, default=10)
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
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
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
    lineage = check_lineage(source_manifest)
    if not lineage["ok"]:
        raise RuntimeError(f"lineage checks failed before training: {json.dumps(lineage, ensure_ascii=False)}")

    schedule = choose_stage2_schedule(args, source_manifest)
    write_json(run_root / "summary" / "stage2_schedule.json", schedule)
    print(f"[SCHEDULE] {json.dumps(schedule, ensure_ascii=False)}")
    base_cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg = update_train_cfg(base_cfg, args, source_manifest, run_root)
    train_cfg_path = run_root / "config" / "goal_mode_train.yaml"
    write_yaml(train_cfg_path, cfg)
    audit_split_dataset(repo_dir, args.python_exe, run_root, train_cfg_path)

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

    sweep = sweep_thresholds_holdout(
        repo_dir,
        args.python_exe,
        run_root,
        cfg,
        list(THRESHOLDS),
        source_manifest,
        sweep_all_thresholds=args.sweep_all_thresholds,
    )
    final_info = copy_final_outputs(run_root, sweep["best_row"], sweep["best_config"])
    if not args.skip_viz:
        run_final_viz(repo_dir, args.python_exe, run_root, Path(final_info["final_config"]))
    validation = validation_report(run_root, source_manifest, sweep["best_row"], final_info)
    baseline_root, baseline_reason = select_full19_baseline(
        Path(args.full19_repro_root).resolve(),
        Path(args.full19_historical_root).resolve(),
    )
    comparison = comparison_report(repo_dir, run_root, source_manifest, validation, baseline_root, baseline_reason)
    package_path = package_final(run_root)
    validation["package_path"] = str(package_path)
    validation["comparison_report"] = str(run_root / "summary" / "stage2_16train_3holdout_comparison_report.json")
    validation["comparison_conclusion"] = comparison["comparison_conclusion"]
    write_json(run_root / "summary" / "validation_report.json", validation)

    print(f"[VALIDATION_STATUS] {validation['validation_status']}")
    print(f"[P95] {validation['p95_dist_mm']}")
    print(f"[SR@1.0] {validation['sr@1.0mm']}")
    print(f"[BEST_CHECKPOINT] {validation['best_checkpoint']}")
    print(f"[REPORT] {run_root / 'summary' / 'validation_report.json'}")
    print(f"[COMPARISON] {run_root / 'summary' / 'stage2_16train_3holdout_comparison_report.json'}")
    print(f"[PACKAGE] {package_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
