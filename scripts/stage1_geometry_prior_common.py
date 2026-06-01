#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_goal_mode_16train_3holdout as split16  # noqa: E402
import run_goal_mode_19full as goal19  # noqa: E402
from upgrade_processed_geometry_gt import _case_tooth_dirs, _load_roi_centroids, upgrade_processed_dir  # noqa: E402
from src.datasets.cej_geometry import (  # noqa: E402
    build_geometry_prior,
    compute_curve_fit_report,
    geometry_prior_to_jsonable,
)
from src.datasets.dataset import build_supervised_audit, list_tooth_dirs, write_supervised_audit  # noqa: E402
from src.datasets.heatmap import fit_curve_and_sample, generate_heatmap_from_points  # noqa: E402
from src.datasets.io import load_volume, save_volume  # noqa: E402
from src.datasets.points import get_points_for_tooth, load_points  # noqa: E402


DEFAULT_RUN_PARENT = Path("/root/cej_isolated_runs/C")
DEFAULT_STAGE1_CONFIG = Path(
    "/root/cej_isolated_runs/C/unsup_to_goal_retrain_20260601_012139_stage1/"
    "experiments/exp03_skeleton_aux_loss/config/train.yaml"
)
DEFAULT_BASE_CONFIG = DEFAULT_STAGE1_CONFIG if DEFAULT_STAGE1_CONFIG.exists() else REPO_ROOT / "configs/default.yaml"
DEFAULT_UNSUP_CKPT = Path("/root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt")
THRESHOLDS = [0.20, 0.25, 0.30, 0.35]
GT_ARTIFACT_NAMES = ("curve_dense_points.npy", "H_GT.nii.gz", "C_GT.nii.gz")
STAGE1_HOLDOUT_HARD_CASES = [
    "052-31/32/41/42/33",
    "040-42/41/32",
    "044-31/32/36/46",
]
BASELINE_STAGE1_ONLY = {
    "experiment": "baseline Stage1-only",
    "threshold": 0.25,
    "mean_dist_mm": 0.628,
    "p95_dist_mm": 1.417,
    "sr@1.0mm": 0.846,
    "failed_or_bad_teeth_count": 3,
    "status": "baseline",
}
NON_INFERIORITY_GOAL = {
    "p95_dist_mm_lte": 1.467,
    "sr@1.0mm_gte": 0.836,
    "failed_or_bad_teeth_count_lte": 3,
    "missing_prediction_count_eq": 0,
    "no_curve_count_eq": 0,
    "holdout_tooth_count_eq": 88,
}


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
    return len(glob.glob(str(processed_root / "*" / "tooth_*")))


def count_case_tooth_dirs(processed_root: Path, case_id: str) -> int:
    return len(glob.glob(str(processed_root / case_id / "tooth_*")))


def fail_if_exists(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} exists, refusing to overwrite: {path}")


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
    roots = {
        "train16": Path(manifest["processed_train16"]),
        "holdout3": Path(manifest["processed_holdout3"]),
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split_name, rows in (("train16", train_rows), ("holdout3", holdout_rows)):
            root = roots[split_name]
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
                        "tooth_dir_count": count_case_tooth_dirs(root, case_id),
                    }
                )


def validate_split_counts(manifest: Dict) -> None:
    expected = {
        "train_case_count": len(split16.TRAIN_CASES),
        "holdout_case_count": len(split16.HOLDOUT_CASES),
        "train_tooth_dir_count": split16.EXPECTED_TRAIN_TOOTH_DIRS,
        "holdout_tooth_dir_count": split16.EXPECTED_HOLDOUT_TOOTH_DIRS,
        "total_tooth_dir_count": split16.EXPECTED_TOTAL_TOOTH_DIRS,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if list(manifest.get("train_cases", [])) != split16.TRAIN_CASES:
        mismatches["train_cases"] = {"expected": split16.TRAIN_CASES, "actual": manifest.get("train_cases")}
    if list(manifest.get("holdout_cases", [])) != split16.HOLDOUT_CASES:
        mismatches["holdout_cases"] = {"expected": split16.HOLDOUT_CASES, "actual": manifest.get("holdout_cases")}
    if mismatches:
        raise RuntimeError(f"16/3 split checks failed: {json.dumps(mismatches, ensure_ascii=False)}")


def setup_inputs(run_root: Path, unsup_ckpt: Path, resume: bool) -> Dict:
    manifest_path = run_root / "input" / "source_manifest.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"--resume requested but source manifest is missing: {manifest_path}")
        manifest = read_json(manifest_path)
        validate_split_counts(manifest)
        return manifest

    input_dir = run_root / "input"
    processed_train = input_dir / "processed_train16"
    processed_holdout = input_dir / "processed_holdout3"
    ckpt_dir = input_dir / "checkpoints"
    fail_if_exists(input_dir, "input directory")
    if not unsup_ckpt.exists():
        raise FileNotFoundError(f"unsupervised checkpoint missing: {unsup_ckpt}")

    source_rows = goal19.validate_sources()
    train_rows, holdout_rows = split16.partition_source_rows(source_rows)
    input_dir.mkdir(parents=True, exist_ok=False)
    copied_train = copy_case_dirs(train_rows, processed_train)
    copied_holdout = copy_case_dirs(holdout_rows, processed_holdout)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pretrained_copy = ckpt_dir / "unsup_last.pt"
    shutil.copy2(unsup_ckpt, pretrained_copy)

    train_tooth_count = count_tooth_dirs(processed_train)
    holdout_tooth_count = count_tooth_dirs(processed_holdout)
    manifest = {
        "created_at": now_iso(),
        "run_root": str(run_root),
        "split_mode": "train16_holdout3",
        "objective": "Stage1-only geometry-prior experiment on fixed 16 train / 3 holdout split.",
        "processed_train16": str(processed_train),
        "processed_holdout3": str(processed_holdout),
        "train_case_count": len(train_rows),
        "holdout_case_count": len(holdout_rows),
        "total_case_count": len(train_rows) + len(holdout_rows),
        "train_tooth_dir_count": train_tooth_count,
        "holdout_tooth_dir_count": holdout_tooth_count,
        "total_tooth_dir_count": train_tooth_count + holdout_tooth_count,
        "tooth_dir_count": train_tooth_count,
        "train_cases": split16.TRAIN_CASES,
        "holdout_cases": split16.HOLDOUT_CASES,
        "source_rows": source_rows,
        "train_source_rows": train_rows,
        "holdout_source_rows": holdout_rows,
        "copied_train_cases": copied_train,
        "copied_holdout_cases": copied_holdout,
        "pretrained_source": str(unsup_ckpt),
        "pretrained_copy": str(pretrained_copy),
        "pretrained_kind": "unsup_pretrain_last",
        "pretrained_source_sha256": sha256_file(unsup_ckpt),
        "pretrained_copy_sha256": sha256_file(pretrained_copy),
        "thresholds": THRESHOLDS,
    }
    validate_split_counts(manifest)
    write_json(manifest_path, manifest)
    write_case_manifest(input_dir / "case_manifest.csv", train_rows, holdout_rows, manifest)
    return manifest


def safe_git_status(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(repo_dir), text=True)
    except Exception as exc:
        return f"git status failed: {exc}"


def cuda_check() -> Dict:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "cuda_available": available,
            "device_count": int(torch.cuda.device_count()) if available else 0,
            "device_name": torch.cuda.get_device_name(0) if available else None,
        }
    except Exception as exc:
        return {"cuda_available": False, "error": str(exc)}


def write_preflight(repo_dir: Path, run_root: Path, manifest: Dict) -> Dict:
    src = Path(manifest["pretrained_source"])
    dst = Path(manifest["pretrained_copy"])
    checks = {
        "git_status_short": safe_git_status(repo_dir),
        "cuda": cuda_check(),
        "unsup_checkpoint_exists": src.exists(),
        "pretrained_copy_exists": dst.exists(),
        "train_tooth_dirs": count_tooth_dirs(Path(manifest["processed_train16"])),
        "holdout_tooth_dirs": count_tooth_dirs(Path(manifest["processed_holdout3"])),
        "expected_train_tooth_dirs": split16.EXPECTED_TRAIN_TOOTH_DIRS,
        "expected_holdout_tooth_dirs": split16.EXPECTED_HOLDOUT_TOOTH_DIRS,
        "pretrained_source_sha256": sha256_file(src) if src.exists() else None,
        "pretrained_copy_sha256": sha256_file(dst) if dst.exists() else None,
    }
    checks["train_tooth_dirs_ok"] = checks["train_tooth_dirs"] == split16.EXPECTED_TRAIN_TOOTH_DIRS
    checks["holdout_tooth_dirs_ok"] = checks["holdout_tooth_dirs"] == split16.EXPECTED_HOLDOUT_TOOTH_DIRS
    checks["pretrained_hash_ok"] = (
        bool(checks["pretrained_source_sha256"])
        and checks["pretrained_source_sha256"] == checks["pretrained_copy_sha256"]
    )
    checks["ok"] = bool(
        checks["cuda"].get("cuda_available")
        and checks["unsup_checkpoint_exists"]
        and checks["pretrained_copy_exists"]
        and checks["train_tooth_dirs_ok"]
        and checks["holdout_tooth_dirs_ok"]
        and checks["pretrained_hash_ok"]
    )
    write_json(run_root / "input" / "preflight_checks.json", checks)
    return checks


def load_base_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"base config missing: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_upgrade_cfg(base_cfg: Dict, processed_dir: Path, shape_prior: bool, shape_prior_sigma_mm: float) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))
    data = cfg.setdefault("data", {})
    data["processed_dir"] = str(processed_dir)
    data["processed_format"] = "nii.gz"
    preprocess = cfg.setdefault("preprocess", {})
    geometry = preprocess.setdefault("geometry_prior", {})
    geometry["write_shape_prior_heatmap"] = bool(shape_prior)
    geometry["shape_prior_sigma_mm"] = float(shape_prior_sigma_mm)
    geometry["shape_prior_name"] = "H_SHAPE_PRIOR"
    preprocess["shape_prior_sigma_mm"] = float(shape_prior_sigma_mm)
    preprocess["shape_prior_name"] = "H_SHAPE_PRIOR"
    return cfg


def _shape_prior_cfg(base_cfg: Dict, shape_prior_sigma_mm: float) -> Dict:
    preprocess_cfg = dict(base_cfg.get("preprocess", {}))
    geometry_cfg = dict(preprocess_cfg.get("geometry_prior", {}))
    return {
        "fmt": base_cfg.get("data", {}).get("processed_format", "nii.gz"),
        "step_mm": float(preprocess_cfg.get("dense_sample_step_mm", 0.2)),
        "sigma_mm": float(shape_prior_sigma_mm),
        "curve_closed": bool(preprocess_cfg.get("curve_closed", True)),
        "curve_smooth": float(preprocess_cfg.get("curve_smooth", 0.0)),
        "constraints": geometry_cfg.get("constraints", {}),
        "shape_prior_name": "H_SHAPE_PRIOR",
    }


def collect_gt_artifact_hashes(processed_dir: Path, fmt: str = "nii.gz") -> Dict:
    records = {}
    tooth_dirs = list_tooth_dirs(str(processed_dir))
    for tdir_raw in tooth_dirs:
        tdir = Path(tdir_raw)
        for artifact in GT_ARTIFACT_NAMES:
            path = tdir / artifact
            rel = str(path.relative_to(processed_dir))
            records[rel] = {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
            }
    return {
        "processed_dir": str(processed_dir),
        "processed_format": fmt,
        "tooth_dir_count": len(tooth_dirs),
        "gt_artifact_names": list(GT_ARTIFACT_NAMES),
        "artifact_record_count": len(records),
        "records": records,
    }


def compare_gt_artifact_hashes(before: Dict, after: Dict) -> Dict:
    changed = []
    removed = []
    new_files = []
    before_records = before.get("records", {})
    after_records = after.get("records", {})
    for rel in sorted(set(before_records) | set(after_records)):
        b = before_records.get(rel, {"exists": False, "sha256": None})
        a = after_records.get(rel, {"exists": False, "sha256": None})
        if b.get("exists") and not a.get("exists"):
            removed.append(rel)
        elif not b.get("exists") and a.get("exists"):
            new_files.append(rel)
        elif b.get("exists") and a.get("exists") and b.get("sha256") != a.get("sha256"):
            changed.append(rel)
    mismatch_count = len(changed) + len(removed) + len(new_files)
    return {
        "processed_dir": before.get("processed_dir"),
        "tooth_dir_count": before.get("tooth_dir_count"),
        "artifact_record_count": before.get("artifact_record_count"),
        "changed_count": len(changed),
        "removed_count": len(removed),
        "unexpected_new_count": len(new_files),
        "mismatch_count": mismatch_count,
        "gt_preserved": mismatch_count == 0,
        "changed": changed,
        "removed": removed,
        "unexpected_new": new_files,
    }


def _save_shape_prior_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _generate_shape_prior_for_split(
    processed_dir: Path,
    base_cfg: Dict,
    shape_prior_sigma_mm: float,
) -> Dict:
    cfg = _shape_prior_cfg(base_cfg, shape_prior_sigma_mm)
    fmt = cfg["fmt"]
    generated = 0
    skipped = []
    errors = []
    for _, tooth_dirs in _case_tooth_dirs(str(processed_dir)):
        centroids, arch_centroid = _load_roi_centroids(tooth_dirs, fmt)
        for tdir_raw in tooth_dirs:
            tdir = Path(tdir_raw)
            meta_path = tdir / "roi_meta.json"
            if not meta_path.exists():
                skipped.append({"tooth_dir": str(tdir), "reason": "missing_roi_meta"})
                continue
            try:
                meta = read_json(meta_path)
                case_id = meta["case_id"]
                tooth_id = int(meta["tooth_id"])
                A, spacing, _ = load_volume(str(tdir / f"A_t.{fmt}"), dtype=None)
                T, _, _ = load_volume(str(tdir / f"T_t.{fmt}"), dtype=np.uint8)
                points_data = load_points(str(tdir / "points.json"))
                pts = get_points_for_tooth(points_data, tooth_id).astype(np.float32)

                geometry_prior = build_geometry_prior(
                    T,
                    spacing=spacing,
                    tooth_id=tooth_id,
                    case_id=case_id,
                    points_vox=pts,
                    full_tooth_centroids_mm=centroids,
                    arch_centroid_mm=arch_centroid,
                    roi_origin_vox=meta.get("roi_origin_in_full", [0, 0, 0]),
                    spacing_full=meta.get("spacing_full", spacing),
                    constraint_params=cfg["constraints"],
                )
                dense_pts, fit_report = fit_curve_and_sample(
                    pts,
                    spacing=spacing,
                    step_mm=cfg["step_mm"],
                    closed=cfg["curve_closed"],
                    smooth=cfg["curve_smooth"],
                    geometry_prior=geometry_prior,
                    return_report=True,
                )
                dense_source = "geometry_prior_fit"
                if dense_pts.shape[0] == 0 and (tdir / "curve_dense_points.npy").exists():
                    dense_pts = np.load(tdir / "curve_dense_points.npy").astype(np.float32)
                    dense_source = "existing_curve_dense_points_fallback"
                    fit_report = {
                        "used_geometry_prior": False,
                        "fallback": True,
                        "fallback_reason": "no_points_used_existing_curve_dense_points",
                        "n_input_points": int(pts.shape[0]) if pts.ndim == 2 else 0,
                        "n_output_points": int(dense_pts.shape[0]),
                    }
                if dense_pts.shape[0] == 0:
                    skipped.append({"tooth_dir": str(tdir), "reason": "no_points_and_no_existing_dense_curve"})
                    continue

                H_shape_prior = generate_heatmap_from_points(
                    A.shape,
                    dense_pts,
                    spacing=spacing,
                    sigma_mm=cfg["sigma_mm"],
                    connect_points=True,
                    close_loop=cfg["curve_closed"],
                )
                save_volume(
                    str(tdir / f"{cfg['shape_prior_name']}.{fmt}"),
                    H_shape_prior.astype(np.float32),
                    affine=None,
                    spacing=spacing,
                )
                curve_fit_report = compute_curve_fit_report(
                    pts,
                    dense_pts,
                    spacing,
                    C_gt=None,
                    geometry_prior=geometry_prior,
                    fit_report=fit_report,
                )
                curve_fit_report["shape_prior"] = {
                    "path": str(tdir / f"{cfg['shape_prior_name']}.{fmt}"),
                    "sigma_mm": float(cfg["sigma_mm"]),
                    "max": float(np.max(H_shape_prior)) if H_shape_prior.size else 0.0,
                    "dense_source": dense_source,
                }
                geometry_prior["fit_summary"] = {
                    "fallback": bool(fit_report.get("fallback", False)),
                    "fallback_reason": fit_report.get("fallback_reason"),
                    "n_output_points": int(fit_report.get("n_output_points", dense_pts.shape[0])),
                    "constraints_passed": bool((curve_fit_report.get("constraints") or {}).get("passed", False)),
                    "manual_to_curve_mean_mm": (curve_fit_report.get("manual_to_curve") or {}).get("mean_mm"),
                }
                np.save(tdir / "shape_prior_dense_points.npy", dense_pts.astype(np.float32))
                _save_shape_prior_json(
                    tdir / "loss_shape_geometry_prior.json",
                    geometry_prior_to_jsonable(geometry_prior),
                )
                _save_shape_prior_json(tdir / "shape_prior_fit_report.json", curve_fit_report)
                generated += 1
            except Exception as exc:
                errors.append({"tooth_dir": str(tdir), "error": repr(exc)})
    return {
        "processed_dir": str(processed_dir),
        "generated_shape_prior_count": int(generated),
        "skipped_count": len(skipped),
        "error_count": len(errors),
        "skipped": skipped[:200],
        "errors": errors[:100],
    }


def inspect_geometry_split(
    processed_dir: Path,
    shape_prior_required: bool,
    *,
    geometry_prior_name: str = "geometry_prior.json",
    curve_report_name: str = "curve_fit_report.json",
    dense_points_name: str = "curve_dense_points.npy",
    require_c_gt: bool = True,
) -> Dict:
    audit = build_supervised_audit(str(processed_dir), "nii.gz")
    records = audit["used_labeled_teeth"]
    fallback_count = 0
    constraints_passed_count = 0
    geometry_prior_present = 0
    curve_report_present = 0
    dense_points_present = 0
    h_gt_present = 0
    c_gt_present = 0
    shape_prior_present = 0
    manual_means = []
    manual_p95s = []
    fallback_reasons = {}
    missing = []

    for rec in records:
        tdir = Path(rec["tooth_dir"])
        expected = {
            "geometry_prior": tdir / geometry_prior_name,
            "curve_fit_report": tdir / curve_report_name,
            "curve_dense_points": tdir / dense_points_name,
            "H_GT": tdir / "H_GT.nii.gz",
        }
        c_gt_path = tdir / "C_GT.nii.gz"
        if require_c_gt:
            expected["C_GT"] = c_gt_path
        if shape_prior_required:
            expected["H_SHAPE_PRIOR"] = tdir / "H_SHAPE_PRIOR.nii.gz"
        missing_names = [name for name, path in expected.items() if not path.exists()]
        if missing_names:
            missing.append({"tooth_dir": str(tdir), "missing": missing_names})

        geometry_prior_present += int(expected["geometry_prior"].exists())
        curve_report_present += int(expected["curve_fit_report"].exists())
        dense_points_present += int(expected["curve_dense_points"].exists())
        h_gt_present += int(expected["H_GT"].exists())
        c_gt_present += int(c_gt_path.exists())
        if shape_prior_required:
            shape_prior_present += int(expected["H_SHAPE_PRIOR"].exists())

        if not expected["curve_fit_report"].exists():
            continue
        report = read_json(expected["curve_fit_report"])
        fit = report.get("fit") or {}
        fallback = bool(fit.get("fallback", False))
        fallback_count += int(fallback)
        if fallback:
            reason = str(fit.get("fallback_reason") or "unknown")
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        constraints = report.get("constraints") or fit.get("constraints") or {}
        constraints_passed_count += int(bool(constraints.get("passed", False)))
        manual = report.get("manual_to_curve") or {}
        if manual.get("mean_mm") is not None:
            manual_means.append(float(manual["mean_mm"]))
        if manual.get("p95_mm") is not None:
            manual_p95s.append(float(manual["p95_mm"]))

    n = max(1, len(records))
    return {
        "processed_dir": str(processed_dir),
        "audit": audit,
        "labeled_tooth_count": len(records),
        "geometry_prior_present_count": geometry_prior_present,
        "curve_fit_report_present_count": curve_report_present,
        "curve_dense_points_present_count": dense_points_present,
        "h_gt_present_count": h_gt_present,
        "c_gt_present_count": c_gt_present,
        "shape_prior_required": bool(shape_prior_required),
        "shape_prior_present_count": shape_prior_present if shape_prior_required else None,
        "fallback_count": fallback_count,
        "fallback_rate": float(fallback_count / n),
        "fallback_reasons": fallback_reasons,
        "constraints_passed_count": constraints_passed_count,
        "constraints_passed_rate": float(constraints_passed_count / n),
        "manual_to_curve_mean_mm": float(sum(manual_means) / len(manual_means)) if manual_means else None,
        "manual_to_curve_p95_mm": sorted(manual_p95s)[int(0.95 * (len(manual_p95s) - 1))] if manual_p95s else None,
        "missing_artifacts": missing,
    }


def upgrade_geometry_labels(
    run_root: Path,
    base_cfg: Dict,
    manifest: Dict,
    shape_prior: bool,
    shape_prior_sigma_mm: float,
    force: bool,
) -> Dict:
    audit_path = run_root / "input" / "geometry_upgrade_audit.json"
    if audit_path.exists() and not force:
        return read_json(audit_path)

    train_root = Path(manifest["processed_train16"])
    holdout_root = Path(manifest["processed_holdout3"])
    upgraded = {}
    for split_name, root in (("train16", train_root), ("holdout3", holdout_root)):
        cfg = build_upgrade_cfg(base_cfg, root, shape_prior, shape_prior_sigma_mm)
        upgraded[split_name] = int(upgrade_processed_dir(cfg))

    train_audit = inspect_geometry_split(train_root, shape_prior_required=shape_prior)
    holdout_audit = inspect_geometry_split(holdout_root, shape_prior_required=shape_prior)
    audit = {
        "created_at": now_iso(),
        "shape_prior_written": bool(shape_prior),
        "shape_prior_sigma_mm": float(shape_prior_sigma_mm),
        "upgraded_counts": upgraded,
        "train16": train_audit,
        "holdout3": holdout_audit,
        "summary": {
            "train_labeled_tooth_count": train_audit["labeled_tooth_count"],
            "holdout_labeled_tooth_count": holdout_audit["labeled_tooth_count"],
            "constraints_passed_rate": {
                "train16": train_audit["constraints_passed_rate"],
                "holdout3": holdout_audit["constraints_passed_rate"],
            },
            "fallback_rate": {
                "train16": train_audit["fallback_rate"],
                "holdout3": holdout_audit["fallback_rate"],
            },
            "manual_to_curve_mean_mm": {
                "train16": train_audit["manual_to_curve_mean_mm"],
                "holdout3": holdout_audit["manual_to_curve_mean_mm"],
            },
            "manual_to_curve_p95_mm": {
                "train16": train_audit["manual_to_curve_p95_mm"],
                "holdout3": holdout_audit["manual_to_curve_p95_mm"],
            },
        },
    }
    write_json(audit_path, audit)
    return audit


def generate_loss_only_shape_prior(
    run_root: Path,
    base_cfg: Dict,
    manifest: Dict,
    shape_prior_sigma_mm: float,
    force: bool,
) -> Dict:
    audit_path = run_root / "input" / "geometry_upgrade_audit.json"
    preservation_path = run_root / "input" / "gt_preservation_audit.json"
    if audit_path.exists() and preservation_path.exists() and not force:
        return read_json(audit_path)

    train_root = Path(manifest["processed_train16"])
    holdout_root = Path(manifest["processed_holdout3"])
    fmt = base_cfg.get("data", {}).get("processed_format", "nii.gz")
    before = {
        "train16": collect_gt_artifact_hashes(train_root, fmt=fmt),
        "holdout3": collect_gt_artifact_hashes(holdout_root, fmt=fmt),
    }
    generated = {
        "train16": _generate_shape_prior_for_split(train_root, base_cfg, shape_prior_sigma_mm),
        "holdout3": _generate_shape_prior_for_split(holdout_root, base_cfg, shape_prior_sigma_mm),
    }
    after = {
        "train16": collect_gt_artifact_hashes(train_root, fmt=fmt),
        "holdout3": collect_gt_artifact_hashes(holdout_root, fmt=fmt),
    }
    preservation = {
        "created_at": now_iso(),
        "mode": "shape_prior_loss_only",
        "gt_modified": False,
        "train16": compare_gt_artifact_hashes(before["train16"], after["train16"]),
        "holdout3": compare_gt_artifact_hashes(before["holdout3"], after["holdout3"]),
    }
    preservation["gt_preserved"] = bool(
        preservation["train16"]["gt_preserved"] and preservation["holdout3"]["gt_preserved"]
    )
    preservation["mismatch_count"] = int(
        preservation["train16"]["mismatch_count"] + preservation["holdout3"]["mismatch_count"]
    )
    write_json(preservation_path, preservation)
    if not preservation["gt_preserved"]:
        raise RuntimeError(f"loss-only shape-prior generation changed GT artifacts: {preservation_path}")

    train_audit = inspect_geometry_split(
        train_root,
        shape_prior_required=True,
        geometry_prior_name="loss_shape_geometry_prior.json",
        curve_report_name="shape_prior_fit_report.json",
        dense_points_name="shape_prior_dense_points.npy",
        require_c_gt=False,
    )
    holdout_audit = inspect_geometry_split(
        holdout_root,
        shape_prior_required=True,
        geometry_prior_name="loss_shape_geometry_prior.json",
        curve_report_name="shape_prior_fit_report.json",
        dense_points_name="shape_prior_dense_points.npy",
        require_c_gt=False,
    )
    audit = {
        "created_at": now_iso(),
        "mode": "shape_prior_loss_only",
        "gt_modified": False,
        "shape_prior_written": True,
        "shape_prior_sigma_mm": float(shape_prior_sigma_mm),
        "generated_counts": generated,
        "gt_preservation_audit": str(preservation_path),
        "train16": train_audit,
        "holdout3": holdout_audit,
        "summary": {
            "train_labeled_tooth_count": train_audit["labeled_tooth_count"],
            "holdout_labeled_tooth_count": holdout_audit["labeled_tooth_count"],
            "constraints_passed_rate": {
                "train16": train_audit["constraints_passed_rate"],
                "holdout3": holdout_audit["constraints_passed_rate"],
            },
            "fallback_rate": {
                "train16": train_audit["fallback_rate"],
                "holdout3": holdout_audit["fallback_rate"],
            },
            "manual_to_curve_mean_mm": {
                "train16": train_audit["manual_to_curve_mean_mm"],
                "holdout3": holdout_audit["manual_to_curve_mean_mm"],
            },
            "manual_to_curve_p95_mm": {
                "train16": train_audit["manual_to_curve_p95_mm"],
                "holdout3": holdout_audit["manual_to_curve_p95_mm"],
            },
            "gt_preserved": preservation["gt_preserved"],
            "gt_mismatch_count": preservation["mismatch_count"],
        },
    }
    write_json(audit_path, audit)
    return audit


def _arg_or_cfg(args: argparse.Namespace, name: str, cfg: Dict, default):
    value = getattr(args, name)
    if value is not None:
        return value
    return cfg.get(name, default)


def render_train_config(
    base_cfg: Dict,
    manifest: Dict,
    run_root: Path,
    args: argparse.Namespace,
    shape_prior: bool,
    shape_prior_sigma_mm: float,
) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("project", {})["device"] = args.device

    data = cfg.setdefault("data", {})
    data["processed_dir"] = manifest["processed_train16"]
    data["holdout_processed_dir"] = manifest["processed_holdout3"]
    data["output_dir"] = str(run_root / "outputs")
    data["processed_format"] = "nii.gz"
    data["use_tooth_mask_channel"] = True
    data["shape_prior_name"] = "H_SHAPE_PRIOR"

    model = cfg.setdefault("model", {})
    model["in_channels"] = 2
    model["out_channels"] = 1
    model["base_channels"] = int(args.base_channels if args.base_channels is not None else model.get("base_channels", 16))
    model["depth"] = int(model.get("depth", 4))
    model.setdefault("num_res_units", 2)
    model.setdefault("norm", "batch")

    preprocess = cfg.setdefault("preprocess", {})
    geometry = preprocess.setdefault("geometry_prior", {})
    geometry["write_shape_prior_heatmap"] = bool(shape_prior)
    geometry["shape_prior_sigma_mm"] = float(shape_prior_sigma_mm)
    geometry["shape_prior_name"] = "H_SHAPE_PRIOR"
    preprocess["shape_prior_sigma_mm"] = float(shape_prior_sigma_mm)
    preprocess["shape_prior_name"] = "H_SHAPE_PRIOR"

    train = cfg.setdefault("train", {})
    train["loss"] = "dice_focal_skeleton"
    train["pretrained_ckpt"] = manifest["pretrained_copy"]
    train["pretrained_strict"] = False
    train["max_epochs"] = int(_arg_or_cfg(args, "max_epochs", train, 50))
    train["epochs"] = int(train["max_epochs"])
    train["min_epochs"] = int(_arg_or_cfg(args, "min_epochs", train, min(8, train["max_epochs"])))
    train["early_stopping_patience"] = int(_arg_or_cfg(args, "patience", train, 8))
    train["early_stopping_min_delta"] = float(_arg_or_cfg(args, "early_stopping_min_delta", train, 0.0005))
    train["lr"] = float(_arg_or_cfg(args, "lr", train, 0.001))
    train["batch_size"] = int(_arg_or_cfg(args, "batch_size", train, 1))
    train["num_workers"] = int(_arg_or_cfg(args, "num_workers", train, 4))
    train["holdout_batch_size"] = int(_arg_or_cfg(args, "holdout_batch_size", train, train["batch_size"]))
    train["holdout_num_workers"] = int(_arg_or_cfg(args, "holdout_num_workers", train, 2))
    train["cache_rate"] = float(_arg_or_cfg(args, "cache_rate", train, 0.0))
    train["holdout_cache_rate"] = float(_arg_or_cfg(args, "holdout_cache_rate", train, 0.0))
    train["weighted_sampler"] = True
    train.setdefault("hard_cases", ["041-31/41/42"])
    train["hard_case_weight"] = float(_arg_or_cfg(args, "hard_case_weight", train, 4.0))
    train["checkpoint_top_k"] = int(_arg_or_cfg(args, "checkpoint_top_k", train, 5))
    train["checkpoint_every_n_epochs"] = int(_arg_or_cfg(args, "checkpoint_every_n_epochs", train, 5))
    train["loss_lambda_dice"] = float(_arg_or_cfg(args, "loss_lambda_dice", train, 1.0))
    train["loss_lambda_bce"] = float(_arg_or_cfg(args, "loss_lambda_bce", train, 1.0))
    train["loss_lambda_focal"] = float(_arg_or_cfg(args, "loss_lambda_focal", train, 0.25))
    train["focal_gamma"] = float(_arg_or_cfg(args, "focal_gamma", train, 2.0))
    train["focal_alpha"] = float(_arg_or_cfg(args, "focal_alpha", train, 0.75))
    train["loss_lambda_skeleton"] = float(_arg_or_cfg(args, "loss_lambda_skeleton", train, 0.2))
    train["skeleton_target_threshold"] = float(_arg_or_cfg(args, "skeleton_target_threshold", train, 0.95))
    train["skeleton_pos_weight"] = float(_arg_or_cfg(args, "skeleton_pos_weight", train, 8.0))
    train["surface_neighborhood_weight"] = float(_arg_or_cfg(args, "surface_neighborhood_weight", train, 0.0))
    train["surface_neighborhood_radius_vox"] = int(_arg_or_cfg(args, "surface_neighborhood_radius_vox", train, 2))
    train["use_shape_prior_channel_or_loss"] = bool(shape_prior)
    train["lambda_shape_prior"] = float(getattr(args, "lambda_shape_prior", 0.0) if shape_prior else 0.0)
    train["shape_prior_sigma_mm"] = float(shape_prior_sigma_mm)

    composite = train.setdefault("composite_score", {})
    if args.score_start_epoch is not None:
        composite["start_epoch"] = int(args.score_start_epoch)
    else:
        composite.setdefault("start_epoch", 10)
    composite.setdefault(
        "weights",
        {
            "loss": 0.35,
            "sym_p95": 0.35,
            "no_curve_count": 0.08,
            "bad_rate": 0.08,
            "vox03_empty": 0.04,
            "cc_count": 0.04,
            "wrap_miss": 0.06,
            "hard_case_viewer": 0.12,
        },
    )
    composite.setdefault(
        "targets",
        {
            "loss": 0.5,
            "sym_p95_mm": 5.0,
            "bad_sym_p95_mm": 5.0,
            "hard_sym_p95_mm": 5.0,
            "hard_bad_sym_p95_mm": 5.0,
            "hard_case_count": 12,
            "vox03": 100.0,
            "cc_count": 4.0,
        },
    )

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
    viz["enable_3d"] = False
    viz["max_cases"] = 3
    viz["max_teeth_per_case"] = 64
    viz["show_pseudo_gt_skeleton"] = True
    viz["pseudo_gt_skeleton_from_interp"] = False
    viz["pseudo_gt_skeleton_from_heatmap_peak"] = True
    viz["use_error_colormap_for_gt_points"] = True
    return cfg


def write_dataset_audits(run_root: Path, cfg: Dict) -> Dict:
    out = {}
    for split_name, key, out_name in (
        ("train16", "processed_dir", "audit_train16"),
        ("holdout3", "holdout_processed_dir", "audit_holdout3"),
    ):
        audit = build_supervised_audit(cfg["data"][key], cfg["data"].get("processed_format", "nii.gz"))
        paths = write_supervised_audit(audit, str(run_root / "input" / out_name))
        out[split_name] = {"audit": audit, "paths": paths}
    return out


def train_artifact_audit(run_root: Path, cfg_path: Path, manifest: Dict) -> Dict:
    outputs_train = run_root / "outputs" / "train"
    train_manifest_path = outputs_train / "train_manifest.json"
    train_manifest = read_json(train_manifest_path) if train_manifest_path.exists() else {}
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    processed_train = Path(cfg.get("data", {}).get("processed_dir", ""))
    processed_holdout = Path(cfg.get("data", {}).get("holdout_processed_dir", ""))

    def under_run_root(path: Path) -> bool:
        try:
            path.resolve().relative_to(run_root.resolve())
            return True
        except Exception:
            return False

    return {
        "best_pt_exists": (outputs_train / "checkpoints" / "best.pt").exists(),
        "last_pt_exists": (outputs_train / "checkpoints" / "last.pt").exists(),
        "metrics_csv_exists": (outputs_train / "metrics.csv").exists(),
        "train_manifest_exists": train_manifest_path.exists(),
        "train_manifest_path": str(train_manifest_path),
        "train_total_tooth_count": train_manifest.get("dataset", {}).get("train", {}).get("total_tooth_dirs"),
        "holdout_total_tooth_count": train_manifest.get("dataset", {}).get("holdout", {}).get("total_tooth_dirs"),
        "train_tooth_count_ok": train_manifest.get("dataset", {}).get("train", {}).get("total_tooth_dirs")
        == split16.EXPECTED_TRAIN_TOOTH_DIRS,
        "holdout_tooth_count_ok": train_manifest.get("dataset", {}).get("holdout", {}).get("total_tooth_dirs")
        == split16.EXPECTED_HOLDOUT_TOOTH_DIRS,
        "config_processed_dir_under_run_root": under_run_root(processed_train),
        "config_holdout_processed_dir_under_run_root": under_run_root(processed_holdout),
        "config_processed_dir": str(processed_train),
        "config_holdout_processed_dir": str(processed_holdout),
        "expected_train_tooth_count": manifest.get("train_tooth_dir_count"),
        "expected_holdout_tooth_count": manifest.get("holdout_tooth_dir_count"),
    }


def create_threshold_config(base_cfg: Dict, manifest: Dict, threshold: float, out_dir: Path, ckpt_path: Path) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["data"]["processed_dir"] = manifest["processed_holdout3"]
    cfg["data"]["holdout_processed_dir"] = manifest["processed_holdout3"]
    cfg["data"]["output_dir"] = str(out_dir)
    cfg.setdefault("infer", {})["ckpt_path"] = str(ckpt_path)
    cfg["infer"]["threshold_theta"] = float(threshold)
    cfg.setdefault("holdout_eval", {})["threshold_theta"] = float(threshold)
    return cfg


def load_csv_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def holdout_case_count(rows: List[Dict]) -> int:
    return len({row.get("case_id") for row in rows if row.get("case_id")})


def goal_checks(summary: Dict, per_tooth_rows: List[Dict], failed_or_bad_count: int) -> Dict:
    mean = summary.get("mean_dist_mm", summary.get("mean"))
    mean = float(mean) if mean is not None else float("inf")
    p95 = goal19.summary_p95(summary)
    sr1 = goal19.summary_sr1(summary)
    missing = int(summary.get("missing_prediction_count", 0) or 0)
    no_curve = int(summary.get("no_curve_count", 0) or 0)
    failed = int(summary.get("failed_tooth_count", 0) or 0)
    tooth_count = int(summary.get("tooth_count", 0) or 0)
    case_count = holdout_case_count(per_tooth_rows)
    criteria = {
        "p95_dist_mm_lte_1_467": p95 <= NON_INFERIORITY_GOAL["p95_dist_mm_lte"],
        "sr_at_1mm_gte_0_836": sr1 >= NON_INFERIORITY_GOAL["sr@1.0mm_gte"],
        "failed_or_bad_teeth_count_lte_3": int(failed_or_bad_count)
        <= NON_INFERIORITY_GOAL["failed_or_bad_teeth_count_lte"],
        "missing_prediction_count_is_0": missing == NON_INFERIORITY_GOAL["missing_prediction_count_eq"],
        "no_curve_count_is_0": no_curve == NON_INFERIORITY_GOAL["no_curve_count_eq"],
        "holdout_tooth_count_is_88": tooth_count == NON_INFERIORITY_GOAL["holdout_tooth_count_eq"],
        "holdout_case_count_is_3": case_count == 3,
    }
    passed = all(criteria.values())
    return {
        "status": "PASS_NON_INFERIOR" if passed else "FAIL",
        "passed_non_inferiority_goal": bool(passed),
        "criteria": criteria,
        "holdout_case_count": case_count,
        "holdout_tooth_count": tooth_count,
        "missing_prediction_count": missing,
        "no_curve_count": no_curve,
        "failed_tooth_count": failed,
        "failed_or_bad_teeth_count": int(failed_or_bad_count),
        "mean_dist_mm": mean,
        "p95_dist_mm": p95,
        "sr@1.0mm": sr1,
    }


def run_threshold_sweep(
    repo_dir: Path,
    python_exe: str,
    run_root: Path,
    cfg: Dict,
    manifest: Dict,
    force_eval: bool,
) -> Dict:
    ckpt_path = run_root / "outputs" / "train" / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best checkpoint missing before threshold sweep: {ckpt_path}")
    sweep_root = run_root / "threshold_sweep_best"
    sweep_root.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for threshold in THRESHOLDS:
        label = f"theta_{threshold:.2f}"
        out_dir = sweep_root / label
        if force_eval and out_dir.exists():
            shutil.rmtree(out_dir)
        cfg_path = run_root / "config" / f"best_{label}.yaml"
        threshold_cfg = create_threshold_config(cfg, manifest, threshold, out_dir, ckpt_path)
        write_yaml(cfg_path, threshold_cfg)
        if not (out_dir / "eval" / "metrics_summary.json").exists():
            run_cmd([python_exe, "-m", "src.infer", "--config", str(cfg_path)], repo_dir, run_root / "logs" / f"infer_best_{label}.log")
            run_cmd([python_exe, "-m", "src.eval", "--config", str(cfg_path)], repo_dir, run_root / "logs" / f"eval_best_{label}.log")
        eval_dir = out_dir / "eval"
        summary = goal19.load_summary(eval_dir)
        per_tooth = eval_dir / "metrics_per_tooth.csv"
        per_tooth_rows = load_csv_rows(per_tooth)
        bad_rows = goal19.load_bad_rows(per_tooth)
        failed_csv = eval_dir / "failed_or_bad_teeth.csv"
        goal19.write_failed_rows(failed_csv, bad_rows)
        checks = goal_checks(summary, per_tooth_rows, len(bad_rows))
        row = {
            "checkpoint_label": "best",
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
                0 if checks["passed_non_inferiority_goal"] else 1,
                0 if checks["missing_prediction_count"] == 0 else 1,
                0 if checks["no_curve_count"] == 0 else 1,
                checks["failed_or_bad_teeth_count"],
                checks["p95_dist_mm"],
                -checks["sr@1.0mm"],
                threshold,
                row,
                threshold_cfg,
                checks,
            )
        )

    csv_paths = [sweep_root / "threshold_sweep_best.csv", run_root / "summary" / "threshold_sweep_best.csv"]
    for csv_path in csv_paths:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    candidates.sort(key=lambda item: item[:7])
    best = candidates[0]
    return {
        "sweep_root": str(sweep_root),
        "sweep_csv": str(csv_paths[0]),
        "summary_sweep_csv": str(csv_paths[1]),
        "rows": rows,
        "best_row": best[7],
        "best_config": best[8],
        "best_checks": best[9],
    }


def copy_final_best(run_root: Path, best_row: Dict, best_cfg: Dict) -> Dict:
    final_root = run_root / "final_best"
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)
    best_out = Path(best_row["output_dir"])
    if (best_out / "infer").exists():
        shutil.copytree(best_out / "infer", final_root / "infer")
    if (best_out / "eval").exists():
        shutil.copytree(best_out / "eval", final_root / "eval")
    final_cfg = json.loads(json.dumps(best_cfg))
    final_cfg["data"]["output_dir"] = str(final_root)
    final_cfg["infer"]["ckpt_path"] = str(run_root / "outputs" / "train" / "checkpoints" / "best.pt")
    cfg_path = run_root / "config" / "final_best.yaml"
    write_yaml(cfg_path, final_cfg)
    return {
        "final_root": str(final_root),
        "final_config": str(cfg_path),
        "metrics_summary": str(final_root / "eval" / "metrics_summary.json"),
        "metrics_per_tooth": str(final_root / "eval" / "metrics_per_tooth.csv"),
        "failed_or_bad_teeth": str(final_root / "eval" / "failed_or_bad_teeth.csv"),
    }


def render_report_md(report: Dict) -> str:
    best = report["selected_validation"]
    goal = report["non_inferiority_goal"]
    criteria = best["criteria"]
    gt_preservation = report.get("gt_preservation_audit") or {}
    lines = [
        f"# {report['experiment_label']}",
        "",
        f"- Run root: `{report['run_root']}`",
        f"- Selected threshold: `{report['selected_threshold']}`",
        f"- Status: `{best['status']}`",
        "",
        "| Metric | Value | Goal | Pass |",
        "| --- | ---: | ---: | --- |",
        f"| mean_dist_mm | {best['mean_dist_mm']:.6f} | baseline 0.628 | - |",
        f"| p95_dist_mm | {best['p95_dist_mm']:.6f} | <= {goal['p95_dist_mm_lte']} | {criteria['p95_dist_mm_lte_1_467']} |",
        f"| sr@1.0mm | {best['sr@1.0mm']:.6f} | >= {goal['sr@1.0mm_gte']} | {criteria['sr_at_1mm_gte_0_836']} |",
        f"| failed/bad teeth | {best['failed_or_bad_teeth_count']} | <= {goal['failed_or_bad_teeth_count_lte']} | {criteria['failed_or_bad_teeth_count_lte_3']} |",
        f"| missing predictions | {best['missing_prediction_count']} | 0 | {criteria['missing_prediction_count_is_0']} |",
        f"| no-curve teeth | {best['no_curve_count']} | 0 | {criteria['no_curve_count_is_0']} |",
        f"| holdout teeth | {best['holdout_tooth_count']} | 88 | {criteria['holdout_tooth_count_is_88']} |",
    ]
    if gt_preservation:
        lines.append(
            f"| GT preserved | {gt_preservation.get('gt_preserved')} | True | "
            f"{gt_preservation.get('mismatch_count') == 0} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Train config: `{report['train_config']}`",
            f"- Sweep CSV: `{report['threshold_sweep']['sweep_csv']}`",
            f"- Final metrics: `{report['final']['metrics_summary']}`",
            f"- Failed/bad teeth: `{report['final']['failed_or_bad_teeth']}`",
        ]
    )
    if report.get("gt_preservation_audit_path"):
        lines.append(f"- GT preservation audit: `{report['gt_preservation_audit_path']}`")
    return "\n".join(lines) + "\n"


def write_experiment_report(
    run_root: Path,
    experiment_key: str,
    experiment_label: str,
    manifest: Dict,
    preflight: Dict,
    geometry_audit: Dict,
    train_config: Path,
    dataset_audits: Dict,
    train_audit: Dict,
    sweep: Dict,
    final: Dict,
) -> Dict:
    gt_preservation_path = run_root / "input" / "gt_preservation_audit.json"
    report = {
        "created_at": now_iso(),
        "experiment_key": experiment_key,
        "experiment_label": experiment_label,
        "run_root": str(run_root),
        "source_manifest": str(run_root / "input" / "source_manifest.json"),
        "case_manifest": str(run_root / "input" / "case_manifest.csv"),
        "preflight": preflight,
        "geometry_upgrade_audit_path": str(run_root / "input" / "geometry_upgrade_audit.json"),
        "geometry_upgrade_audit": geometry_audit,
        "gt_preservation_audit_path": str(gt_preservation_path) if gt_preservation_path.exists() else None,
        "gt_preservation_audit": read_json(gt_preservation_path) if gt_preservation_path.exists() else None,
        "train_config": str(train_config),
        "dataset_audits": dataset_audits,
        "train_artifact_audit": train_audit,
        "threshold_sweep": sweep,
        "final": final,
        "selected_threshold": sweep["best_row"]["threshold"],
        "selected_threshold_row": sweep["best_row"],
        "selected_validation": sweep["best_checks"],
        "baseline_stage1_only": BASELINE_STAGE1_ONLY,
        "non_inferiority_goal": NON_INFERIORITY_GOAL,
        "recommendation": (
            "candidate_for_mainline" if sweep["best_checks"]["passed_non_inferiority_goal"] else "do_not_merge_by_default"
        ),
        "input_counts": {
            "train_tooth_dir_count": manifest["train_tooth_dir_count"],
            "holdout_tooth_dir_count": manifest["holdout_tooth_dir_count"],
            "train_cases": manifest["train_cases"],
            "holdout_cases": manifest["holdout_cases"],
        },
    }
    json_path = run_root / "summary" / f"{experiment_key}_report.json"
    md_path = run_root / "summary" / f"{experiment_key}_report.md"
    write_json(json_path, report)
    md_path.write_text(render_report_md(report), encoding="utf-8")
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    write_json(json_path, report)
    return report


def add_common_args(parser: argparse.ArgumentParser, default_prefix: str) -> None:
    parser.add_argument("--repo-dir", default=str(REPO_ROOT))
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--unsup-ckpt", default=str(DEFAULT_UNSUP_CKPT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-upgrade", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--score-start-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--holdout-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--holdout-num-workers", type=int, default=None)
    parser.add_argument("--cache-rate", type=float, default=None)
    parser.add_argument("--holdout-cache-rate", type=float, default=None)
    parser.add_argument("--hard-case-weight", type=float, default=None)
    parser.add_argument("--checkpoint-top-k", type=int, default=None)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=None)
    parser.add_argument("--loss-lambda-dice", type=float, default=None)
    parser.add_argument("--loss-lambda-bce", type=float, default=None)
    parser.add_argument("--loss-lambda-focal", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--loss-lambda-skeleton", type=float, default=None)
    parser.add_argument("--skeleton-target-threshold", type=float, default=None)
    parser.add_argument("--skeleton-pos-weight", type=float, default=None)
    parser.add_argument("--surface-neighborhood-weight", type=float, default=None)
    parser.add_argument("--surface-neighborhood-radius-vox", type=int, default=None)
    parser.set_defaults(default_prefix=default_prefix)


def prepare_run_root(args: argparse.Namespace, default_prefix: str) -> Path:
    run_root = (
        Path(args.run_root).resolve()
        if args.run_root
        else (DEFAULT_RUN_PARENT / f"{default_prefix}_{timestamp()}").resolve()
    )
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"run root exists; use --resume to continue: {run_root}")
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        (run_root / "config").mkdir()
        (run_root / "logs").mkdir()
        (run_root / "summary").mkdir()
    return run_root


def run_experiment(
    args: argparse.Namespace,
    *,
    experiment_key: str,
    experiment_label: str,
    default_prefix: str,
    shape_prior: bool,
    loss_only_geometry: bool = False,
) -> Dict:
    repo_dir = Path(args.repo_dir).resolve()
    run_root = prepare_run_root(args, default_prefix)
    base_cfg = load_base_config(Path(args.base_config).resolve())
    shape_prior_sigma_mm = float(args.shape_prior_sigma_mm)

    manifest = setup_inputs(run_root, Path(args.unsup_ckpt).resolve(), args.resume)
    preflight = write_preflight(repo_dir, run_root, manifest)
    if loss_only_geometry:
        if not shape_prior:
            raise ValueError("loss_only_geometry requires shape_prior=True")
        geometry_audit = generate_loss_only_shape_prior(
            run_root,
            base_cfg,
            manifest,
            shape_prior_sigma_mm=shape_prior_sigma_mm,
            force=bool(args.force_upgrade),
        )
    else:
        geometry_audit = upgrade_geometry_labels(
            run_root,
            base_cfg,
            manifest,
            shape_prior=shape_prior,
            shape_prior_sigma_mm=shape_prior_sigma_mm,
            force=bool(args.force_upgrade),
        )

    cfg = render_train_config(
        base_cfg,
        manifest,
        run_root,
        args,
        shape_prior=shape_prior,
        shape_prior_sigma_mm=shape_prior_sigma_mm,
    )
    train_cfg_path = run_root / "config" / "train.yaml"
    write_yaml(train_cfg_path, cfg)
    dataset_audits = write_dataset_audits(run_root, cfg)

    manifest["geometry_upgrade_audit"] = str(run_root / "input" / "geometry_upgrade_audit.json")
    manifest["train_config"] = str(train_cfg_path)
    manifest["shape_prior_enabled"] = bool(shape_prior)
    manifest["shape_prior_sigma_mm"] = float(shape_prior_sigma_mm)
    manifest["geometry_mode"] = "shape_prior_loss_only" if loss_only_geometry else "gt_upgrade"
    manifest["gt_modified"] = not bool(loss_only_geometry)
    if loss_only_geometry:
        manifest["gt_preservation_audit"] = str(run_root / "input" / "gt_preservation_audit.json")
    write_json(run_root / "input" / "source_manifest.json", manifest)

    if args.setup_only:
        print(f"[OK] setup complete: {run_root}")
        print(f"[CONFIG] {train_cfg_path}")
        print(f"[GEOMETRY_AUDIT] {run_root / 'input' / 'geometry_upgrade_audit.json'}")
        return {"run_root": str(run_root), "setup_only": True}

    best_ckpt = run_root / "outputs" / "train" / "checkpoints" / "best.pt"
    if not args.skip_train and not best_ckpt.exists():
        run_cmd(
            [args.python_exe, "-m", "src.train_loss_earlystop", "--config", str(train_cfg_path)],
            repo_dir,
            run_root / "logs" / "train.log",
        )
    if args.train_only:
        print(f"[OK] training complete: {best_ckpt}")
        return {"run_root": str(run_root), "train_only": True, "best_checkpoint": str(best_ckpt)}

    train_audit = train_artifact_audit(run_root, train_cfg_path, manifest)
    sweep = run_threshold_sweep(repo_dir, args.python_exe, run_root, cfg, manifest, force_eval=bool(args.force_eval))
    final = copy_final_best(run_root, sweep["best_row"], sweep["best_config"])
    report = write_experiment_report(
        run_root,
        experiment_key,
        experiment_label,
        manifest,
        preflight,
        geometry_audit,
        train_cfg_path,
        dataset_audits,
        train_audit,
        sweep,
        final,
    )
    selected = report["selected_validation"]
    print(f"[RUN_ROOT] {run_root}")
    print(f"[STATUS] {selected['status']}")
    print(f"[THRESHOLD] {report['selected_threshold']}")
    print(f"[MEAN] {selected['mean_dist_mm']}")
    print(f"[P95] {selected['p95_dist_mm']}")
    print(f"[SR@1.0] {selected['sr@1.0mm']}")
    print(f"[FAILED_OR_BAD] {selected['failed_or_bad_teeth_count']}")
    print(f"[REPORT] {report['report_json']}")
    return report


def _load_report_for_root(root: Path, key: str) -> Optional[Dict]:
    path = root / "summary" / f"{key}_report.json"
    if path.exists():
        return read_json(path)
    return None


def _report_row(label: str, report: Dict) -> Dict:
    selected = report["selected_validation"]
    return {
        "experiment": label,
        "threshold": report["selected_threshold"],
        "mean_dist_mm": selected["mean_dist_mm"],
        "p95_dist_mm": selected["p95_dist_mm"],
        "sr@1.0mm": selected["sr@1.0mm"],
        "failed_or_bad_teeth_count": selected["failed_or_bad_teeth_count"],
        "status": selected["status"],
        "run_root": report["run_root"],
    }


def _is_better_or_equal(row_a: Dict, row_b: Dict) -> bool:
    return (
        row_a["failed_or_bad_teeth_count"],
        row_a["p95_dist_mm"],
        -row_a["sr@1.0mm"],
    ) <= (
        row_b["failed_or_bad_teeth_count"],
        row_b["p95_dist_mm"],
        -row_b["sr@1.0mm"],
    )


def write_comparison(exp1_root: Path, exp2_root: Path, out_summary_dir: Path) -> Optional[Dict]:
    exp1 = _load_report_for_root(exp1_root, "geometry_prior_preprocess")
    exp2 = _load_report_for_root(exp2_root, "geometry_prior_loss")
    if exp1 is None or exp2 is None:
        return None
    row1 = _report_row("exp1 preprocess prior", exp1)
    row2 = _report_row("exp2 shape-prior loss", exp2)
    exp1_pass = exp1["selected_validation"]["passed_non_inferiority_goal"]
    exp2_pass = exp2["selected_validation"]["passed_non_inferiority_goal"]
    exp2_better = _is_better_or_equal(row2, row1)
    if exp1_pass and exp2_pass and exp2_better:
        conclusion = "adopt_exp2_keep_exp1_fallback"
    elif exp1_pass and not (exp2_pass and exp2_better):
        conclusion = "adopt_exp1_no_loss"
    elif not exp1_pass:
        conclusion = "do_not_merge_geometry_prior_by_default"
    else:
        conclusion = "preprocess_only"
    comparison = {
        "created_at": now_iso(),
        "baseline": BASELINE_STAGE1_ONLY,
        "exp1_report": exp1.get("report_json"),
        "exp2_report": exp2.get("report_json"),
        "rows": [BASELINE_STAGE1_ONLY, row1, row2],
        "exp1_passed": bool(exp1_pass),
        "exp2_passed": bool(exp2_pass),
        "exp2_better_or_equal_to_exp1": bool(exp2_better),
        "conclusion": conclusion,
        "conclusion_rules": [
            "exp1 pass and exp2 not clearly better: adopt exp1 without loss",
            "exp1 pass and exp2 pass with better metrics: adopt exp2, keep exp1 fallback",
            "exp1 fail: do not merge geometry prior by default",
            "exp1 pass and exp2 fail: preprocess only",
        ],
    }
    out_summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_summary_dir / "geometry_prior_exp1_exp2_comparison.json"
    md_path = out_summary_dir / "geometry_prior_exp1_exp2_comparison.md"
    write_json(json_path, comparison)
    md = [
        "# Geometry Prior Exp1/Exp2 Comparison",
        "",
        "| Experiment | threshold | mean_dist_mm | p95_dist_mm | sr@1.0mm | failed/bad | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in comparison["rows"]:
        md.append(
            f"| {row['experiment']} | {row['threshold']} | {row['mean_dist_mm']:.6f} | "
            f"{row['p95_dist_mm']:.6f} | {row['sr@1.0mm']:.6f} | "
            f"{row['failed_or_bad_teeth_count']} | {row['status']} |"
        )
    md.extend(["", f"Conclusion: `{conclusion}`", ""])
    md_path.write_text("\n".join(md), encoding="utf-8")
    comparison["comparison_json"] = str(json_path)
    comparison["comparison_md"] = str(md_path)
    write_json(json_path, comparison)
    return comparison


def find_latest_exp1(parent: Path) -> Optional[Path]:
    runs = sorted(parent.glob("stage1_geometry_prior_preprocess_*"))
    for root in reversed(runs):
        if (root / "summary" / "geometry_prior_preprocess_report.json").exists():
            return root
    return None
