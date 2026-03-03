#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import yaml


def parse_max_cases(raw: str):
    if raw in (None, "", "null", "None"):
        return None
    try:
        v = int(raw)
    except Exception:
        return None
    return v if v > 0 else None


def run_cmd(cmd: Sequence[str], cwd: Path) -> None:
    print("[CMD]", " ".join(str(x) for x in cmd))
    proc = subprocess.run(list(cmd), cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (rc={proc.returncode}): {' '.join(str(x) for x in cmd)}")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def run_usage_check(
    python_exe: str,
    repo_dir: Path,
    manual_root: Path,
    tf_root: Path,
    processed_root: Path,
    report_csv: Path,
    report_json: Path,
    max_cases,
) -> Dict:
    cmd = [
        python_exe,
        "scripts/check_manual_points_usage.py",
        "--manual-root",
        str(manual_root),
        "--toothfairy-root",
        str(tf_root),
        "--processed-root",
        str(processed_root),
        "--output-csv",
        str(report_csv),
        "--output-json",
        str(report_json),
    ]
    if max_cases is not None:
        cmd += ["--max-cases", str(max_cases)]
    run_cmd(cmd, cwd=repo_dir)
    return load_json(report_json)


def collect_failed_case_ids(report_payload: Dict) -> List[str]:
    failed = []
    for rec in report_payload.get("cases", []):
        status = str(rec.get("status", ""))
        case_id = str(rec.get("case_id", "")).strip()
        if case_id and not status.startswith("PASS"):
            failed.append(case_id)
    return sorted(set(failed))


def run_manual_resume(
    python_exe: str,
    repo_dir: Path,
    manual_root: Path,
    tf_root: Path,
    run_root: Path,
    preprocess_config: str,
    case_prefix: str,
    max_cases,
    rerun_on_source_update: bool = True,
    force_cases: List[str] = None,
) -> None:
    cmd = [
        python_exe,
        "scripts/run_manual_preprocess_resume.py",
        "--manual-root",
        str(manual_root),
        "--toothfairy-root",
        str(tf_root),
        "--run-root",
        str(run_root),
        "--base-config",
        preprocess_config,
        "--repo-dir",
        str(repo_dir),
        "--case-prefix",
        case_prefix,
    ]
    if rerun_on_source_update:
        cmd.append("--rerun-on-source-update")
    if max_cases is not None:
        cmd += ["--max-cases", str(max_cases)]
    if force_cases:
        cmd += ["--force-cases", ",".join(sorted(set(force_cases)))]
    run_cmd(cmd, cwd=repo_dir)


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def has_tooth_outputs(case_dir: Path) -> bool:
    if not case_dir.is_dir():
        return False
    for p in case_dir.iterdir():
        if p.is_dir() and p.name.startswith("tooth_"):
            return True
    return False


def build_supervised_subset(run_root: Path, sup_run_root: Path, selected_cases: List[str]) -> Path:
    processed_root = run_root / "processed"
    subset_root = sup_run_root / "processed_manual"
    subset_root.mkdir(parents=True, exist_ok=True)

    keep = set(selected_cases)
    for p in subset_root.iterdir():
        if p.name not in keep:
            remove_path(p)

    for case_id in selected_cases:
        src = processed_root / case_id
        if not has_tooth_outputs(src):
            continue
        dst = subset_root / case_id
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and Path(os.path.realpath(dst)) == src.resolve():
                continue
            remove_path(dst)
        os.symlink(src, dst)
    return subset_root


def render_supervised_config(
    repo_dir: Path,
    base_sup_config: str,
    out_config: str,
    tf_root: Path,
    subset_dir: Path,
    sup_output_dir: Path,
    pretrained_ckpt: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> Path:
    base_cfg_path = (repo_dir / base_sup_config).resolve() if not os.path.isabs(base_sup_config) else Path(base_sup_config)
    out_cfg_path = (repo_dir / out_config).resolve() if not os.path.isabs(out_config) else Path(out_config)
    cfg = yaml.safe_load(base_cfg_path.read_text())

    project = cfg.setdefault("project", {})
    project["device"] = "auto"

    data = cfg.setdefault("data", {})
    data["raw_layout"] = "toothfairy3"
    data["raw_dir"] = str(tf_root)
    data["images_tr_dir"] = str(tf_root / "imagesTr")
    data["labels_tr_dir"] = str(tf_root / "labelsTr")
    data["points_dir"] = str(tf_root / "pointsTr")
    data["meta_dir"] = None
    data["processed_dir"] = str(subset_dir)
    data["output_dir"] = str(sup_output_dir)

    train = cfg.setdefault("train", {})
    train["epochs"] = int(epochs)
    train["batch_size"] = int(batch_size)
    train["num_workers"] = int(num_workers)
    train["pretrained_ckpt"] = pretrained_ckpt
    train["pretrained_strict"] = False

    out_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    out_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))
    return out_cfg_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto pipeline: manual preprocess overwrite -> usage check -> supervised train/infer/viz/eval."
    )
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--manual-root", default="/root/手工标注1")
    parser.add_argument("--toothfairy-root", default="/root/ToothFairy3")
    parser.add_argument("--run-root", default="/root/cej_runs/run_unsup_001")
    parser.add_argument("--sup-run-root", default="/root/cej_runs/run_sup_manual_auto")
    parser.add_argument("--preprocess-config", default="configs/server_preprocess.yaml")
    parser.add_argument("--base-sup-config", default="configs/server_sup_manual6.yaml")
    parser.add_argument("--sup-config-out", default="configs/server_sup_manual_auto.yaml")
    parser.add_argument(
        "--pretrained-ckpt",
        default="/root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--case-prefix", default="auto", choices=["auto", "F", "P", "both"])
    parser.add_argument("--max-cases", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    manual_root = Path(args.manual_root).resolve()
    tf_root = Path(args.toothfairy_root).resolve()
    run_root = Path(args.run_root).resolve()
    sup_run_root = Path(args.sup_run_root).resolve()
    processed_root = run_root / "processed"
    sup_output_dir = sup_run_root / "outputs"
    max_cases = parse_max_cases(args.max_cases)

    report_csv = run_root / "manual_points_usage_report.csv"
    report_json = run_root / "manual_points_usage_report.json"

    # 1) Incremental manual preprocess overwrite by source change.
    run_manual_resume(
        python_exe=args.python_exe,
        repo_dir=repo_dir,
        manual_root=manual_root,
        tf_root=tf_root,
        run_root=run_root,
        preprocess_config=args.preprocess_config,
        case_prefix=args.case_prefix,
        max_cases=max_cases,
        rerun_on_source_update=True,
    )

    # 2) Usage check; if failed, force-rerun failed cases once, then re-check.
    report = run_usage_check(
        python_exe=args.python_exe,
        repo_dir=repo_dir,
        manual_root=manual_root,
        tf_root=tf_root,
        processed_root=processed_root,
        report_csv=report_csv,
        report_json=report_json,
        max_cases=max_cases,
    )
    failed_ids = collect_failed_case_ids(report)
    if failed_ids:
        print(f"[INFO] force rerun failed cases: {failed_ids}")
        run_manual_resume(
            python_exe=args.python_exe,
            repo_dir=repo_dir,
            manual_root=manual_root,
            tf_root=tf_root,
            run_root=run_root,
            preprocess_config=args.preprocess_config,
            case_prefix=args.case_prefix,
            max_cases=max_cases,
            rerun_on_source_update=True,
            force_cases=failed_ids,
        )
        report = run_usage_check(
            python_exe=args.python_exe,
            repo_dir=repo_dir,
            manual_root=manual_root,
            tf_root=tf_root,
            processed_root=processed_root,
            report_csv=report_csv,
            report_json=report_json,
            max_cases=max_cases,
        )
        failed_ids = collect_failed_case_ids(report)

    summary = report.get("summary", {})
    if failed_ids:
        print(f"[ERROR] still failed cases after force-rerun: {failed_ids}")
        print(f"[ERROR] usage report: {report_json}")
        return 3

    selected_cases = sorted(
        {
            str(rec.get("case_id"))
            for rec in report.get("cases", [])
            if str(rec.get("status", "")).startswith("PASS") and str(rec.get("case_id", "")).strip()
        }
    )
    if not selected_cases:
        print("[ERROR] no PASS cases available for supervised training")
        return 4

    print(f"[INFO] usage check all_used={summary.get('all_used')} selected_cases={len(selected_cases)}")
    print(f"[INFO] cases: {selected_cases}")

    # 3) Build supervised subset symlinks from validated PASS cases.
    subset_dir = build_supervised_subset(run_root=run_root, sup_run_root=sup_run_root, selected_cases=selected_cases)
    print(f"[INFO] subset_dir={subset_dir}")

    # 4) Render supervised config for this run.
    sup_cfg = render_supervised_config(
        repo_dir=repo_dir,
        base_sup_config=args.base_sup_config,
        out_config=args.sup_config_out,
        tf_root=tf_root,
        subset_dir=subset_dir,
        sup_output_dir=sup_output_dir,
        pretrained_ckpt=args.pretrained_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[INFO] supervised config: {sup_cfg}")

    # 5) Supervised pipeline.
    if not args.skip_train:
        run_cmd([args.python_exe, "-m", "src.train", "--config", str(sup_cfg)], cwd=repo_dir)
    if not args.skip_infer:
        run_cmd([args.python_exe, "-m", "src.infer", "--config", str(sup_cfg)], cwd=repo_dir)
    if not args.skip_viz:
        run_cmd([args.python_exe, "-m", "src.viz", "--config", str(sup_cfg)], cwd=repo_dir)
    if not args.skip_eval:
        run_cmd([args.python_exe, "-m", "src.eval", "--config", str(sup_cfg)], cwd=repo_dir)

    print("[INFO] pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
