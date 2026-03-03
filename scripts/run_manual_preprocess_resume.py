#!/usr/bin/env python3
import argparse
import copy
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_state(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                data.setdefault("done_cases", [])
                data.setdefault("failed_cases", {})
                data.setdefault("source_meta", {})
                data.setdefault("last_case", None)
                data.setdefault("updated_at", now_iso())
                return data
        except Exception:
            pass
    return {
        "done_cases": [],
        "failed_cases": {},
        "source_meta": {},
        "last_case": None,
        "updated_at": now_iso(),
    }


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))


def file_meta(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    st = path.stat()
    return {
        "path": str(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def has_tooth_outputs(case_processed_dir: Path) -> bool:
    if not case_processed_dir.is_dir():
        return False
    for child in case_processed_dir.iterdir():
        if child.is_dir() and child.name.startswith("tooth_"):
            return True
    return False


def parse_max_cases(raw: Optional[str]) -> Optional[int]:
    if raw in (None, "", "null", "None"):
        return None
    try:
        v = int(raw)
    except Exception:
        return None
    return v if v > 0 else None


def parse_force_cases(raw: Optional[str]) -> set:
    if raw in (None, "", "null", "None"):
        return set()
    items = []
    for x in str(raw).replace(";", ",").split(","):
        s = x.strip()
        if s:
            items.append(s)
    return set(items)


def choose_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def detect_points_file(manual_case_dir: Path) -> Optional[Path]:
    return choose_existing_path(
        [
            manual_case_dir / "cej_points_ras.xlsx",
            manual_case_dir / "points.xlsx",
        ]
    )


def detect_boundary_file(manual_case_dir: Path) -> Optional[Path]:
    return choose_existing_path(
        [
            manual_case_dir / "scan_boundary_ras.xlsx",
            manual_case_dir / "CBCT_8_Boundary_Points.xlsx",
        ]
    )


def normalize_sid(raw: str) -> str:
    sid = raw.strip()
    if sid.isdigit() and len(sid) < 3:
        return sid.zfill(3)
    return sid


def parse_dir_case_hint(dir_name: str) -> Tuple[Optional[str], Optional[str]]:
    # 1) Full case id folder: ToothFairy3F_009
    m = re.fullmatch(r"ToothFairy3([FP])_(\d+)", dir_name)
    if m:
        return m.group(1), normalize_sid(m.group(2))

    # 2) Generic folder with trailing numeric sid: 009
    m = re.search(r"(\d+)$", dir_name)
    if m:
        return None, normalize_sid(m.group(1))

    return None, None


def find_tooth_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "牙位" in str(c):
            return c
    for c in df.columns:
        if "tooth" in str(c).lower():
            return c
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"group", "grp"} or "group" in lc:
            return c
    return None


def infer_prefix_from_points(points_xlsx: Path) -> Optional[str]:
    try:
        df = pd.read_excel(points_xlsx, nrows=128)
    except Exception:
        return None
    tooth_col = find_tooth_col(df)
    if tooth_col is None:
        return None

    cnt = {"F": 0, "P": 0}
    values = df[tooth_col].dropna().astype(str).head(128)
    for v in values:
        m = re.match(r"\s*([FP])[_\-]", v, flags=re.IGNORECASE)
        if m:
            cnt[m.group(1).upper()] += 1
    if cnt["F"] == 0 and cnt["P"] == 0:
        return None
    return "F" if cnt["F"] >= cnt["P"] else "P"


def case_files(tf_root: Path, case_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    image = choose_existing_path(
        [
            tf_root / "imagesTr" / f"{case_id}_0000.nii.gz",
            tf_root / "imagesTr" / f"{case_id}_0000.nii",
        ]
    )
    label = choose_existing_path(
        [
            tf_root / "labelsTr" / f"{case_id}.nii.gz",
            tf_root / "labelsTr" / f"{case_id}.nii",
        ]
    )
    return image, label


def build_case_ids(sid: str, prefix_hint: Optional[str], inferred_prefix: Optional[str], case_prefix: str) -> List[str]:
    if prefix_hint in {"F", "P"}:
        return [f"ToothFairy3{prefix_hint}_{sid}"]

    if case_prefix == "F":
        order = ["F"]
    elif case_prefix == "P":
        order = ["P"]
    elif case_prefix == "both":
        order = ["F", "P"]
    else:  # auto
        if inferred_prefix in {"F", "P"}:
            other = "P" if inferred_prefix == "F" else "F"
            order = [inferred_prefix, other]
        else:
            order = ["F", "P"]

    return [f"ToothFairy3{p}_{sid}" for p in order]


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def build_single_case_cfg(
    base_cfg: Dict[str, Any],
    run_root: Path,
    raw_case_root: Path,
    boundary_path: Optional[Path],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    data = cfg.setdefault("data", {})
    data["raw_layout"] = "case_dirs"
    data["raw_dir"] = str(raw_case_root)
    data["images_tr_dir"] = None
    data["labels_tr_dir"] = None
    data["points_dir"] = None
    data["meta_dir"] = None
    data["raw_a_name"] = "A.nii.gz"
    data["raw_b_name"] = "B.nii.gz"
    data["raw_points_name"] = "points.json"
    data["raw_meta_name"] = "meta.json"
    data["cej_points_name"] = "cej_points_ras.xlsx"
    data["scan_boundary_path"] = str(boundary_path) if boundary_path is not None else "__missing_scan_boundary__.xlsx"
    data["processed_dir"] = str(run_root / "processed")
    data["output_dir"] = str(run_root / "outputs")
    data["max_cases"] = 1
    return cfg


def discover_manual_cases(manual_root: Path, tf_root: Path, case_prefix: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for d in sorted([p for p in manual_root.glob("*") if p.is_dir()]):
        points_file = detect_points_file(d)
        if points_file is None:
            continue

        prefix_hint, sid = parse_dir_case_hint(d.name)
        if sid is None:
            continue

        inferred = infer_prefix_from_points(points_file)
        candidates = build_case_ids(sid, prefix_hint, inferred, case_prefix)

        chosen_case_id = None
        chosen_image = None
        chosen_label = None
        for cid in candidates:
            image_path, label_path = case_files(tf_root, cid)
            if image_path is not None and label_path is not None:
                chosen_case_id = cid
                chosen_image = image_path
                chosen_label = label_path
                break

        if chosen_case_id is None:
            cases.append(
                {
                    "manual_dir": d,
                    "points_file": points_file,
                    "boundary_file": detect_boundary_file(d),
                    "case_id": None,
                    "error": f"missing_image_or_label_for_candidates:{','.join(candidates)}",
                }
            )
            continue

        boundary_file = detect_boundary_file(d)
        cases.append(
            {
                "manual_dir": d,
                "points_file": points_file,
                "boundary_file": boundary_file,
                "case_id": chosen_case_id,
                "image_path": chosen_image,
                "label_path": chosen_label,
                "source_meta": {
                    "manual_dir": str(d),
                    "points_file": file_meta(points_file),
                    "boundary_file": file_meta(boundary_file),
                    "sid": sid,
                },
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume preprocess for manually annotated CEJ points.")
    parser.add_argument("--manual-root", default="/root/手工标注1")
    parser.add_argument("--toothfairy-root", default="/root/ToothFairy3")
    parser.add_argument("--run-root", default="/root/cej_runs/run_unsup_001")
    parser.add_argument("--state-name", default="manual_preprocess_state.json")
    parser.add_argument("--base-config", default="configs/server_preprocess.yaml")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--max-cases", default=None)
    parser.add_argument(
        "--case-prefix",
        default="auto",
        choices=["auto", "F", "P", "both"],
        help="How to map folder sid to ToothFairy3 case prefix when folder name has no explicit prefix.",
    )
    parser.add_argument(
        "--rerun-on-source-update",
        action="store_true",
        help="If source xlsx changed since last successful run, rerun even when processed outputs already exist.",
    )
    parser.add_argument(
        "--force-cases",
        default="",
        help="Comma-separated case_ids to rerun regardless of existing processed outputs.",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    manual_root = Path(args.manual_root).resolve()
    tf_root = Path(args.toothfairy_root).resolve()
    run_root = Path(args.run_root).resolve()
    processed_dir = run_root / "processed"
    state_path = run_root / args.state_name

    base_cfg_path = (repo_dir / args.base_config).resolve() if not os.path.isabs(args.base_config) else Path(args.base_config)
    if not base_cfg_path.exists():
        print(f"[ERROR] base config not found: {base_cfg_path}")
        return 2

    if not manual_root.is_dir():
        print(f"[ERROR] manual root not found: {manual_root}")
        return 2
    if not (tf_root / "imagesTr").is_dir() or not (tf_root / "labelsTr").is_dir():
        print(f"[ERROR] toothfairy root missing imagesTr/labelsTr: {tf_root}")
        return 2

    base_cfg = yaml.safe_load(base_cfg_path.read_text())
    state = load_state(state_path)
    done_set = set(state.get("done_cases", []))
    failed_map = dict(state.get("failed_cases", {}))
    source_meta_map = dict(state.get("source_meta", {}))

    cases = discover_manual_cases(manual_root, tf_root, args.case_prefix)
    max_cases = parse_max_cases(args.max_cases)
    if max_cases is not None:
        cases = cases[:max_cases]
    force_cases = parse_force_cases(args.force_cases)

    print(f"[INFO] discovered_manual_dirs={len(cases)}")
    if force_cases:
        print(f"[INFO] force_cases={sorted(force_cases)}")

    for rec in cases:
        case_id = rec.get("case_id")
        if case_id is None:
            key = str(rec["manual_dir"].name)
            reason = rec.get("error", "unknown_case_mapping_error")
            print(f"[FAIL] manual_dir={key} reason={reason}")
            failed_map[key] = reason
            state["failed_cases"] = failed_map
            state["done_cases"] = sorted(done_set)
            state["source_meta"] = source_meta_map
            save_state(state_path, state)
            continue

        state["last_case"] = case_id
        case_processed_dir = processed_dir / case_id
        current_source_meta = rec.get("source_meta", {})
        prev_source_meta = source_meta_map.get(case_id)
        force_rerun = case_id in force_cases

        if has_tooth_outputs(case_processed_dir):
            if force_rerun:
                print(f"[RERUN] case={case_id} forced by --force-cases")
            elif args.rerun_on_source_update and prev_source_meta is not None and prev_source_meta != current_source_meta:
                print(f"[RERUN] case={case_id} source changed")
            else:
                print(f"[SKIP] case={case_id} already processed")
                done_set.add(case_id)
                failed_map.pop(case_id, None)
                source_meta_map[case_id] = current_source_meta
                state["done_cases"] = sorted(done_set)
                state["failed_cases"] = failed_map
                state["source_meta"] = source_meta_map
                save_state(state_path, state)
                continue

        print(f"[RUN] case={case_id}")
        with tempfile.TemporaryDirectory(prefix=f"manual_{case_id}_") as tmp_root:
            tmp_root_p = Path(tmp_root)
            raw_root = tmp_root_p / "raw"
            one_case_dir = raw_root / case_id
            one_case_dir.mkdir(parents=True, exist_ok=True)

            link_or_copy(rec["image_path"], one_case_dir / "A.nii.gz")
            link_or_copy(rec["label_path"], one_case_dir / "B.nii.gz")
            link_or_copy(rec["points_file"], one_case_dir / "cej_points_ras.xlsx")

            boundary_for_cfg: Optional[Path] = None
            if rec.get("boundary_file") is not None:
                boundary_for_cfg = one_case_dir / "scan_boundary_ras.xlsx"
                link_or_copy(rec["boundary_file"], boundary_for_cfg)

            cfg = build_single_case_cfg(base_cfg, run_root, raw_root, boundary_for_cfg)
            cfg_path = tmp_root_p / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))

            cmd = [args.python_exe, "-m", "src.preprocess", "--config", str(cfg_path)]
            proc = subprocess.run(cmd, cwd=repo_dir)

        if proc.returncode == 0 and has_tooth_outputs(case_processed_dir):
            print(f"[DONE] case={case_id}")
            done_set.add(case_id)
            failed_map.pop(case_id, None)
            source_meta_map[case_id] = current_source_meta
        elif proc.returncode == 0:
            print(f"[FAIL] case={case_id} reason=no_tooth_outputs")
            failed_map[case_id] = "no_tooth_outputs"
        else:
            print(f"[FAIL] case={case_id} reason=returncode_{proc.returncode}")
            failed_map[case_id] = f"returncode_{proc.returncode}"

        state["done_cases"] = sorted(done_set)
        state["failed_cases"] = failed_map
        state["source_meta"] = source_meta_map
        save_state(state_path, state)

    state["done_cases"] = sorted(done_set)
    state["failed_cases"] = failed_map
    state["source_meta"] = source_meta_map
    save_state(state_path, state)
    print(f"[INFO] done={len(state['done_cases'])} failed={len(state['failed_cases'])} state={state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
