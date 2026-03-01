#!/usr/bin/env python3
import argparse
import copy
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def strip_nii_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def case_id_from_image_name(name: str) -> str:
    stem = strip_nii_ext(name)
    if stem.endswith("_0000"):
        return stem[:-5]
    return stem


def find_label_path(labels_dir: Path, case_id: str) -> Path | None:
    p1 = labels_dir / f"{case_id}.nii.gz"
    if p1.exists():
        return p1
    p2 = labels_dir / f"{case_id}.nii"
    if p2.exists():
        return p2
    return None


def has_tooth_outputs(case_processed_dir: Path) -> bool:
    if not case_processed_dir.is_dir():
        return False
    for child in case_processed_dir.iterdir():
        if child.is_dir() and child.name.startswith("tooth_"):
            return True
    return False


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                data.setdefault("done_cases", [])
                data.setdefault("failed_cases", {})
                data.setdefault("last_case", None)
                data.setdefault("updated_at", now_iso())
                return data
        except Exception:
            pass
    return {
        "done_cases": [],
        "failed_cases": {},
        "last_case": None,
        "updated_at": now_iso(),
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))


def build_single_case_cfg(base_cfg: dict, run_root: Path, work_raw_dir: Path) -> dict:
    cfg = copy.deepcopy(base_cfg)
    data = cfg.setdefault("data", {})
    data["raw_layout"] = "toothfairy3"
    data["raw_dir"] = str(work_raw_dir)
    data["images_tr_dir"] = str(work_raw_dir / "imagesTr")
    data["labels_tr_dir"] = str(work_raw_dir / "labelsTr")
    data["points_dir"] = str(work_raw_dir / "pointsTr")
    data["meta_dir"] = None
    data["processed_dir"] = str(run_root / "processed")
    data["output_dir"] = str(run_root / "outputs")
    data["max_cases"] = 1
    return cfg


def symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume ToothFairy3 preprocess by case.")
    parser.add_argument("--raw-dir", default="/root/ToothFairy3")
    parser.add_argument("--run-root", default="/root/cej_runs/run_unsup_001")
    parser.add_argument("--base-config", default="configs/server_preprocess.yaml")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--max-cases", default=None, help="Limit cases for this run")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    run_root = Path(args.run_root).resolve()
    processed_dir = run_root / "processed"
    state_path = run_root / "preprocess_state.json"

    base_cfg_path = (repo_dir / args.base_config).resolve() if not os.path.isabs(args.base_config) else Path(args.base_config)
    if not base_cfg_path.exists():
        print(f"[ERROR] base config not found: {base_cfg_path}")
        return 2

    base_cfg = yaml.safe_load(base_cfg_path.read_text())
    state = load_state(state_path)

    images_dir = raw_dir / "imagesTr"
    labels_dir = raw_dir / "labelsTr"
    points_dir = raw_dir / "pointsTr"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"[ERROR] missing required dirs: imagesTr={images_dir.is_dir()} labelsTr={labels_dir.is_dir()}")
        save_state(state_path, state)
        return 1

    image_paths = sorted(
        [Path(p) for p in glob.glob(str(images_dir / "*.nii"))]
        + [Path(p) for p in glob.glob(str(images_dir / "*.nii.gz"))]
    )

    cases = []
    for image_path in image_paths:
        case_id = case_id_from_image_name(image_path.name)
        label_path = find_label_path(labels_dir, case_id)
        if label_path is None:
            state["failed_cases"][case_id] = "missing_label"
            continue
        cases.append((case_id, image_path, label_path))

    max_cases = None
    if args.max_cases not in (None, "", "null", "None"):
        try:
            max_cases = int(args.max_cases)
        except ValueError:
            max_cases = None
    if max_cases is not None and max_cases > 0:
        cases = cases[:max_cases]

    print(f"[INFO] discovered_cases={len(cases)}")
    done_set = set(state.get("done_cases", []))
    failed_map = dict(state.get("failed_cases", {}))

    for case_id, image_path, label_path in cases:
        state["last_case"] = case_id
        case_processed_dir = processed_dir / case_id
        if has_tooth_outputs(case_processed_dir):
            print(f"[SKIP] case={case_id} already processed")
            done_set.add(case_id)
            failed_map.pop(case_id, None)
            state["done_cases"] = sorted(done_set)
            state["failed_cases"] = failed_map
            save_state(state_path, state)
            continue

        print(f"[RUN] case={case_id}")
        with tempfile.TemporaryDirectory(prefix=f"pp_{case_id}_") as tmp_root:
            tmp_root_p = Path(tmp_root)
            one_raw = tmp_root_p / "raw"
            one_images = one_raw / "imagesTr"
            one_labels = one_raw / "labelsTr"
            one_points = one_raw / "pointsTr"
            one_images.mkdir(parents=True, exist_ok=True)
            one_labels.mkdir(parents=True, exist_ok=True)
            one_points.mkdir(parents=True, exist_ok=True)

            symlink_or_copy(image_path, one_images / image_path.name)
            symlink_or_copy(label_path, one_labels / label_path.name)

            point_src = points_dir / f"{case_id}.json"
            if point_src.exists():
                symlink_or_copy(point_src, one_points / point_src.name)

            cfg = build_single_case_cfg(base_cfg, run_root, one_raw)
            tmp_cfg = tmp_root_p / "config.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))

            cmd = [args.python_exe, "-m", "src.preprocess", "--config", str(tmp_cfg)]
            proc = subprocess.run(cmd, cwd=repo_dir)

        if proc.returncode == 0 and has_tooth_outputs(case_processed_dir):
            done_set.add(case_id)
            failed_map.pop(case_id, None)
            print(f"[DONE] case={case_id}")
        elif proc.returncode == 0:
            failed_map[case_id] = "no_tooth_outputs"
            print(f"[FAIL] case={case_id} reason=no_tooth_outputs")
        else:
            failed_map[case_id] = f"returncode_{proc.returncode}"
            print(f"[FAIL] case={case_id} reason=returncode_{proc.returncode}")

        state["done_cases"] = sorted(done_set)
        state["failed_cases"] = failed_map
        save_state(state_path, state)

    state["done_cases"] = sorted(done_set)
    state["failed_cases"] = failed_map
    save_state(state_path, state)
    print(f"[INFO] done={len(state['done_cases'])} failed={len(state['failed_cases'])} state={state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
