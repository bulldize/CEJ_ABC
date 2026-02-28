#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname "$0")/.." && pwd)
CONFIG_PATH=${1:-configs/default.yaml}
export PYTHONPATH="$ROOT_DIR"
export MPLCONFIGDIR="$ROOT_DIR/outputs/.mpl_cache"
mkdir -p "$MPLCONFIGDIR"

PY_BIN=${PY_BIN:-"$ROOT_DIR/.venv/bin/python"}
if [ ! -x "$PY_BIN" ]; then
  PY_BIN="python3"
fi

RAW_DIR=$("$PY_BIN" - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r"))
print(cfg["data"]["raw_dir"])
PY
)

CASE_DIR="$RAW_DIR/TF_008"
A_DST="$CASE_DIR/A.nii.gz"
B_DST="$CASE_DIR/B.nii.gz"

NEED_PREPARE=0
if [ ! -d "$RAW_DIR" ] || [ -z "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
  NEED_PREPARE=1
elif [ ! -e "$A_DST" ] || [ ! -e "$B_DST" ]; then
  NEED_PREPARE=1
fi

if [ "$NEED_PREPARE" -eq 1 ]; then
  echo "[run_all] preparing TF_008 case..."
  "$PY_BIN" - <<PY
import os
from pathlib import Path

root = Path("${ROOT_DIR}")
raw_dir = Path("${RAW_DIR}")
src_dir = root / "data" / "TF_008"
if not src_dir.exists():
    raise SystemExit("[run_all] data/TF_008 not found; please provide TF_008 data.")

def resolve_nifti(p: Path) -> Path:
    if p.is_dir():
        candidate = p / p.name
        if candidate.exists():
            return candidate
        for f in p.iterdir():
            if f.name.endswith(".nii") or f.name.endswith(".nii.gz"):
                return f
    return p

case_dir = raw_dir / "TF_008"
case_dir.mkdir(parents=True, exist_ok=True)

a_src = resolve_nifti(src_dir / "ToothFairy3F_008_volume.nii")
b_src = resolve_nifti(src_dir / "ToothFairy3F_008label.nii")

if not a_src.is_file():
    raise SystemExit(f"[run_all] missing volume file: {a_src}")
if not b_src.is_file():
    raise SystemExit(f"[run_all] missing label file: {b_src}")

a_dst = case_dir / "A.nii.gz"
b_dst = case_dir / "B.nii.gz"

def ensure_link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            raise SystemExit(f"[run_all] unexpected directory at {dst}")
        dst.unlink()
    try:
        os.symlink(src, dst)
    except Exception:
        # fallback to copy if symlink fails
        import shutil
        shutil.copy2(src, dst)

ensure_link(a_src, a_dst)
ensure_link(b_src, b_dst)
print(f"[run_all] TF_008 prepared at {case_dir}")
PY
fi

echo "[run_all] preprocess"
"$PY_BIN" -m src.preprocess --config "$CONFIG_PATH"

echo "[run_all] train"
"$PY_BIN" -m src.train --config "$CONFIG_PATH"

echo "[run_all] infer"
"$PY_BIN" -m src.infer --config "$CONFIG_PATH"

echo "[run_all] eval"
"$PY_BIN" -m src.eval --config "$CONFIG_PATH"

echo "[run_all] viz"
"$PY_BIN" -m src.viz --config "$CONFIG_PATH"

echo "[run_all] done"
