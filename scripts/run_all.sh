#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname "$0")/.." && pwd)
CONFIG_PATH=${1:-configs/default.yaml}
export PYTHONPATH="$ROOT_DIR"

PY_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PY_BIN" ]; then
  PY_BIN="python3"
fi

RAW_DIR=$("$PY_BIN" - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r"))
print(cfg["data"]["raw_dir"])
PY
)

if [ ! -d "$RAW_DIR" ] || [ -z "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
  echo "[run_all] data/raw is empty. Creating synthetic case..."
  "$PY_BIN" -m src.synth --output_dir "$RAW_DIR" --case_id case_0001
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
