#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname "$0")/.." && pwd)
CONFIG_PATH=${1:-configs/default.yaml}
export PYTHONPATH="$ROOT_DIR"

RAW_DIR=$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r"))
print(cfg["data"]["raw_dir"])
PY
)

if [ ! -d "$RAW_DIR" ] || [ -z "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
  echo "[run_all] data/raw is empty. Creating synthetic case..."
  python3 -m src.synth --output_dir "$RAW_DIR" --case_id case_0001
fi

echo "[run_all] preprocess"
python3 -m src.preprocess --config "$CONFIG_PATH"

echo "[run_all] train"
python3 -m src.train --config "$CONFIG_PATH"

echo "[run_all] infer"
python3 -m src.infer --config "$CONFIG_PATH"

echo "[run_all] eval"
python3 -m src.eval --config "$CONFIG_PATH"

echo "[run_all] viz"
python3 -m src.viz --config "$CONFIG_PATH"

echo "[run_all] done"
