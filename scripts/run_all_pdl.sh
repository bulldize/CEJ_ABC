#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname "$0")/.." && pwd)
RUNTIME_DIR=${PDL_RUNTIME_DIR:-/Users/bulldize/Desktop/华西口腔_CEJ_ABC/runtime/pdl}
CONFIG_PATH=${1:-"$RUNTIME_DIR/configs/pdl_runtime.yaml"}
export PYTHONPATH="$ROOT_DIR"
export MPLCONFIGDIR="$RUNTIME_DIR/.mpl_cache"
mkdir -p "$MPLCONFIGDIR"

PY_BIN=${PY_BIN:-"$ROOT_DIR/.venv/bin/python"}
if [ ! -x "$PY_BIN" ]; then
  PY_BIN="python3"
fi

echo "[run_all_pdl] pdl preprocess"
"$PY_BIN" -m src.pdl_preprocess --config "$CONFIG_PATH"

echo "[run_all_pdl] pdl extract"
"$PY_BIN" -m src.pdl_extract --config "$CONFIG_PATH"

echo "[run_all_pdl] pdl viz"
"$PY_BIN" -m src.pdl_viz --config "$CONFIG_PATH"

echo "[run_all_pdl] pdl eval"
"$PY_BIN" -m src.pdl_eval --config "$CONFIG_PATH"

echo "[run_all_pdl] done"
