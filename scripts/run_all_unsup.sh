#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname "$0")/.." && pwd)
CONFIG_PATH=${1:-configs/unsup_mae.yaml}
export PYTHONPATH="$ROOT_DIR"

PY_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PY_BIN" ]; then
  PY_BIN="python3"
fi

echo "[run_all_unsup] preprocess (unsupervised)"
"$PY_BIN" -m src.preprocess_unsup --config "$CONFIG_PATH"

echo "[run_all_unsup] pretrain (masked reconstruction)"
"$PY_BIN" -m src.pretrain_mae --config "$CONFIG_PATH"

FINETUNE_ENABLED=$("$PY_BIN" - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r"))
print(str(cfg.get("finetune", {}).get("enable", False)))
PY
)

if [ "$FINETUNE_ENABLED" = "True" ] || [ "$FINETUNE_ENABLED" = "true" ]; then
  FT_CFG=$("$PY_BIN" - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r"))
print(cfg.get("finetune", {}).get("config", "configs/default.yaml"))
PY
)
  FT_CKPT=$("$PY_BIN" - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r"))
print(cfg.get("finetune", {}).get("pretrained_ckpt", ""))
PY
)
  FT_STRICT=$("$PY_BIN" - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r"))
print(str(cfg.get("finetune", {}).get("pretrained_strict", False)))
PY
)
  echo "[run_all_unsup] finetune (supervised)"
  if [ "$FT_STRICT" = "True" ] || [ "$FT_STRICT" = "true" ]; then
    "$PY_BIN" -m src.train --config "$FT_CFG" --pretrained "$FT_CKPT" --pretrained-strict
  else
    "$PY_BIN" -m src.train --config "$FT_CFG" --pretrained "$FT_CKPT"
  fi
fi

echo "[run_all_unsup] done"
