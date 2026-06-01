#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT_DIR"

PY_BIN=${PY_BIN:-"$ROOT_DIR/.venv/bin/python"}
if [ ! -x "$PY_BIN" ]; then
  PY_BIN="python"
fi

RUN_TS=${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}
RUN_BASE=${RUN_BASE:-/root/cej_isolated_runs/C}
UNSUP_CKPT=${UNSUP_CKPT:-/root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt}
STAGE2_MAX_ATTEMPTS=${STAGE2_MAX_ATTEMPTS:-3}
STAGE2_TARGET_TRAIN_STEPS=${STAGE2_TARGET_TRAIN_STEPS:-3437}
STAGE2_TARGET_STEP_MULTIPLIER=${STAGE2_TARGET_STEP_MULTIPLIER:-2}
STAGE1=${STAGE1:-$RUN_BASE/unsup_to_goal_retrain_${RUN_TS}_stage1}
STAGE2_BASE=${STAGE2:-$RUN_BASE/unsup_to_goal_retrain_${RUN_TS}_stage2_goal19}

echo "[unsup_to_goal] repo=$ROOT_DIR"
echo "[unsup_to_goal] run_ts=$RUN_TS"
echo "[unsup_to_goal] stage1=$STAGE1"
echo "[unsup_to_goal] stage2_base=$STAGE2_BASE"
echo "[unsup_to_goal] unsup_ckpt=$UNSUP_CKPT"

"$PY_BIN" "$ROOT_DIR/scripts/run_cej_unsup_retrain.py" \
  --repo-dir "$ROOT_DIR" \
  --python-exe "$PY_BIN" \
  --run-root "$STAGE1" \
  --unsup-ckpt "$UNSUP_CKPT" \
  --experiments exp03_skeleton_aux_loss \
  --setup-only

"$PY_BIN" "$ROOT_DIR/scripts/run_cej_unsup_retrain.py" \
  --repo-dir "$ROOT_DIR" \
  --python-exe "$PY_BIN" \
  --run-root "$STAGE1" \
  --unsup-ckpt "$UNSUP_CKPT" \
  --experiments exp03_skeleton_aux_loss \
  --resume \
  --stop-on-failure

STAGE1_BEST="$STAGE1/experiments/exp03_skeleton_aux_loss/outputs/train/checkpoints/best.pt"

TARGET_STEPS=$STAGE2_TARGET_TRAIN_STEPS
ACCEPTED=0
FINAL_STAGE2=""

for ATTEMPT in $(seq 1 "$STAGE2_MAX_ATTEMPTS"); do
  if [ "$ATTEMPT" -eq 1 ]; then
    STAGE2="$STAGE2_BASE"
  else
    STAGE2="${STAGE2_BASE}_retry${ATTEMPT}_steps${TARGET_STEPS}"
  fi
  echo "[unsup_to_goal] stage2_attempt=$ATTEMPT target_train_steps=$TARGET_STEPS run_root=$STAGE2"
  "$PY_BIN" "$ROOT_DIR/scripts/run_goal_mode_19full.py" \
    --repo-dir "$ROOT_DIR" \
    --python-exe "$PY_BIN" \
    --run-root "$STAGE2" \
    --best-ckpt "$STAGE1_BEST" \
    --unsup-ckpt "$UNSUP_CKPT" \
    --target-train-steps "$TARGET_STEPS" \
    --device cuda

  FINAL_STAGE2="$STAGE2"
  ACCEPTED=$("$PY_BIN" - <<PY
import json
from pathlib import Path
p = Path("${STAGE2}") / "summary" / "acceptance_report.json"
print(1 if p.exists() and json.loads(p.read_text()).get("accepted") else 0)
PY
)
  if [ "$ACCEPTED" -eq 1 ]; then
    break
  fi
  TARGET_STEPS=$((TARGET_STEPS * STAGE2_TARGET_STEP_MULTIPLIER))
done

echo "[unsup_to_goal] stage1_best=$STAGE1_BEST"
echo "[unsup_to_goal] final_stage2=$FINAL_STAGE2"
echo "[unsup_to_goal] acceptance_report=$FINAL_STAGE2/summary/acceptance_report.json"
echo "[unsup_to_goal] final_metrics=$FINAL_STAGE2/final/eval/metrics_summary.json"

if [ "$ACCEPTED" -ne 1 ]; then
  echo "[unsup_to_goal] ERROR: Stage 2 did not meet strict acceptance after $STAGE2_MAX_ATTEMPTS attempts." >&2
  exit 2
fi
