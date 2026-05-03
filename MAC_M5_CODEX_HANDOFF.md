# Mac M5 Codex Handoff

Last updated: 2026-05-03 UTC

This file is for the next Codex session running on a Mac M5 / Apple Silicon machine. Read this file first, then read `NEXT_SESSION_HANDOFF.md` for cloud-side history.

## Current Source State

- GitHub repo: `https://github.com/bulldize/CEJ_ABC`
- Branch: `codex/manual-points-group-col`
- Minimum expected commit: `7a4f70b docs: add next session handoff for incremental CEJ training`
- Current model family: CEJ full-mouth inference from single-tooth ROI heatmaps.
- Main usable checkpoint from the cloud run:
  `run_sup_inc_041055_001/outputs/train/checkpoints/last.pt`
- Main recommended review HTML from the cloud run:
  `run_sup_inc_041055_001/outputs_infer_unconstrained/viz/3d/index.html`
- Experimental ridgecurve output exists but was rejected by the user:
  `run_sup_inc_041055_001/outputs_infer_ridgecurve_001`
  Do not treat it as the main result.

## Critical Mac Constraint

Do not copy the Linux `.venv` to Mac and expect it to run.

The cloud `.venv` points to `/usr/local/miniconda3/envs/py310/bin/python` and contains Linux/CUDA wheels. Mac M5 needs a local macOS arm64 virtual environment. Checkpoints and NIfTI files are portable; the Linux Python environment is not.

## Mac Path Convention

Use these paths unless the user explicitly says otherwise:

```bash
REPO_DIR="$HOME/workspace/CEJ_ABC"
RUNTIME_ROOT="$HOME/cej_runtime"
TOOTHFAIRY_ROOT="$HOME/ToothFairy3"
UPLOADS_DIR="$RUNTIME_ROOT/uploads"
CEJ_RUNS_DIR="$RUNTIME_ROOT/cej_runs"
BACKUP_MODELS_DIR="$RUNTIME_ROOT/backup_models"
```

Expected raw ToothFairy3 layout on Mac:

```text
$TOOTHFAIRY_ROOT/imagesTr
$TOOTHFAIRY_ROOT/labelsTr
$TOOTHFAIRY_ROOT/pointsTr      # optional
```

## Restore From Transfer Bundle

The transfer bundle should contain runtime data only, not ToothFairy3 raw and not GitHub source files. Expected bundle name example:

```text
cej_abc_mac_runtime_data_20260503.tar.gz
```

Restore:

```bash
mkdir -p "$HOME/cej_runtime"
tar -xzf cej_abc_mac_runtime_data_20260503.tar.gz -C "$HOME/cej_runtime"
```

After extraction, confirm these exist:

```bash
test -f "$HOME/cej_runtime/cej_runs/run_sup_inc_041055_001/outputs/train/checkpoints/last.pt"
test -f "$HOME/cej_runtime/cej_runs/run_manual_inc_041055_001/manual_points_usage_report.json"
test -d "$HOME/cej_runtime/uploads/41-55"
```

If the tarball was created with absolute `/root/...` paths, extract it into a temporary directory and move these subtrees into the Mac runtime convention above. Do not preserve `/root` on Mac.

## Clone And Setup Environment

```bash
mkdir -p "$HOME/workspace"
cd "$HOME/workspace"
git clone https://github.com/bulldize/CEJ_ABC CEJ_ABC
cd "$HOME/workspace/CEJ_ABC"
git checkout codex/manual-points-group-col

bash scripts/setup_mac_mps.sh
source .venv/bin/activate
```

Verify MPS:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('mps_available', bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()))
PY
```

If `mps_available` is false or an MPS operator fails, use CPU by generating config with `--device cpu`.

## Generate Mac Config

Create a Mac-local config from the server config:

```bash
cd "$HOME/workspace/CEJ_ABC"
source .venv/bin/activate

python scripts/create_mac_config.py \
  --base-config configs/server_sup_manual_inc_041055.yaml \
  --out configs/mac_sup_manual_inc_041055.yaml \
  --runtime-root "$HOME/cej_runtime" \
  --toothfairy-root "$HOME/ToothFairy3" \
  --device mps
```

This rewrites:

- `project.device` to `mps`
- ToothFairy3 raw paths to `$HOME/ToothFairy3`
- processed and output paths to `$HOME/cej_runtime/cej_runs/run_sup_inc_041055_001`
- pretrained checkpoint to the transferred latest checkpoint
- inference flags to keep unconstrained CEJ curve behavior

## Direct Inference And Visualization On Mac

Run this after raw ToothFairy3 exists locally and runtime data has been restored:

```bash
cd "$HOME/workspace/CEJ_ABC"
source .venv/bin/activate

python -m src.infer --config configs/mac_sup_manual_inc_041055.yaml
python -m src.viz --config configs/mac_sup_manual_inc_041055.yaml
```

Expected output:

```text
$HOME/cej_runtime/cej_runs/run_sup_inc_041055_001/outputs_mac/infer
$HOME/cej_runtime/cej_runs/run_sup_inc_041055_001/outputs_mac/viz/3d/index.html
```

For quick review without recomputing, open the transferred cloud HTML if present:

```text
$HOME/cej_runtime/cej_runs/run_sup_inc_041055_001/outputs_infer_unconstrained/viz/3d/index.html
```

## Continue With New Manual Uploads On Mac

Use a new run tag. Do not overwrite existing run directories.

```bash
cd "$HOME/workspace/CEJ_ABC"
source .venv/bin/activate

MANUAL_ROOT="$HOME/cej_runtime/uploads/<NEW_BATCH>"
TF_ROOT="$HOME/ToothFairy3"
RUN_TAG="<NEW_TAG>"              # example: 056070_001
PRE_RUN="$HOME/cej_runtime/cej_runs/run_manual_inc_${RUN_TAG}"
SUP_RUN="$HOME/cej_runtime/cej_runs/run_sup_inc_${RUN_TAG}"
SUP_CFG="configs/mac_sup_manual_inc_${RUN_TAG}.yaml"
PREV_CKPT="$HOME/cej_runtime/cej_runs/run_sup_inc_041055_001/outputs/train/checkpoints/last.pt"

python scripts/run_manual_supervised_pipeline.py \
  --repo-dir "$HOME/workspace/CEJ_ABC" \
  --python-exe "$HOME/workspace/CEJ_ABC/.venv/bin/python" \
  --manual-root "$MANUAL_ROOT" \
  --toothfairy-root "$TF_ROOT" \
  --run-root "$PRE_RUN" \
  --sup-run-root "$SUP_RUN" \
  --preprocess-config configs/server_preprocess.yaml \
  --base-sup-config configs/server_sup_manual6.yaml \
  --sup-config-out "$SUP_CFG" \
  --pretrained-ckpt "$PREV_CKPT" \
  --epochs 20 \
  --batch-size 1 \
  --num-workers 1 \
  --case-prefix auto
```

Mac MPS may be slower than cloud GPU. For heavy training, prefer cloud GPU; for small incremental runs and visualization, Mac is acceptable.

## Validation Commands

Usage check:

```bash
python scripts/check_manual_points_usage.py \
  --manual-root "$MANUAL_ROOT" \
  --toothfairy-root "$TF_ROOT" \
  --processed-root "$PRE_RUN/processed" \
  --report-csv "$PRE_RUN/manual_points_usage_report.csv" \
  --report-json "$PRE_RUN/manual_points_usage_report.json" \
  --case-prefix auto
```

Model and output check:

```bash
tail -n 5 "$SUP_RUN/outputs/train/metrics.csv"
ls -lh "$SUP_RUN/outputs/train/checkpoints/last.pt"
ls -lh "$SUP_RUN/outputs/viz/3d/index.html"
```

## Do Not Use Unless Explicitly Requested

- `configs/server_sup_manual_merged_041055.yaml`
- `run_manual_merged_041055_001`
- `run_sup_manual_merged_041055_001`
- `outputs_infer_ridgecurve_001` as a main result

Those were temporary/experimental paths from prior investigation.

## Current Quality Notes

- `041-55` incremental data has `all_used=true` in the usage report.
- `044-11` and `044-15` have no source manual GT; they are pure inference/test-set style teeth.
- User currently prefers not to pursue the new ridgecurve heatmap-to-curve method.
- Main usable inference remains unconstrained curve output from `outputs_infer_unconstrained`.
- If a tooth has only a few predicted curve points, first inspect `H_pred.nii.gz`; it may be a post-processing issue rather than pure model failure.
