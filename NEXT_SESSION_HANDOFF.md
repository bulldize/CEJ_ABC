# CEJ ABC Next Session Handoff

Last updated: 2026-05-03 UTC

## Current Git State

- Repo: `/root/workspace/CEJ_ABC`
- Branch: `codex/manual-points-group-col`
- Last pushed commit: `0629045 feat(workflow): support json manual points and unconstrained CEJ inference`
- Important: `src.infer` now defaults to **not** constraining predicted CEJ curve to the tooth mask.
- Uncommitted local scratch may exist: `configs/server_sup_manual_merged_041055.yaml`. It was from an accidental merged old+new run and should not be used unless intentionally reviewed.

## Current Data/Model State

- ToothFairy3 root: `/root/ToothFairy3`
- Latest incremental manual upload processed: `/root/uploads/41-55`
- Incremental preprocess run root: `/root/cej_runs/run_manual_inc_041055_001`
- Incremental supervised run root: `/root/cej_runs/run_sup_inc_041055_001`
- Latest trained checkpoint:
  `/root/cej_runs/run_sup_inc_041055_001/outputs/train/checkpoints/last.pt`
- Latest checkpoint backup tar:
  `/root/backup_models/run_sup_inc_041055_001_last_checkpoint_20260503.tar.gz`
- Latest normal viz HTML:
  `/root/cej_runs/run_sup_inc_041055_001/outputs/viz/3d/index.html`
- Latest unconstrained inference/viz HTML:
  `/root/cej_runs/run_sup_inc_041055_001/outputs_infer_unconstrained/viz/3d/index.html`
- Incremental usage report:
  `/root/cej_runs/run_manual_inc_041055_001/manual_points_usage_report.json`

## Completed Upload 41-55

- Processed cases: `ToothFairy3F_041`, `044`, `050`, `051`, `052`, `053`, `054`, `055`
- Usage check: `all_used=true`, `fail_cases=0`
- Manual source format: Slicer `.mrk.json` only, no Excel.
- Case `055`: ignored `/root/uploads/41-55/55/F_41.mrk.json` via `/root/uploads/41-55/55/cej_ignore_files.txt` because user requested dropping that JSON.
- Each used manual case has `cej_usage_status.json` written under upload dir and processed dir.

## Important Behavior Changes Already Committed

- JSON-only manual annotations are supported through `src/datasets/manual_points_source.py`.
- Excel and Slicer `.mrk.json` are normalized into the same training point table.
- For Slicer `.mrk.json`, tooth id is preferably inferred from file name, then label fallback.
- `cej_ignore_files.txt` can ignore specific `.mrk.json` files inside a manual case folder.
- Incremental manual pipeline writes usage markers so next sessions can see which uploaded cases were used.
- `infer.constrain_curve_to_tooth_mask=false` is now explicit in generated configs and default config.
- `infer.constrain_curve_to_tooth_surface=false`, `keep_lcc_for_curve=false`, `fit_pred_curve=true` are explicit for inference.

## How To Continue With A New Uploaded Batch

Set these variables for the next uploaded manual batch:

```bash
cd /root/workspace/CEJ_ABC
source .venv/bin/activate

MANUAL_ROOT=/root/uploads/<NEW_BATCH_DIR>
TF_ROOT=/root/ToothFairy3
PREV_CKPT=/root/cej_runs/run_sup_inc_041055_001/outputs/train/checkpoints/last.pt
RUN_TAG=<NEW_TAG>   # example: 056070_001
PRE_RUN=/root/cej_runs/run_manual_inc_${RUN_TAG}
SUP_RUN=/root/cej_runs/run_sup_inc_${RUN_TAG}
SUP_CFG=configs/server_sup_manual_inc_${RUN_TAG}.yaml
```

Run incremental preprocess + usage check + supervised fine-tune + infer + viz + eval:

```bash
python scripts/run_manual_supervised_pipeline.py \
  --repo-dir /root/workspace/CEJ_ABC \
  --python-exe /root/workspace/CEJ_ABC/.venv/bin/python \
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
  --num-workers 4 \
  --case-prefix auto
```

Expected outputs:

```bash
$PRE_RUN/manual_points_usage_report.json
$SUP_RUN/outputs/train/checkpoints/last.pt
$SUP_RUN/outputs/train/metrics.csv
$SUP_RUN/outputs/infer
$SUP_RUN/outputs/viz/3d/index.html
$SUP_RUN/outputs/eval
```

Backup new checkpoint after training:

```bash
mkdir -p /root/backup_models
tar -czf /root/backup_models/run_sup_inc_${RUN_TAG}_last_checkpoint_$(date +%Y%m%d).tar.gz \
  -C "$SUP_RUN/outputs/train/checkpoints" last.pt
```

## Resume / Incremental Rules

- Use a new `RUN_TAG` for each new uploaded batch to avoid overwriting old models.
- Use the previous best checkpoint as `--pretrained-ckpt`; do not overwrite prior checkpoint dirs.
- The pipeline reruns preprocessing for cases whose source upload changed, and skips unchanged completed cases in that run root.
- To ignore a bad JSON file, create `cej_ignore_files.txt` inside that manual case folder with one file name per line.
- Do not use the old accidental merged run roots unless explicitly requested:
  `/root/cej_runs/run_manual_merged_041055_001`
  `/root/cej_runs/run_sup_manual_merged_041055_001`

## Quick Checks After A Run

```bash
python scripts/check_manual_points_usage.py \
  --manual-root "$MANUAL_ROOT" \
  --toothfairy-root "$TF_ROOT" \
  --processed-root "$PRE_RUN/processed" \
  --report-csv "$PRE_RUN/manual_points_usage_report.csv" \
  --report-json "$PRE_RUN/manual_points_usage_report.json" \
  --case-prefix auto

python - <<'PY'
import json, os
p=os.environ.get('PRE_RUN','') + '/manual_points_usage_report.json'
with open(p) as f:
    r=json.load(f)
print(r['summary'])
PY

tail -n 5 "$SUP_RUN/outputs/train/metrics.csv"
ls -lh "$SUP_RUN/outputs/train/checkpoints/last.pt"
ls -lh "$SUP_RUN/outputs/viz/3d/index.html"
```

## Known Quality Notes

- Current model is useful for iteration but not final clinical use.
- Last quick training-set check on 41-55 fitted curves: mean GT-to-pred distance about `0.785 mm`, p95 about `1.104 mm`, SR@1.0mm about `88.9%`.
- This is training-set-only and optimistic. Next improvement should add patient-level holdout eval.
- Training currently includes all tooth dirs from selected cases. If labels are sparse by tooth, consider changing supervised dataset to train only teeth with manual points/H_GT non-empty.
- Main current bottleneck is supervised label semantics and post-processing, not more unsupervised pretraining.
