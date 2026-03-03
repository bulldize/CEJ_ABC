# Progress

## Current Snapshot (2026-03-03)

- Branch: `codex/manual-points-group-col`
- Latest commit (before this doc sync): `8e1667c`
- Runtime context: cloud server + ToothFairy3 full dataset + incremental manual CEJ uploads.

## Completed Milestones

- Completed: ToothFairy3 batch preprocess resume (`scripts/run_preprocess_resume.py`), with case-level skip for already processed outputs.
- Completed: Manual annotation preprocess resume (`scripts/run_manual_preprocess_resume.py`) with:
  - source-change rerun (`--rerun-on-source-update`)
  - forced overwrite for selected cases (`--force-cases`)
- Completed: Manual Excel `group` column compatibility in CEJ mark conversion (`src/datasets/mark_points.py`).
- Completed: Unsupervised pretrain resume support (`src/pretrain_mae.py`):
  - auto-resume from `checkpoints/last.pt`
  - optimizer state restore
  - metrics append mode
- Completed: Unsupervised run reached epoch 120:
  - metrics: `/root/cej_runs/run_unsup_001/unsup/pretrain/metrics.csv`
  - checkpoint: `/root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt`
- Completed: Manual-point usage gate script:
  - `scripts/check_manual_points_usage.py`
  - verifies Excel valid rows vs processed kept points (+ per-tooth mismatch diagnostics)
- Completed: One-shot automation pipeline for continuous manual uploads:
  - `scripts/run_manual_supervised_pipeline.py`
  - flow: preprocess overwrite -> usage gate -> supervised train -> infer -> viz -> eval

## Latest Training Status

### Manual Data Coverage

- Manual source root: `/root/手工标注1`
- Cases discovered: `008, 009, 010, 018, 021, 023`
- Latest coverage check: all pass (`all_used=true`, 6/6)
- Report:
  - `/root/cej_runs/run_unsup_001/manual_points_usage_report.csv`
  - `/root/cej_runs/run_unsup_001/manual_points_usage_report.json`

### Supervised Fine-tune (Latest)

- Run root: `/root/cej_runs/run_sup_manual6_002`
- Init checkpoint: `/root/cej_runs/run_sup_manual6_001/outputs/train/checkpoints/last.pt`
- Config: `/root/workspace/CEJ_ABC/configs/server_sup_manual_auto.yaml`
- Train result:
  - epochs: 20
  - loss: `0.4984 -> 0.4520`
  - checkpoint: `/root/cej_runs/run_sup_manual6_002/outputs/train/checkpoints/last.pt`
- Eval summary:
  - mean: `4.0264 mm`
  - p95: `8.3479 mm`
  - sr@1.0: `0.1112`
  - sr@1.5: `0.1811`
- 3D viz:
  - `/root/cej_runs/run_sup_manual6_002/outputs/viz/3d/index.html`

## Validation Record

- Passed: manual point usage gate for all 6 manual cases (`PASS_FULLY_USED`).
- Passed: supervised retrain on updated labels (`run_sup_manual6_002`).
- Passed: infer/viz/eval generation for all six cases with updated labels included (including TF21 tooth_37 and tooth_46).

## Next Actions

1. Add explicit train/val/test split config for manual-labeled small sets (currently train/eval are same processed set).
2. Add checkpoint retention policy for unsupervised pretrain (e.g., keep every 20 epochs in addition to `last.pt`).
3. Continue incremental manual-upload cycles using `scripts/run_manual_supervised_pipeline.py` as default entrypoint.
