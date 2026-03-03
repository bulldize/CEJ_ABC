# Implementation Plan

## Baseline (Done)

1. Completed: End-to-end supervised CEJ pipeline (preprocess/train/infer/eval/viz) with MONAI runtime package.
2. Completed: Raw data auto-discovery for `case_dirs` and ToothFairy3 `imagesTr/labelsTr`.
3. Completed: Interactive 3D visualization workflow with fixed process legend and per-case tooth viewers.
4. Completed: Unsupervised pretrain MVP with resume support (`src.pretrain_mae`).
5. Completed: Manual annotation incremental preprocess pipeline:
   - resume + skip completed
   - rerun on source update
   - force rerun selected cases
6. Completed: Manual point usage gate (`check_manual_points_usage.py`) to ensure Excel points are truly consumed by processed samples.
7. Completed: One-shot automation pipeline for continuous upload scenarios (`run_manual_supervised_pipeline.py`).

## Current Focus

1. In progress: Keep docs/specs/memory synchronized with the cloud execution workflow and latest supervised baseline.
2. In progress: Use updated manual labels as default supervised fine-tune source and regenerate infer/viz/eval artifacts per upload cycle.

## Next Milestones

1. Pending: Add explicit `train/val/test` split support for manual-labeled datasets (current flow evaluates on the same processed pool).
2. Pending: Add epoch-interval checkpoint retention for unsupervised pretrain (e.g., keep `epoch_020.pt`, `epoch_040.pt`, ... + `last.pt`).
3. Pending: Add lightweight experiment tracker summary (run ID, init ckpt, epochs, final loss, eval summary, artifact paths) to reduce manual bookkeeping.
4. Optional: Add CI smoke for manual usage gate script and forced-rerun path.
