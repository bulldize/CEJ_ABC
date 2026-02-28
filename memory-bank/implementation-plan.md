# Implementation Plan

## Baseline (Done)

1. Completed: End-to-end supervised CEJ pipeline (preprocess/train/infer/eval/viz) with MONAI runtime package.
2. Completed: Raw data auto-discovery for `case_dirs` and ToothFairy3 `imagesTr/labelsTr`.
3. Completed: Interactive 3D visualization workflow with fixed process legend and configurable colors.
4. Completed: Full-mouth pseudo-GT compare export pipeline for medical/QC/model review.
5. Completed: Thin-curve default export + consistency gate (IoU/Dice) with fail-fast behavior.
6. Completed: Minimal export retention policy (only 3 compare NIfTI files + `export_meta.json` per case).

## Current Focus

1. In progress: Keep docs synchronized with current runtime/export contract (`README`, `memory-bank`, `doc/specs`).

## Next Milestones

1. Pending: Non-smoke training and evaluation baseline on real ToothFairy3 batch to improve `CEJ_model_compare_full.nii.gz` quality.
2. Pending: Publish Colab-ready batch preprocess/infer instructions for ToothFairy3 structure.
3. Optional: Add overlapping-segment export format (e.g., `.seg.nrrd`) if medical reviewers require independent visibility of fully overlapping curves.
