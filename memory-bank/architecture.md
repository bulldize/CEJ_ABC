# Architecture

## Core Python Modules

- `src/datasets/*`: raw IO, ROI crop/stitch, CEJ points conversion, heatmap generation, MONAI dataset/transforms.
- `src/datasets/raw_cases.py`: unified raw case discovery for `raw_layout: auto|case_dirs|toothfairy3`.
- `src/datasets/mark_points.py`: Excel/mark-space CEJ point conversion (supports `牙位` / `tooth` / `group` columns).
- `src/models/unet3d.py`: MONAI-based 3D UNet wrapper.
- `src/preprocess.py`: supervised preprocess for tooth ROIs + pseudo-GT heatmap generation.
- `src/pretrain_mae.py`: unsupervised masked-reconstruction pretrain with auto-resume.
- `src/train.py`: supervised training loop, pretrained loading, checkpoint/metrics logging.
- `src/infer.py`: per-tooth inference, postprocess, full-volume stitch-back.
- `src/eval.py`: CEJ point-to-curve distance metrics (Mean/P95/SR@tau).
- `src/viz.py`: interactive 3D viewer generation (MPR + surface + heatmaps + curves + process legend).

## Workflow Scripts (Operational Layer)

- `scripts/run_preprocess_resume.py`: ToothFairy3 batch preprocess resume with case-level skip.
- `scripts/run_manual_preprocess_resume.py`: manual annotation incremental overwrite:
  - skip completed cases
  - rerun on source update
  - force rerun selected cases (`--force-cases`)
- `scripts/check_manual_points_usage.py`: quality gate between manual Excel rows and processed points.
- `scripts/run_manual_supervised_pipeline.py`: one-shot cloud workflow:
  - manual preprocess overwrite
  - usage quality gate
  - supervised train/infer/viz/eval

## Data and Coordinate Contracts

- Canonical world frame: `A.nii.gz` affine (`qform/sform`).
- External CEJ annotations are aligned to A-world space before voxel conversion.
- Per-tooth processed outputs keep ROI metadata (`roi_origin_in_full`, shape, spacing, affine mapping hints).
- NIfTI save path preserves affine/qform/sform consistency for 3D Slicer interoperability.

## Pseudo-GT Consistency Path

- Generation path is fixed as:
  - manual points -> interpolated dense curve -> pseudo-GT heatmap -> pseudo-GT skeleton
- Skeleton is derived from heatmap peak extraction by default (not copied from interpolation).
- Export stage computes overlap metrics between interpolated and skeleton curves:
  - thresholds: IoU >= 0.95, Dice >= 0.97
  - behavior: fail-fast when any tooth violates thresholds.

## Manual Upload Runtime Contract

1. New manual uploads are placed under `/root/手工标注1`.
2. Preprocess overwrite uses source metadata (`size`, `mtime_ns`) to detect changed files.
3. Usage gate must pass (`all_used=true`) before launching supervised fine-tune.
4. Supervised run consumes only validated `PASS` cases via symlinked subset directory.
5. Infer/viz/eval are generated under the supervised run root for traceability.

## Compare Export Contract (Per Case)

- `CEJ_medical_compare_full.nii.gz`
  - `1=tooth_body`, `2=manual_points`, `3=pseudo_gt_interpolated_curve`
- `CEJ_qc_compare_full.nii.gz`
  - `1=tooth_body`, `2=pseudo_gt_interpolated_curve`, `3=pseudo_gt_skeleton`
- `CEJ_model_compare_full.nii.gz`
  - `1=tooth_body`, `2=pseudo_gt_interpolated_curve`, `3=predicted_curve`
- `export_meta.json`
  - includes compare definitions, voxel stats, and per-tooth consistency metrics.
