# Architecture

## Core Modules

- `src/datasets/*`: raw IO, ROI crop/stitch, CEJ points conversion, heatmap generation, MONAI dataset/transforms.
- `src/datasets/raw_cases.py`: unified raw case discovery for `raw_layout: auto|case_dirs|toothfairy3`.
- `src/models/unet3d.py`: MONAI-based 3D UNet wrapper.
- `src/train.py`: supervised training loop, checkpointing, metrics logging.
- `src/infer.py`: per-tooth inference, postprocess, full-volume stitch-back.
- `src/eval.py`: CEJ point-to-curve distance metrics (Mean/P95/SR@tau).
- `src/viz.py`: interactive 3D viewer generation (MPR + surface + heatmaps + curves + process legend).
- `src/export_pseudo_gt_full.py`: full-mouth compare export for medical/QC/model review and consistency gating.
- `src/postprocess/skeleton.py`: heatmap-to-curve extraction and overlap consistency metrics (IoU/Dice).

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

## Compare Export Contract (Per Case)

- `CEJ_medical_compare_full.nii.gz`
  - `1=tooth_body`, `2=manual_points`, `3=pseudo_gt_interpolated_curve`
- `CEJ_qc_compare_full.nii.gz`
  - `1=tooth_body`, `2=pseudo_gt_interpolated_curve`, `3=pseudo_gt_skeleton`
- `CEJ_model_compare_full.nii.gz`
  - `1=tooth_body`, `2=pseudo_gt_interpolated_curve`, `3=predicted_curve`
- `export_meta.json`
  - includes compare definitions, voxel stats, and per-tooth consistency metrics.

## Visualization Contract

- Main entry: `outputs/viz/3d/index.html`.
- Each viewer includes fixed process legend rows:
  - `1 标注点`
  - `2 插值曲线`
  - `3 伪GT热图`
  - `4 伪GT骨架`
  - `5 推理热图` (if available)
  - `6 推理曲线` (if available)
- Predicted curve default color is red (`#FF1744`) for clearer contrast.
