# Progress

## Current Snapshot (2026-03-01)

- Branch: `codex/cej-thin-slicer-consistency`
- Latest status: default smoke pipeline passes (`run_all.sh`) and unit tests pass (`pytest -q`).
- Export contract is now minimal and stable:
  - `CEJ_medical_compare_full.nii.gz`
  - `CEJ_qc_compare_full.nii.gz`
  - `CEJ_model_compare_full.nii.gz`
  - `export_meta.json`
- Per-case export folders are auto-cleaned; only the four files above are kept.

## Completed Milestones

- Completed: Unified raw case discovery for both layouts (`case_dirs` and `toothfairy3`) with `raw_layout: auto`.
- Completed: macOS Bash 3.2 compatibility for `scripts/run_all.sh` (no `readarray` dependency).
- Completed: 3D viewer default path and workflow finalized (`outputs/viz/3d/index.html`).
- Completed: 3D viewer fixed process legend and stronger predicted-curve color (`#FF1744`) for visibility.
- Completed: Pseudo-GT processing chain locked as non-overwrite consistency flow:
  - manual points -> interpolated curve -> heatmap -> pseudo-GT skeleton
- Completed: Export consistency gate enabled by default:
  - IoU >= 0.95
  - Dice >= 0.97
  - fail-fast on violation (`SystemExit(2)`).
- Completed: Slicer default switched to thin-curve mode (tube radii all `0.0`, no thick shell by default).
- Completed: Compare file segment semantics aligned for medical review, QC, and model evaluation:
  - medical: `1=tooth`, `2=manual points`, `3=interpolated curve`
  - qc: `1=tooth`, `2=interpolated curve`, `3=pseudo-GT skeleton`
  - model: `1=tooth`, `2=interpolated curve`, `3=predicted curve`

## Validation Record

- Passed: `pytest -q` (includes thin-curve export, consistency, and viz legend checks).
- Passed: `bash scripts/run_all.sh` with `configs/default.yaml`.
- Passed: `python -m src.export_pseudo_gt_full --config configs/default.yaml`.
- Output verified under `outputs/pseudo_gt_review_nifti/<case_id>/` with the 3 compare NIfTI files + `export_meta.json`.

## Known Notes

- In `CEJ_qc_compare_full.nii.gz`, when interpolated curve and skeleton fully overlap, label `3` can visually hide label `2` at those voxels (single-label voxel format); this is expected.
- `CEJ_model_compare_full.nii.gz` segment `3` quality depends on training maturity; smoke-level training is not clinically meaningful yet.

## Next Actions

- Next: Run non-smoke training schedule and regenerate model compare volumes for meaningful CEJ prediction quality.
- Next: Prepare Colab-oriented ToothFairy3 batch preprocess/train/infer execution note (if batch run is prioritized).
- Optional: If clinicians need simultaneously visible overlapping segments, add `.seg.nrrd` export alongside current NIfTI labelmaps.
