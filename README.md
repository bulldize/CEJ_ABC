# CEJ Curve Prediction MVP (A1+A2)

End-to-end, single-tooth ROI pipeline for CEJ curve prediction with a runnable MVP skeleton.
This repo follows `doc/specs/PRD.md` and the tech-route spec under `doc/specs/` strictly.

## Quick Start

```bash
# optional: create a venv and install deps
pip install -r requirements.txt

# run the full pipeline (preprocess -> train -> infer -> eval -> viz)
bash scripts/run_all.sh

# regenerate interactive 3D viewers only
python -m src.viz --config configs/default.yaml
```

`scripts/run_all.sh` auto-detects raw layout:
- case-directory layout: if `data/raw_tf008/` is empty, it prepares TF_008 from `data/TF_008` (no synthetic data)
- ToothFairy3 layout (`imagesTr/labelsTr`): it skips TF_008 auto-prepare and uses the existing dataset directly
The script is compatible with macOS default Bash (3.2), no `readarray` dependency.
Matplotlib cache is stored under `outputs/.mpl_cache` to avoid permission warnings.
3D viewer index is written to `outputs/viz/3d/index.html` (Slicer-like MPR + surface overlays).

## Unsupervised Pretrain (Masked Reconstruction)

MVP skeleton for MAE-style masked reconstruction pretrain on tooth ROIs.

```bash
# preprocess TF_008 into unsupervised ROI dataset
python -m src.preprocess_unsup --config configs/unsup_mae.yaml

# run masked reconstruction pretrain
python -m src.pretrain_mae --config configs/unsup_mae.yaml

# or use one-shot script
bash scripts/run_all_unsup.sh
```

Notes:
- Uses ROI/牙体 mask by default; set `unsup.use_label_mask: false` to fall back to full-volume ROI.
- Pretrain output checkpoint: `outputs/unsup/pretrain/checkpoints/last.pt`
- To finetune, enable `finetune.enable: true` in `configs/unsup_mae.yaml` or pass `--pretrained` to `src.train`.

## Data Layout

### Raw (Case-Directory Layout)

```
data/raw_tf008/{case_id}/
  A.nii.gz            # CBCT volume, float32
  B.nii.gz            # segmentation, int labels
  cej_points_ras.xlsx # CEJ points in mark (RAS) space (per-case)
  points.json         # sparse CEJ points per tooth
  meta.json           # required only for .npy inputs
```

`points.json` (raw, full-volume coordinates):

```json
{
  "case_id": "case_0001",
  "coord_type": "voxel",
  "space": "full",
  "points": {
    "11": [[x,y,z], [x,y,z]],
    "12": [[x,y,z]]
  }
}
```

If points are in physical space, set `coord_type: "world"` and provide `affine` in `meta.json`.
If `cej_points_ras.xlsx` exists, preprocess will convert it to `points.json` (voxel/full) and write `mark_meta.json`.
If the per-case `cej_points_ras.xlsx` is missing, preprocess will look for `data/cej_points_ras.xlsx`.
If no CEJ file is found, it will write an empty `points.json` and treat the case as unsupervised.

Global mark boundary (single file):
```
data/scan_boundary_ras.xlsx
```

### Raw (ToothFairy3 `imagesTr/labelsTr` Layout)

Also supported directly (no manual reorganization required):

```
{raw_dir}/
  imagesTr/
    ToothFairy3F_001_0000.nii.gz
    ToothFairy3P_545_0000.nii.gz
    ...
  labelsTr/
    ToothFairy3F_001.nii.gz
    ToothFairy3P_545.nii.gz
    ...
  pointsTr/                 # optional
    ToothFairy3F_001.json
    ToothFairy3P_545.json
```

Matching rule:
- image case id is inferred by dropping `_0000` from image filename.
- label filename must match `{case_id}.nii.gz` (or `.nii`).
- when `raw_layout: auto`, this layout is selected automatically if both `imagesTr/` and `labelsTr/` exist.

Set in config:

```yaml
data:
  raw_layout: toothfairy3   # or auto
  raw_dir: /path/to/ToothFairy3
  images_tr_dir: /path/to/ToothFairy3/imagesTr
  labels_tr_dir: /path/to/ToothFairy3/labelsTr
  points_dir: /path/to/ToothFairy3/pointsTr   # optional
  max_cases: 2                                  # optional smoke limit
```

### Processed

```
data/processed/{case_id}/tooth_{tooth_id}/
  A_t.nii.gz
  T_t.nii.gz
  H_GT.nii.gz
  points.json
  roi_meta.json
  curve_dense_points.npy
```

`roi_meta.json` includes ROI origin, shape, spacing, and affine for ROI <-> full mapping.
If `points.json` is empty or missing, `H_GT.nii.gz` is still generated as an all-zero heatmap for smoke tests, and
`curve_dense_points.npy` will be empty.
See `schemas/` for JSON schema.

## Outputs

```
outputs/
  preprocess/
  train/
    checkpoints/
    metrics.csv
  infer/
    {case_id}/
      tooth_{tooth_id}/
        H_pred.nii.gz
        C_pred.nii.gz
      Y_pred.nii.gz
      H_pred.nii.gz
  eval/
    metrics_summary.json
    metrics_per_tooth.csv
  viz/
    3d/
      index.html
      {case_id}/tooth_{tooth_id}/viewer.html
    # optional 2D outputs (when viz.enable_2d=true)
    roi/
    prior/
    pseudo_gt/
    infer/
    error/
```

## Key Assumptions

- Array axis order is (X, Y, Z) with voxel indices [x, y, z].
- Distances are computed in mm using spacing.

## Updates

- 2026-03-01: `scripts/run_all.sh` is now Bash 3.2 compatible and honors custom `raw_a_name/raw_b_name` during TF_008 auto-prepare.
- 2026-03-01: `src.preprocess`/`src.infer` now use unified raw-case discovery (`raw_layout: auto/case_dirs/toothfairy3`) with optional `max_cases`.
- 2026-03-01: Removed in-repo `MONAI/` source checkout; project now uses installed `monai` package from `requirements.txt`.
- 2026-02-11: Supervised pipeline `scripts/run_all.sh` fully passes on TF_008 after MONAI integration.
- 2026-02-11: Default raw dir is `data/raw_tf008`, with TF_008 auto-prepare from `data/TF_008`.
- 2026-02-11: Unsupervised pipeline `scripts/run_all_unsup.sh` passes on TF_008.
- A1+A2 only: no R_t in training loss; optional gating in inference only.

## Commands

```bash
# preprocess only
python -m src.preprocess --config configs/default.yaml

# train only
python -m src.train --config configs/default.yaml

# infer only
python -m src.infer --config configs/default.yaml

# eval only
python -m src.eval --config configs/default.yaml

# viz only
python -m src.viz --config configs/default.yaml

# export full-mouth pseudo-GT NIfTI for medical review in 3D Slicer
python -m src.export_pseudo_gt_full --config configs/default.yaml

# real ToothFairy3 smoke config (imagesTr/labelsTr)
python -m src.preprocess --config configs/toothfairy3_real_smoke.yaml
python -m src.train --config configs/toothfairy3_real_smoke.yaml
python -m src.infer --config configs/toothfairy3_real_smoke.yaml
python -m src.eval --config configs/toothfairy3_real_smoke.yaml
python -m src.viz --config configs/toothfairy3_real_smoke.yaml
```

## Notes

- For .npy inputs, provide `meta.json` with `spacing` and `affine`.
- `preprocess.resample_to_target` is off by default; enable it to resample to `target_spacing_mm`.
- Dense curve fitting uses closed-loop interpolation by default (`preprocess.curve_closed=true`) for CEJ ring stability.
- The default config uses placeholder values from the PRD and tech-route spec.
- `src.viz` defaults to interactive 3D HTML output (`viz.enable_3d=true`) and keeps 2D overlays optional (`viz.enable_2d=false`).
- 3D viewer legend is in Chinese with fixed category colors for quick reading: 推理曲线/推理热图/标注点/伪GT热图/伪GT骨架.
- Interpolated dense pseudo-GT curve is also visible in 3D viewer as `插值曲线`, and exported for Slicer as `C_pseudo_gt_interp_curve_full.nii.gz`.
- 默认开启“流程一致”而非强制覆盖：热力图由插值曲线连续栅格化生成，骨架由热力图峰值（`>=0.999`）反提，保证 1→2→3 一致来源。
- 为避免 3D Slicer 表面渲染出现重合闪烁（z-fighting），默认导出非重叠环层：`2=内层骨架加粗`，`3=外层插值加粗壳层`。
- Slicer review should use `Y_pseudo_gt_review_slicer_full.nii.gz` (int16 labelmap): `1=牙体`, `2=伪GT骨架(加粗内层)`, `3=插值曲线(加粗外层)`。
- 若需同时看手工点，使用 `Y_pseudo_gt_review_with_points_full.nii.gz` 或单独加载 `P_manual_points_tube_full.nii.gz`。
- Export also provides separate masks: `C_pseudo_gt_skeleton_tube_full.nii.gz`, `C_pseudo_gt_interp_curve_tube_full.nii.gz`, `P_manual_points_tube_full.nii.gz`.
- NIfTI export now writes consistent qform/sform from source affine for better external-tool orientation consistency.
- Coordinate convention: all external annotations are aligned to the world coordinate system defined by `A.nii.gz` sform/qform (affine) before converting to voxel space.
- Preprocess auto-detects CEJ point coordinates as `world_ras` or `world_lps` using the A volume affine and records the choice in `mark_meta.json`.
- TODO: confirm external software workflow for direct curve/point import; current integration exports review-ready NIfTI labelmaps/masks.
