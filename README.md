# CEJ Curve Prediction MVP (A1+A2)

End-to-end, single-tooth ROI pipeline for CEJ curve prediction with a runnable MVP skeleton.
This repo follows `doc/specs/PRD.md` and the tech-route spec under `doc/specs/` strictly.

## Quick Start

```bash
# optional: create a venv and install deps
pip install -r requirements.txt

# run the full pipeline (preprocess -> train -> infer -> eval -> viz)
bash scripts/run_all.sh
```

If `data/raw_tf008/` is empty, the script will prepare a TF_008 case from `data/TF_008` (no synthetic data).
Matplotlib cache is stored under `outputs/.mpl_cache` to avoid permission warnings.

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

### Raw

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
```

## Notes

- For .npy inputs, provide `meta.json` with `spacing` and `affine`.
- `preprocess.resample_to_target` is off by default; enable it to resample to `target_spacing_mm`.
- The default config uses placeholder values from the PRD and tech-route spec.
- Coordinate convention: all external annotations are aligned to the world coordinate system defined by `A.nii.gz` sform/qform (affine) before converting to voxel space.
- Preprocess auto-detects CEJ point coordinates as `world_ras` or `world_lps` using the A volume affine and records the choice in `mark_meta.json`.
- TODO: confirm external software import format and add export option for points/curves.
