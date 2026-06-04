# Loss-Only Geometry Prior Delivery Package

Source run root: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_only_20260601_110722`

This package contains the loss-only Stage1 experiment artifacts with original GT preserved.

## Key Results

- Selected threshold: `0.25`
- Status: `FAIL`
- mean_dist_mm: `0.6476492881774902`
- p95_dist_mm: `1.6594260096549975`
- sr@1.0mm: `0.8627450980392157`
- failed/bad teeth: `5`
- missing predictions: `0`
- no-curve teeth: `0`
- GT preserved: `True`
- GT mismatch count: `0`

## Main Entry Points

- 3D viewer index: `visualization/3d/index.html`
- Model checkpoint: `model/best.pt`
- Inference config: `config/final_best.yaml`
- Viz config: `config/viz_package.yaml`
- Final inference outputs: `inference/final_best/infer/`
- Final eval metrics: `evaluation/final_best/eval/metrics_summary.json`
- Failed/bad teeth list: `evaluation/final_best/eval/failed_or_bad_teeth.csv`
- GT preservation audit: `reports/gt_preservation_audit.json`

## Notes

The package intentionally excludes train16 processed data and 2D PNG visualizations.
It includes processed holdout inputs so inference/eval/viz can be reproduced on the fixed holdout split.
