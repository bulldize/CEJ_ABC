# Stage1 geometry-prior weak loss only, original GT preserved

- Run root: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_only_20260601_110722`
- Selected threshold: `0.25`
- Status: `FAIL`

| Metric | Value | Goal | Pass |
| --- | ---: | ---: | --- |
| mean_dist_mm | 0.647649 | baseline 0.628 | - |
| p95_dist_mm | 1.659426 | <= 1.467 | False |
| sr@1.0mm | 0.862745 | >= 0.836 | True |
| failed/bad teeth | 5 | <= 3 | False |
| missing predictions | 0 | 0 | True |
| no-curve teeth | 0 | 0 | True |
| holdout teeth | 88 | 88 | True |
| GT preserved | True | True | True |

## Artifacts

- Train config: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_only_20260601_110722/config/train.yaml`
- Sweep CSV: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_only_20260601_110722/threshold_sweep_best/threshold_sweep_best.csv`
- Final metrics: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_only_20260601_110722/final_best/eval/metrics_summary.json`
- Failed/bad teeth: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_only_20260601_110722/final_best/eval/failed_or_bad_teeth.csv`
- GT preservation audit: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_only_20260601_110722/input/gt_preservation_audit.json`
