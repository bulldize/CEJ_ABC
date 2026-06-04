# Stage1 geometry-prior preprocessing plus weak shape-prior loss

- Run root: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_20260601_101405`
- Selected threshold: `0.25`
- Status: `FAIL`

| Metric | Value | Goal | Pass |
| --- | ---: | ---: | --- |
| mean_dist_mm | 3.504990 | baseline 0.628 | - |
| p95_dist_mm | 18.751021 | <= 1.467 | False |
| sr@1.0mm | 0.617255 | >= 0.836 | False |
| failed/bad teeth | 24 | <= 3 | False |
| missing predictions | 0 | 0 | True |
| no-curve teeth | 3 | 0 | False |
| holdout teeth | 88 | 88 | True |

## Artifacts

- Train config: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_20260601_101405/config/train.yaml`
- Sweep CSV: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_20260601_101405/threshold_sweep_best/threshold_sweep_best.csv`
- Final metrics: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_20260601_101405/final_best/eval/metrics_summary.json`
- Failed/bad teeth: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_20260601_101405/final_best/eval/failed_or_bad_teeth.csv`
