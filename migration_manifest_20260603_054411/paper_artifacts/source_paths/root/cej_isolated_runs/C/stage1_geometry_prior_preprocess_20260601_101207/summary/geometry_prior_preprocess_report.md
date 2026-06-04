# Stage1 geometry-prior preprocessing

- Run root: `/root/cej_isolated_runs/C/stage1_geometry_prior_preprocess_20260601_101207`
- Selected threshold: `0.35`
- Status: `FAIL`

| Metric | Value | Goal | Pass |
| --- | ---: | ---: | --- |
| mean_dist_mm | 2.868568 | baseline 0.628 | - |
| p95_dist_mm | 4.035110 | <= 1.467 | False |
| sr@1.0mm | 0.563922 | >= 0.836 | False |
| failed/bad teeth | 25 | <= 3 | False |
| missing predictions | 0 | 0 | True |
| no-curve teeth | 2 | 0 | False |
| holdout teeth | 88 | 88 | True |

## Artifacts

- Train config: `/root/cej_isolated_runs/C/stage1_geometry_prior_preprocess_20260601_101207/config/train.yaml`
- Sweep CSV: `/root/cej_isolated_runs/C/stage1_geometry_prior_preprocess_20260601_101207/threshold_sweep_best/threshold_sweep_best.csv`
- Final metrics: `/root/cej_isolated_runs/C/stage1_geometry_prior_preprocess_20260601_101207/final_best/eval/metrics_summary.json`
- Failed/bad teeth: `/root/cej_isolated_runs/C/stage1_geometry_prior_preprocess_20260601_101207/final_best/eval/failed_or_bad_teeth.csv`
