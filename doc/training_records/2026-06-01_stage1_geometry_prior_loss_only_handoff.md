# Stage1 Geometry Prior Loss-Only Handoff

Date: 2026-06-01

## Decision

Do not merge the current geometry-prior loss into the main Stage1 path.

The loss-only variant preserves the original GT and is much safer than the GT-upgrade variants, but it still misses the conservative non-inferiority gate:

- `p95_dist_mm` is above the goal.
- `failed/bad teeth` is above the goal.
- `sr@1.0mm` improves versus baseline, but that is not enough for adoption.

Keep this as an ablation artifact. The current mainline should remain Stage1-only `exp03_skeleton_aux_loss` without the geometry-prior loss.

## Fixed Split

Holdout cases:

- `ToothFairy3F_040`
- `ToothFairy3F_044`
- `ToothFairy3F_052`

Holdout tooth dirs: 88.

Baseline Stage1-only:

| Experiment | Threshold | mean_dist_mm | p95_dist_mm | sr@1.0mm | failed/bad | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline Stage1-only | 0.25 | 0.628 | 1.417 | 0.846 | 3 | baseline |

Conservative non-inferiority gate:

- `p95_dist_mm <= 1.467`
- `sr@1.0mm >= 0.836`
- `failed_or_bad_teeth_count <= 3`
- `missing_prediction_count == 0`
- `no_curve_count == 0`
- `holdout_tooth_count == 88`

## Experiment Results

| Experiment | Run Root | Threshold | mean_dist_mm | p95_dist_mm | sr@1.0mm | failed/bad | missing | no_curve | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp1 GT preprocess prior | `/root/cej_isolated_runs/C/stage1_geometry_prior_preprocess_20260601_101207` | 0.35 | 2.869 | 4.035 | 0.564 | 25 | 0 | 2 | FAIL |
| exp2 GT preprocess + weak loss | `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_20260601_101405` | 0.25 | 3.505 | 18.751 | 0.617 | 24 | 0 | 3 | FAIL |
| loss-only prior, GT preserved | `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_only_20260601_110722` | 0.25 | 0.648 | 1.659 | 0.863 | 5 | 0 | 0 | FAIL |

Loss-only threshold sweep:

| Threshold | mean_dist_mm | p95_dist_mm | sr@1.0mm | failed/bad | Status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.20 | 0.633 | 1.645 | 0.857 | 6 | FAIL |
| 0.25 | 0.648 | 1.659 | 0.863 | 5 | FAIL |
| 0.30 | 0.636 | 1.590 | 0.864 | 6 | FAIL |
| 0.35 | 0.623 | 1.525 | 0.869 | 6 | FAIL |

## Loss-Only Implementation

Runner:

```bash
python scripts/run_stage1_geometry_prior_loss_only_exp.py --setup-only
python scripts/run_stage1_geometry_prior_loss_only_exp.py --run-root <run_root> --resume
```

Key behavior:

- Copies the fixed 16 train / 3 holdout processed split into an isolated run root.
- Does not modify `curve_dense_points.npy`, `H_GT.nii.gz`, or `C_GT.nii.gz`.
- Generates only loss-side prior files:
  - `H_SHAPE_PRIOR.nii.gz`
  - `shape_prior_dense_points.npy`
  - `loss_shape_geometry_prior.json`
  - `shape_prior_fit_report.json`
- Enables `train.use_shape_prior_channel_or_loss: true`.
- Uses `lambda_shape_prior: 0.02` and `shape_prior_sigma_mm: 2.0`.
- Adds loss term:

```text
lambda_shape_prior * mean(sigmoid(logits)^2 * (1 - shape_prior))
```

GT preservation audit for the completed loss-only run:

- Path: `/root/cej_isolated_runs/C/stage1_geometry_prior_loss_only_20260601_110722/input/gt_preservation_audit.json`
- `gt_preserved: true`
- `mismatch_count: 0`

Shape-prior generation audit:

- train labeled teeth: 426 / 426
- holdout labeled teeth: 65 / 65
- generation errors: 0

## Delivery Package

Package directory:

```text
/root/cej_isolated_runs/C/stage1_geometry_prior_loss_only_20260601_110722/deliverables/loss_only_viz_infer_package_20260601_114314
```

Compressed package:

```text
/root/cej_isolated_runs/C/stage1_geometry_prior_loss_only_20260601_110722/deliverables/loss_only_viz_infer_package_20260601_114314.tar.gz
```

Tar SHA256:

```text
abc6aeb22ed097978011090a2540b9588b314ccc12b7336186fcf9591ef69634
```

Package contents:

- `model/best.pt`
- `model/last.pt`
- `model/unsup_last.pt`
- `config/train.yaml`
- `config/final_best.yaml`
- `config/viz_package.yaml`
- `inference/final_best/infer/`
- `evaluation/final_best/eval/`
- `evaluation/threshold_sweep_best.csv`
- `visualization/3d/index.html`
- 88 per-tooth 3D HTML viewers
- `input/processed_holdout3/`
- `reports/`
- `logs/`
- `SHA256SUMS.txt`

The package intentionally excludes:

- train16 processed data
- 2D PNG visualization outputs
- multi-threshold prediction volumes

The 3D viewer includes the loss-only `H_SHAPE_PRIOR` as a `Loss shape prior` layer.

## Verification

Static and focused tests run:

```bash
python -m py_compile scripts/stage1_geometry_prior_common.py scripts/run_stage1_geometry_prior_loss_only_exp.py src/datasets/dataset.py src/train_loss_earlystop.py src/train_holdout.py src/viz.py
pytest -q tests/test_supervised_dataset.py tests/test_train_loss_sampler.py tests/test_cej_geometry.py tests/test_viz_legend_presence.py
```

Result:

- pytest: 9 passed
- real-data smoke test loaded `shape_prior` with label-matching shape and computed the shape-prior loss term.

Training result:

- early stopped at epoch 23
- best epoch 15
- selected threshold 0.25

## Files Changed For This Handoff

- `scripts/run_stage1_geometry_prior_loss_only_exp.py`
- `scripts/run_stage1_geometry_prior_loss_exp.py`
- `scripts/run_stage1_geometry_prior_preprocess_exp.py`
- `scripts/stage1_geometry_prior_common.py`
- `scripts/upgrade_processed_geometry_gt.py`
- `src/datasets/dataset.py`
- `src/train_loss_earlystop.py`
- `src/train_holdout.py`
- `src/viz.py`
- `tests/test_supervised_dataset.py`

Related Stage1 experiment utilities in this commit:

- `scripts/run_goal_mode_16train_3holdout.py`
- `scripts/run_stage1_16train_direct_eval.py`
- `scripts/run_stage1_data_learning_curve.py`
- `scripts/run_stage1_full19_direct_eval.py`
- `STAGE1_ONLY_EXPERIMENT_CONCLUSION.md`

## Next Steps

1. Keep geometry prior loss out of the default Stage1 path.
2. Use the package above for review of failure cases, especially `ToothFairy3F_052` teeth 31, 32, 41, and 42.
3. If this idea is revisited, tune the prior loss strength against p95 and failed/bad count, not just `sr@1.0mm`.
