# CEJ unsup reproduction final package

## Checkpoint lineage
- /root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt
- /root/cej_isolated_runs/C/goal模式_unsup_repro_19full_001/input/checkpoints/unsup_last.pt
- stage2_all19_from_stage1_exp03_trainonly_seed42_lr1e-3_base16_seed42_lr1e-3_eval_epoch_049
- /root/cej_isolated_runs/C/goal模式_unsup_repro_19full_001/candidates/stage2_all19_from_stage1_exp03_trainonly_seed42_lr1e-3_base16_seed42_lr1e-3/outputs/train/checkpoints/topk/epoch_049_score_1.024918.pt

## Selected evaluation
- evaluation split: strict_test
- threshold: 0.35
- metrics: /root/cej_isolated_runs/C/goal模式_unsup_repro_19full_001/final/eval/metrics_summary.json
- per-tooth metrics: /root/cej_isolated_runs/C/goal模式_unsup_repro_19full_001/final/eval/metrics_per_tooth.csv
- viewer: /root/cej_isolated_runs/C/goal模式_unsup_repro_19full_001/final/viz/3d/index.html
- final config: /root/cej_isolated_runs/C/goal模式_unsup_repro_19full_001/config/final_selected.yaml

## Strict split
- train cases (16): ToothFairy3F_008, ToothFairy3F_009, ToothFairy3F_010, ToothFairy3F_018, ToothFairy3F_021, ToothFairy3F_023, ToothFairy3F_025, ToothFairy3F_026, ToothFairy3F_027, ToothFairy3F_033, ToothFairy3F_041, ToothFairy3F_050, ToothFairy3F_051, ToothFairy3F_053, ToothFairy3F_054, ToothFairy3F_055
- strict test cases (3): ToothFairy3F_040, ToothFairy3F_044, ToothFairy3F_052
- stage2 training uses processed_stage1_train for both processed_dir and holdout_processed_dir.
- stage2 final inference/evaluation uses processed_stage1_holdout through eval_processed_dir.

## Reproduce this run
```bash
python scripts/run_goal_mode_19full.py --setup-only --candidate-name direct_base16_seed42_lr1e-3 --base-channels 16 --lr 1e-3
python scripts/run_goal_mode_19full.py --resume --train-only --stage stage1 --stage1-candidate stage1_exp03_trainonly_seed42_lr1e-3 --base-channels 16 --lr 1e-3 --max-epochs 50 --min-epochs 10 --patience 8 --early-stopping-min-delta 5e-4 --score-start-epoch 10 --loss-lambda-skeleton 0.2 --skeleton-pos-weight 8.0 --surface-neighborhood-weight 0.0
python scripts/run_goal_mode_19full.py --resume --train-only --stage stage2 --stage1-candidate stage1_exp03_trainonly_seed42_lr1e-3 --base-channels 16 --lr 1e-3 --max-epochs 80 --min-epochs 20 --patience 16 --score-start-epoch 5 --loss-lambda-skeleton 0.4 --skeleton-pos-weight 10.0 --surface-neighborhood-weight 0.05
python scripts/run_goal_mode_19full.py --resume --stage stage2 --stage1-candidate stage1_exp03_trainonly_seed42_lr1e-3 --base-channels 16 --lr 1e-3 --max-epochs 80 --min-epochs 20 --patience 16 --score-start-epoch 5 --loss-lambda-skeleton 0.4 --skeleton-pos-weight 10.0 --surface-neighborhood-weight 0.05 --skip-train
```

## Re-train after adding data
1. Add new PASS cases to PASS_SOURCES and assign them to STAGE1_TRAIN_CASES or STAGE1_HOLDOUT_CASES.
2. Use a fresh --run-root so previous checkpoints, predictions, and manifests are not overwritten.
3. Run setup and verify source_manifest.json plus audit_all19/supervised_dataset_audit.json: all intended cases are present and invalid_tooth_dirs is 0.
4. Train stage1 from input/checkpoints/unsup_last.pt. Do not initialize from an old exp03 checkpoint.
5. Train stage2 from the fresh stage1 checkpoint. Keep strict-test cases out of stage2 processed_dir and holdout_processed_dir.
6. Run the final --skip-train command to sweep thresholds, evaluate strict test, generate final/viz/3d/index.html, and rebuild this package.
