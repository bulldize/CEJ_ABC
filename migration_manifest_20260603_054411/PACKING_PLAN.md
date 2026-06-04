# Packing Plan

## Full or Core Migration

- `/root/workspace/CEJ_ABC`, excluding virtualenv/cache files.
- `/root/ToothFairy3`.
- `/root/cej_runs/run_unsup_001`, excluding only unsup visualization output.
- Exp03 latest and old core experiment directories.
- `goal模式_unsup_repro_19full_001`, excluding threshold sweep bodies and full inference/visualization outputs.
- `goal模式_19full_001`, excluding threshold sweep bodies and full inference/visualization outputs.
- `/root/backup_models`.
- selected sensitive server configuration.

## Logs and Configs for Weaker Experiments

Weaker or exploratory experiments are not migrated full-size. Their lightweight evidence is copied into `paper_artifacts/source_paths/`:

- stage1 full19 direct eval;
- stage1 16-train direct eval;
- stage1 data learning curve;
- geometry prior variants;
- goal mode variants;
- older supervised/manual/curvefix runs.

## Size Drivers

Current root filesystem usage is about 198G. The largest contributors are:

- `/root/cej_isolated_runs`: about 120G;
- `/root/ToothFairy3`: about 28G;
- `/root/cej_runs`: about 24G;
- project `.venv`: about 7.4G, excluded;
- `/root/.cache`: about 7.2G, excluded;
- `/root/transfer_bundles`: about 6.3G, excluded.
