# CEJ_ABC Migration Bundle 20260603_054411

This bundle is intended for server migration and paper-oriented reproducibility.

## Included Core Content

- Repository: `/root/workspace/CEJ_ABC`, excluding `.venv`, caches, and bytecode.
- Raw dataset: `/root/ToothFairy3`.
- Unsupervised run: `/root/cej_runs/run_unsup_001`, including raw/preprocessed run data and `unsup/pretrain/checkpoints/last.pt`.
- Exp03 training core:
  - `/root/cej_isolated_runs/C/unsup_to_goal_retrain_20260601_012139_stage1/experiments/exp03_skeleton_aux_loss`
  - `/root/cej_isolated_runs/C/cej_loss_earlystop_20260507_004/experiments/exp03_skeleton_aux_loss`
- Main reproduced experiment:
  - `/root/cej_isolated_runs/C/goal模式_unsup_repro_19full_001`
- Current best experiment core:
  - `/root/cej_isolated_runs/C/goal模式_19full_001`
- Lightweight model backups: `/root/backup_models`.
- Server proxy/VPN-related configuration:
  - `/root/xray`
  - `/root/clash`
  - `/root/.proxy_env`
  - selected shell and pip config files.

## Included Paper Artifacts

The `paper_artifacts/source_paths/` directory contains copied lightweight evidence:

- split manifests and audit JSON/CSV/TXT files;
- final metrics and per-tooth CSV files;
- threshold sweep CSV summaries only;
- configs, logs, summaries, reports, and command-relevant YAML/JSON/CSV/LOG/TXT/MD files;
- a small representative subset of 3D HTML viewer files from the current best run.

The paper artifact copies preserve source paths under `source_paths/root/...`.

## Default Exclusions

The archive intentionally excludes:

- `.venv`, `.pytest_cache`, `__pycache__`, and `*.pyc`;
- large cache directories such as `/root/.cache`, `/root/.npm`, `/root/.vscode-server`, and `/tmp`;
- old transfer bundle `/root/transfer_bundles`;
- full visualization directories except for a small copied paper subset;
- large inference outputs such as `final/infer`, `outputs*/infer`, and `outputs_test/infer`;
- large threshold sweep bodies such as `threshold_sweep*`.

## Restore Summary

On a new server, extract from `/` to recreate the original `/root/...` layout:

```bash
cd /
tar -xzf /path/to/CEJ_ABC_migration_20260603_054411.tar.gz
```

Then recreate the Python environment from `requirements.txt` plus `env_manifest/pip_freeze_venv.txt`.
