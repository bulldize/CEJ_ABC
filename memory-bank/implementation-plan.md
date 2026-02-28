# Implementation Plan

1. Completed: Create project skeleton, config, schemas, scripts, and README.
2. Completed: Implement data IO, ROI crop/stitch, points handling, heatmap generation.
3. Completed: Implement model (3D U-Net), training, inference, postprocess, evaluation.
4. Completed: Implement visualization/export (interactive 3D viewer + pseudo-GT full-mouth NIfTI export) and unit tests.
5. Completed: Support both raw layouts (`case_dirs` and `toothfairy3`) with `raw_layout=auto`.
6. Completed: Stabilize runner (`scripts/run_all.sh`) for macOS Bash 3.2 and custom raw filename config.
7. Next: Run full real-data benchmark (beyond smoke) and lock target hyperparameters from evaluation reports.
