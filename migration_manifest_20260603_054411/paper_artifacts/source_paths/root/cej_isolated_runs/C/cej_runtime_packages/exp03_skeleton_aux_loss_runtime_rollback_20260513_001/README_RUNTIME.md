# CEJ Runtime Package: exp03_skeleton_aux_loss_runtime_rollback_20260513_001

This is the rollback runtime package for the known successful `exp03_skeleton_aux_loss` baseline.

## Files

- `checkpoints/best.pt`: exp03 best checkpoint for inference and visualization.
- `checkpoints/last.pt`: exp03 last checkpoint for provenance.
- `configs/runtime_infer.yaml`: portable runtime config. Run commands from this package directory.
- `configs/runtime_infer_absolute.yaml`: absolute-path config for this machine.
- `configs/train_original.yaml`: original exp03 training config.
- `metrics/`: exp03 training and holdout summaries.

## Recommended Runtime Commands

From the extracted package directory:

```bash
python -m src.infer --config configs/runtime_infer.yaml
python -m src.viz --config configs/runtime_infer.yaml
```

On this machine, from any working directory:

```bash
python -m src.infer --config /root/cej_isolated_runs/C/cej_runtime_packages/exp03_skeleton_aux_loss_runtime_rollback_20260513_001/configs/runtime_infer_absolute.yaml
python -m src.viz --config /root/cej_isolated_runs/C/cej_runtime_packages/exp03_skeleton_aux_loss_runtime_rollback_20260513_001/configs/runtime_infer_absolute.yaml
```

## Rollback Notes

- Source experiment: `/root/cej_isolated_runs/C/cej_loss_earlystop_20260507_004/experiments/exp03_skeleton_aux_loss`
- Source status: `success`
- Best epoch: `22`
- Best composite score: `0.8143920010183066`
- Best holdout sym_p95: `5.099019527435303`
- Runtime geometry gating: `false`

The failed gated/FDI split experiment outputs are left untouched.
