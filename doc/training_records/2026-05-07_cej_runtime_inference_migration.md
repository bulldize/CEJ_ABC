# CEJ Runtime 推理分支迁移说明

日期：2026-05-07

## 目标

把本次最佳 CEJ 模型接入 `cej runtime` 推理 + 可视化分支，同时不影响之前模型的推理。


```

最佳训练包：

```text
/Users/bulldize/Desktop/temp/实验文件/exp03_skeleton_aux_loss_training_bundle.tar.gz
```

## 最小安全方案

为了不影响旧模型推理，`cej runtime` 分支建议只新增一个新 config，不修改 `src/infer.py` 和 `src/viz.py`。

也就是说：

- 旧模型继续使用旧 config 和旧 `ckpt_path`。
- 新模型使用新 config 和新 `ckpt_path`。
- 两套模型通过配置隔离，不共享输出目录。

推荐新增：

```text
configs/runtime_infer_exp03_skeleton_best.yaml
```

关键字段如下：

```yaml
project:
  seed: 42
  device: cuda

data:
  processed_dir: /你的/runtime/processed
  output_dir: /你的/runtime/outputs_exp03_skeleton_best
  processed_format: nii.gz
  use_tooth_mask_channel: true

model:
  in_channels: 2
  out_channels: 1
  base_channels: 16
  depth: 4
  num_res_units: 2
  norm: batch

infer:
  ckpt_path: /root/cej_isolated_runs/C/cej_loss_earlystop_20260507_004/experiments/exp03_skeleton_aux_loss/outputs/train/checkpoints/best.pt
  threshold_theta: 0.3
  use_prior_gating: false
  use_sliding_window: false
  constrain_curve_to_tooth_mask: false
  constrain_curve_to_tooth_surface: false
  keep_lcc_for_curve: false
  fit_pred_curve: true
```

推理命令：

```bash
python -m src.infer --config configs/runtime_infer_exp03_skeleton_best.yaml
```

可视化继续使用同一个 `data.output_dir`：

```bash
python -m src.viz --config configs/runtime_infer_exp03_skeleton_best.yaml
```

输出位置：

```text
{data.output_dir}/infer/{case_id}/tooth_{tooth_id}/
  H_pred.nii.gz
  C_pred.nii.gz
  C_pred_fit.nii.gz
```

## 不影响旧模型推理的规则

1. 不修改旧 config。

旧模型 config 里的这些字段保持不变：

```yaml
data:
  processed_dir: ...
  output_dir: ...

infer:
  ckpt_path: /旧模型/checkpoints/last.pt 或 best.pt
```

2. 不让新旧模型共用 `data.output_dir`。

如果共用输出目录，新模型会覆盖：

```text
outputs/infer/
outputs/viz/
```

因此新模型必须使用独立目录，例如：

```text
outputs_exp03_skeleton_best
```

3. 不修改模型结构字段。

新旧模型只要 checkpoint 对应的 config 结构一致即可。当前最佳模型需要：

```yaml
model:
  in_channels: 2
  out_channels: 1
  base_channels: 16
  depth: 4
  num_res_units: 2
  norm: batch
```

如果旧模型也是同结构，旧 config 继续可用；如果旧模型结构不同，不要拿新 config 加载旧 checkpoint。

4. 不修改 `src/infer.py` 和 `src/viz.py`。

当前最佳模型与现有推理和可视化接口兼容。为了最大限度避免影响旧模型，runtime 分支不需要改这两个文件。

## 可选移植项

如果 runtime 分支环境里 MONAI 版本和训练环境不一致，建议移植下面两个兼容性修复：

```text
src/datasets/transforms.py
src/datasets/dataset.py
```

作用：

- 避免依赖 MONAI 的 `ClipIntensityPercentiles` / `ClipIntensityPercentilesd`。
- 改为本地 `np.percentile + np.clip` 或 `torch.quantile + torch.clamp`。

这属于数据读取/预处理兼容修复，理论上不改变模型结构和 checkpoint 加载方式。但如果追求对旧模型推理零风险，可以先只新增 config，不移植这两个代码改动。

## 仅评估时需要的改动

如果 runtime 分支需要跑评估，建议移植：

```text
src/eval.py
```

改动含义：

- 默认评估 `C_pred_fit.nii.gz`。
- 缺预测不再静默跳过，会记录为失败/罚分。

如果 runtime 分支只做推理和可视化，不需要移植 `src/eval.py`。

## 不建议移植到 runtime 的文件

以下文件主要服务训练实验，不建议进入 `cej runtime` 推理分支：

```text
scripts/run_cej_unsup_retrain.py
scripts/run_cej_experiments.py
src/train_loss_earlystop.py
src/train_holdout.py
src/train.py
tests/test_train_loss_sampler.py
```

这些文件不会被 `python -m src.infer` 或 `python -m src.viz` 直接需要。

## 验证步骤

迁移后建议做两个 smoke test。

第一步，旧模型原 config 推理：

```bash
python -m src.infer --config configs/旧模型_runtime.yaml
```

确认旧输出仍能生成：

```text
旧 data.output_dir/infer/
```

第二步，新模型新 config 推理：

```bash
python -m src.infer --config configs/runtime_infer_exp03_skeleton_best.yaml
```

确认新输出生成到独立目录：

```text
新 data.output_dir/infer/
```

如果还要可视化：

```bash
python -m src.viz --config configs/runtime_infer_exp03_skeleton_best.yaml
```

## 结论

最稳妥的 runtime 迁移方式是：

1. 不改 `src/infer.py`。
2. 不改 `src/viz.py`。
3. 新增一个新模型专用 config。
4. 新模型使用独立 `data.output_dir`。
5. 旧模型继续使用旧 config、旧 checkpoint、旧 output_dir。

这样新模型接入不会影响之前模型的推理。
