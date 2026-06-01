# 从 `unsup_last.pt` 到 `goal模式_19full` 的完整复现任务

## 任务目标

在下一次会话中，假设本机 CUDA/GPU 已可用，不继续排查 GPU 问题。用 `unsup_last.pt` 作为唯一初始模型，重新跑完两段训练链路并完成验收：

1. `unsup_last.pt` -> 训练得到新的 `exp03_skeleton_aux_loss/outputs/train/checkpoints/best.pt`
2. 新生成的 Stage 1 `best.pt` -> 运行 `goal模式_19full` 流程 -> 得到最终 accepted 指标和完整产物

当前代码 commit 必须是：

```bash
cbca7f9c53367c21d446317efdda49bc2faea187
```

## 重要约束

- 不要覆盖已有成功基线或已有历史产物。
- 不要复用本轮失败的部分目录：
  `/root/cej_isolated_runs/C/unsup_to_goal_retrain_20260530_134051_stage1`
- 下一次新建新的 run root，例如：
  `/root/cej_isolated_runs/C/unsup_to_goal_retrain_<YYYYMMDD_HHMMSS>_stage1`
  `/root/cej_isolated_runs/C/unsup_to_goal_retrain_<YYYYMMDD_HHMMSS>_stage2_goal19`
- 不要求 checkpoint bit-for-bit 一致，验收以指标通过和 lineage 可追溯为准。
- 如果 CUDA 在下次会话仍不可用，不要继续排 GPU；直接记录 blocker 并停止。

## 已知输入

唯一初始 checkpoint：

```bash
/root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt
```

仓库路径：

```bash
/root/workspace/CEJ_ABC
```

Stage 1 脚本：

```bash
scripts/run_cej_unsup_retrain.py
```

Stage 2 脚本：

```bash
scripts/run_goal_mode_19full.py
```

## 预检查

从仓库目录执行：

```bash
cd /root/workspace/CEJ_ABC
git rev-parse HEAD
git status --short
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY
```

要求：

- `git rev-parse HEAD` 输出 `cbca7f9c53367c21d446317efdda49bc2faea187`
- `torch.cuda.is_available()` 为 `True`
- `nvidia-smi` 能看到至少 1 张 GPU

## Stage 1：从 `unsup_last.pt` 训练 `exp03_skeleton_aux_loss`

设置新的 timestamp 和 run root：

```bash
RUN_TS=$(date -u +%Y%m%d_%H%M%S)
STAGE1=/root/cej_isolated_runs/C/unsup_to_goal_retrain_${RUN_TS}_stage1
```

先准备输入：

```bash
python scripts/run_cej_unsup_retrain.py \
  --repo-dir /root/workspace/CEJ_ABC \
  --run-root "$STAGE1" \
  --unsup-ckpt /root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt \
  --experiments exp03_skeleton_aux_loss \
  --setup-only
```

检查 Stage 1 input manifest：

```bash
python - <<PY
import json
from pathlib import Path
p = Path("$STAGE1/input/source_manifest.json")
data = json.loads(p.read_text())
print(json.dumps({
  "unsup_checkpoint_source": data.get("unsup_checkpoint_source"),
  "unsup_checkpoint_copy": data.get("unsup_checkpoint_copy"),
  "train_tooth_dirs": data.get("train_tooth_dirs"),
  "holdout_tooth_dirs": data.get("holdout_tooth_dirs"),
  "train_cases": data.get("train_cases"),
  "holdout_cases": data.get("holdout_cases"),
}, ensure_ascii=False, indent=2))
PY
```

然后运行 Stage 1 训练：

```bash
python scripts/run_cej_unsup_retrain.py \
  --repo-dir /root/workspace/CEJ_ABC \
  --run-root "$STAGE1" \
  --unsup-ckpt /root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt \
  --experiments exp03_skeleton_aux_loss \
  --resume \
  --stop-on-failure
```

Stage 1 必须产生：

```bash
$STAGE1/experiments/exp03_skeleton_aux_loss/outputs/train/checkpoints/best.pt
$STAGE1/experiments/exp03_skeleton_aux_loss/outputs/train/checkpoints/last.pt
$STAGE1/experiments/exp03_skeleton_aux_loss/outputs/train/train_manifest.json
$STAGE1/experiments/exp03_skeleton_aux_loss/outputs/train/metrics.csv
$STAGE1/experiments/exp03_skeleton_aux_loss/logs/train_loss_earlystop.log
$STAGE1/experiments/exp03_skeleton_aux_loss/config/train.yaml
```

Lineage 要求：

- `$STAGE1/input/source_manifest.json` 中 `unsup_checkpoint_source` 必须是 `/root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt`
- Stage 1 `train_manifest.json` 中 `pretrained.path` 应指向 `$STAGE1/input/checkpoints/unsup_last.pt`
- `$STAGE1/input/checkpoints/unsup_last.pt` 的 hash 必须等于原始 `last.pt`

## Stage 2：用新 Stage 1 `best.pt` 跑 `goal模式_19full`

设置 Stage 2 run root：

```bash
STAGE2=/root/cej_isolated_runs/C/unsup_to_goal_retrain_${RUN_TS}_stage2_goal19
STAGE1_BEST=$STAGE1/experiments/exp03_skeleton_aux_loss/outputs/train/checkpoints/best.pt
```

运行 goal 模式。当前默认复现流程不要把 Stage 2 的 `min_epochs=20` 当作硬要求；脚本会按数据规模自动决定第一轮候选训练长度，再用最终 infer/eval 指标验收：

```bash
python scripts/run_goal_mode_19full.py \
  --repo-dir /root/workspace/CEJ_ABC \
  --run-root "$STAGE2" \
  --best-ckpt "$STAGE1_BEST" \
  --unsup-ckpt /root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt \
  --device cuda
```

如果严格验收没有通过，再开新的 Stage 2 run root，用更大的 `--target-train-steps` / `--max-epochs` 继续试验；不要覆盖已经生成的 run root。

Stage 2 必须产生：

```bash
$STAGE2/input/source_manifest.json
$STAGE2/input/case_manifest.csv
$STAGE2/config/goal_mode_train.yaml
$STAGE2/config/final_selected.yaml
$STAGE2/outputs_final/train/checkpoints/best.pt
$STAGE2/outputs_final/train/checkpoints/last.pt
$STAGE2/outputs_final/train/metrics.csv
$STAGE2/outputs_final/train/train_manifest.json
$STAGE2/logs/train_goal_mode.log
$STAGE2/final/eval/metrics_summary.json
$STAGE2/final/eval/metrics_per_tooth.csv
$STAGE2/summary/acceptance_report.json
$STAGE2/package/goal模式_19full_training_bundle.tar.gz
```

Stage 2 lineage 要求：

- `$STAGE2/input/source_manifest.json` 中 `pretrained_source` 必须等于 `$STAGE1_BEST`
- `$STAGE2/input/source_manifest.json` 中 `pretrained_copy` 必须存在
- `pretrained_copy` 的 hash 必须等于 `$STAGE1_BEST`
- `$STAGE2/input/source_manifest.json` 中 `case_count == 19`
- `$STAGE2/input/source_manifest.json` 中 `tooth_dir_count == 585`

## 最终验收指标

以 `$STAGE2/summary/acceptance_report.json` 和 `$STAGE2/final/eval/metrics_summary.json` 为准。

必须全部满足：

- `accepted == true`
- `case_count == 19`
- `tooth_count == 585`
- `missing_prediction_count == 0`
- `no_curve_count == 0`
- `failed_tooth_count == 0`
- `failed_or_bad_teeth_count == 0`
- `mean_dist_mm <= 0.35`
- `p95_dist_mm <= 0.65`
- `sr@1.0mm >= 0.99`

原始成功参考值：

- `mean_dist_mm = 0.2973125875`
- `p95_dist_mm = 0.5984374464`
- `sr@1.0mm = 0.9957592056`
- selected threshold: `0.35`

## 训练指标纠错记录（2026-06-01）

本次复现中曾错误地把 Stage 2 训练 CSV 中的监控指标当作最终验收指标来判断是否继续训练，例如：

- `outputs_final/train/metrics.csv` 中的 `holdout_manual_point_p95`
- `outputs_final/train/metrics.csv` 中的 `holdout_manual_point_sr1`
- `outputs_final/train/metrics.csv` 中的 `holdout_sym_p95`

这些字段只是训练期 holdout/proxy monitor，用于选择 `best.pt`，不能和最终验收阈值 `mean_dist_mm <= 0.35`、`p95_dist_mm <= 0.65`、`sr@1.0mm >= 0.99` 直接比较。

权威验收只能来自完成 inference/eval 后的：

```bash
$STAGE2/final/eval/metrics_summary.json
$STAGE2/summary/acceptance_report.json
```

历史成功基线也证明二者不是同一量纲：`goal模式_19full_001` 的 Stage 2 `best_epoch=6` 时训练代理指标约为 `holdout_manual_point_p95=1.9438244581`、`holdout_manual_point_sr1=0.6269135292`，但最终 eval 指标为 `mean_dist_mm=0.2973125875`、`p95_dist_mm=0.5984374464`、`sr@1.0mm=0.9957592056`，严格验收通过。

结论：Stage 2 不能等待训练 CSV 里的 proxy p95 降到 `0.65` 或 proxy sr@1 升到 `0.99`；那会误判并浪费训练时间。

## Stage 2 停止规则（默认复现策略）

Stage 2 的停止规则以最终 eval strict acceptance 为准，不以训练 epoch 数或训练 CSV proxy 指标为准。

默认规则：

1. Stage 2 训练长度按数据规模自动决定，不能固定写死 6、7 或 20 epoch。默认以目标训练步数 `target_train_steps=3437` 为基准，估算：
   - `steps_per_epoch = ceil(tooth_dir_count / batch_size)`
   - `max_epochs = ceil(target_train_steps / steps_per_epoch)`，并受 `auto_min_epochs` / `auto_max_epochs` 约束
   - `score_start_epoch` 也按目标步数比例自动推算
2. 数据更多时，每个 epoch 覆盖的 tooth 更多，因此需要的 epoch 可以更少；数据更少时，每个 epoch 覆盖的 tooth 更少，因此需要的 epoch 应更多。
3. 本次 19 case / 585 tooth / batch=1 时，自动策略会落在约 7 epoch；这只是当前数据规模下的结果，不是固定规则。
4. 第一轮训练自然结束后，必须对当前 `$STAGE2/outputs_final/train/checkpoints/best.pt` 跑 threshold sweep / final eval，生成 `final/eval/metrics_summary.json` 和 `summary/acceptance_report.json`。
5. 只有当 strict acceptance 全部通过时停止并接受该 Stage 2。对当前 `goal模式_19full` 数据，expected case/tooth 数是 19/585；如果未来数据集增减，expected 数应来自 `$STAGE2/input/source_manifest.json`：
   - `case_count == expected_case_count`
   - `tooth_count == expected_tooth_count`
   - `missing_prediction_count == 0`
   - `no_curve_count == 0`
   - `failed_tooth_count == 0`
   - `failed_or_bad_teeth_count == 0`
   - `mean_dist_mm <= 0.35`
   - `p95_dist_mm <= 0.65`
   - `sr@1.0mm >= 0.99`
6. 如果 strict acceptance 未通过，不要继续使用同一个 run root 覆盖训练产物；新建新的 Stage 2 run root，并提高 `--target-train-steps` / `--max-epochs` 或调整策略后重跑。
7. 训练 CSV 只能用于判断是否已经产生 `best.pt`、当前 best epoch 是多少、以及是否值得启动一次 eval；不能作为验收通过/失败的最终证据。

默认一键流程：

```bash
bash scripts/run_unsup_to_goal_repro.sh
```

## 建议生成最终复现验收报告

建议在 `$STAGE2/summary/` 下生成：

```bash
reproduction_lineage_acceptance_report.json
```

报告至少包含：

- repo commit
- Stage 1 run root
- Stage 2 run root
- 原始 unsup checkpoint 路径、大小和 sha256
- Stage 1 `unsup_last.pt` copy 路径、大小和 sha256
- Stage 1 `best.pt` 和 `last.pt` 路径、大小和 sha256
- Stage 2 `pretrained_copy` 路径、大小和 sha256
- Stage 2 final `best.pt` 和 `last.pt` 路径、大小和 sha256
- strict acceptance criteria 每项 PASS/FAIL
- lineage checks 每项 PASS/FAIL
- artifact checks 每项 PASS/FAIL
- final metrics 摘要

验收报告中 `overall_accepted` 只有在 strict criteria、lineage checks、artifact checks 全部通过时才为 `true`。

## 本轮状态记录

本轮已经做过但未完成训练：

- 新建并准备过 Stage 1 输入：
  `/root/cej_isolated_runs/C/unsup_to_goal_retrain_20260530_134051_stage1`
- 该目录内 `input/source_manifest.json` 已指向正确的原始 `unsup_last.pt`
- 尝试启动 Stage 1 训练后失败，失败原因是当前会话 CUDA 不可用：
  `RuntimeError: CUDA unknown error`
- 该失败目录不要复用，不要覆盖。下一次开新的 run root。

本轮已验证过的历史成功基线：

- Stage 1 历史成功：
  `/root/cej_isolated_runs/C/cej_loss_earlystop_20260507_004/experiments/exp03_skeleton_aux_loss`
- Stage 2 历史成功：
  `/root/cej_isolated_runs/C/goal模式_19full_001`
- 历史 lineage 验收报告：
  `/root/cej_isolated_runs/C/goal模式_19full_001/summary/reproduction_lineage_acceptance_report.json`

这些历史产物只作为参考，不要作为“本次重新训练完成”的替代。
