# 无监督预训练（Masked Reconstruction）MVP

## 目标
在现有 CEJ ROI 流程基础上，新增“无监督预训练 → 监督微调”的骨架流程。
本阶段仅要求**跑通流程（冒烟测试）**，模型细节与效果指标后续再补充。

## 核心思路
- **输入粒度**：牙体 ROI（优先），如 mask 不可靠可回退为全量 ROI。
- **方法**：Masked Reconstruction（MAE 风格），对 ROI 做 patch 级遮挡，仅在 masked 区域计算重建损失。
- **命名**：明确为 “masked reconstruction pretrain”，避免与 ViT/MAE 实现混淆。

## 数据与预处理
- 来源：`data/TF_008`（无标签预训练样例）
- 预处理脚本：`src/preprocess_unsup.py`
- 输出：`data/processed_unsup/{case_id}/tooth_{id}/A_t.{fmt}, T_t.{fmt}, roi_meta.json`

## 训练与损失
- 训练脚本：`src/pretrain_mae.py`
- 模型：UNet3D（MVP 占位）
- 损失：`masked_mse`（仅 masked 区域）
- mask 参数：`mask_ratio`, `mask_patch_size` 均为配置项

## 验收（MVP）
- 预处理成功生成 ROI 数据
- 预训练脚本可运行 1 个 epoch / 1 个 step 并产出 checkpoint
- 输出位置：`outputs/unsup/pretrain/checkpoints/last.pt`
- 支持断点续跑：再次启动时自动从 `last.pt` + `metrics.csv` 继续。

## 当前云端执行状态（2026-03-03）

- 已完成无监督预训练到 `epoch=120`（run: `/root/cej_runs/run_unsup_001`）。
- 指标文件：`/root/cej_runs/run_unsup_001/unsup/pretrain/metrics.csv`
- checkpoint：`/root/cej_runs/run_unsup_001/unsup/pretrain/checkpoints/last.pt`
- 当前 checkpoint 保留策略：仅保留 `last.pt`（后续可扩展为每 N epoch 保留快照）。

## 配置入口
见 `configs/unsup_mae.yaml`：
- `unsup.*`：数据来源、ROI 与 mask 相关
- `train_unsup.*`：训练超参与 mask 参数
- `finetune.*`：监督微调入口（可选）

## TODO（后续补充）
- 在真实多病例数据上补齐无监督预训练基线（当前 TF_008 为 smoke 级验证）。
- 替换为 ViT/MAE 编码器-解码器结构
- 支持更复杂的 mask 策略（结构化/块状/重要性）
- 加入更合理的无监督验收指标（表示质量/线性探针等）
- 对 TF_008 label 语义范围进行确认与适配
- 增加多 case 无监督数据集接入与数据清洗

---

## 更新记录
- 2026-02-11：`run_all_unsup.sh`（TF_008）冒烟跑通，输出 `outputs/unsup/pretrain/checkpoints/last.pt`。
- 2026-03-01：文档状态同步；将“MONAI 无监督流程跑通”从 TODO 移至已完成，下一步聚焦多病例基线与指标。
- 2026-03-03：`src.pretrain_mae.py` 已支持 checkpoint/optimizer/metrics 续跑逻辑，云端无监督训练已跑至 120 epoch。
