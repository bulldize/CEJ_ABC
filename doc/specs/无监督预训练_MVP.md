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

## 配置入口
见 `configs/unsup_mae.yaml`：
- `unsup.*`：数据来源、ROI 与 mask 相关
- `train_unsup.*`：训练超参与 mask 参数
- `finetune.*`：监督微调入口（可选）

## TODO（后续补充）
- 替换为 ViT/MAE 编码器-解码器结构
- 支持更复杂的 mask 策略（结构化/块状/重要性）
- 加入更合理的无监督验收指标（表示质量/线性探针等）
- 对 TF_008 label 语义范围进行确认与适配
- 增加多 case 无监督数据集接入与数据清洗
