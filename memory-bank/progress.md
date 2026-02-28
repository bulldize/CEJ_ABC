# Progress

- Completed: Read PRD and tech-route spec, confirm constraints, create skeleton plan.
- Completed: Implemented MVP skeleton (configs, scripts, src modules, tests, schemas).
- Completed: Code review fixes (npy affine handling, point-surface distance report, viz output structure + num_slices).
- Completed: Minimal synthetic preprocess + npy IO check via python3 script.
- Completed: Pytest suite (3 tests) passed.
- Completed: Installed `scikit-image` and dependencies.
- Completed: Full pipeline run via `scripts/run_all.sh` succeeded (preprocess/train/infer/eval/viz).
- Completed: Integrated cej_points_ras.xlsx conversion to voxel points.json + mark_meta.json in preprocess.
- In Progress: Optional additional real-case validation.
- Next: Run `bash scripts/run_all.sh` if deps are installed.

一句话目标：
将每个 case 的标注坐标（cej_points_ras.xlsx）自动转换为体素 points.json，并生成 mark_meta.json 以规范坐标系映射。

已确认关键决策：
- 目标用户：项目内 CEJ pipeline 使用者
- 技术栈：Python + pandas/openpyxl
- 关键约束：无旋转/无轴交换；转换集成到 preprocess；覆盖 points.json
- 已选方案/库：扫描 8 角点推导 affine_vox_to_mark；mark_meta.json 记录仿射与元信息

当前进度：
- 已完成：转换模块（src/datasets/mark_points.py）+ preprocess 集成 + 配置/文档/Schema 更新
- 进行中：无
- 未开始：外部软件导出格式确认与支持

明确约束：
- 必须做：每个 case 存在 cej_points_ras.xlsx 时自动覆盖 points.json
- 禁止做：在 scripts/ 下新增转换脚本

下一个会话只做的一件事：
确认外部软件所需的导出格式，并补充对应导出功能。

---

更新（2026-02-08）：
- Completed: 真实数据 case0002 已按标准落盘（A/B/cej_points_ras.xlsx），raw 仅保留该 case。
- Completed: 预处理支持无 points 数据（无 points.json 时跳过点/热力图相关输出）。
- Completed: 真实 case0002 预处理跑通并生成 `data/processed/case0002/`。
- Completed: 修复 cej_points_ras.xlsx 牙位解析正则，points.json 成功生成并写入 28 颗牙的点。
- Completed: 依据真实 A 仿射自动选择 world_lps 坐标转换（x/y 翻转），points 全部落入体素范围。
- Completed: mark_meta.json 记录 points_coord 与 in-bounds 统计；case0002 所有点均在 3mm 表面阈值内。

一句话目标：
跑通真实数据 case0002 的预处理，并兼容无监督（无 points）数据集。

已确认关键决策：
- 目标用户：项目内 CEJ pipeline 使用者
- 技术栈：Python + pandas/openpyxl
- 关键约束：本次只跑真实 case0002；无 cej_points_ras.xlsx 则无 points.json（无监督）
- 已选方案/库：case0002 落盘 + preprocess 内部跳过点/热力图输出

当前进度：
- 已完成：真实数据落盘与预处理跑通；无 points 兼容逻辑
- 进行中：无
- 未开始：多 case 批量处理与无监督训练流程定义

明确约束：
- 必须做：cej_points_ras.xlsx 存在时自动生成 points.json
- 禁止做：在 scripts/ 下新增转换脚本

下一个会话只做的一件事：
明确无监督训练/评估所需的数据输出与流程调整。

---

更新（2026-02-10）：
- Completed: 坐标统一约定已确认并写入文档：以 `A.nii.gz` 的 sform/qform（affine）定义的世界坐标系为真实坐标基准，外部标注需先对齐到该世界坐标系再转体素。

已确认关键决策：
- 坐标统一基准：以 `A.nii.gz` 的 sform/qform（affine）世界坐标系为准（真实坐标）。

---

更新（2026-02-10）：
- Completed: 无监督预训练骨架（ROI 预处理 + masked reconstruction 训练 + 脚本 + 配置）已落地。
- Completed: 新增 `run_all_unsup.sh` 与 `configs/unsup_mae.yaml`，支持 pretrain →（可选）finetune。
- Completed: 文档补充无监督 MVP 方案与 TODO 清单。

一句话目标：
跑通无监督预训练（masked reconstruction）→ 监督微调的最小流程骨架。

已确认关键决策：
- 目标用户：项目内 CEJ pipeline 使用者
- 技术栈：Python + PyTorch + nibabel
- 关键约束：不干扰现有监督主流程；只做冒烟级可跑通；未填内容必须 TODO
- 已选方案/库：UNet3D + patch mask + masked MSE

当前进度：
- 已完成：unsup 预处理与训练脚本、配置、文档、可选微调入口
- 进行中：无
- 未开始：替换 ViT/MAE、补充无监督评估指标

明确约束：
- 必须做：ROI/牙体为默认输入粒度；输出骨架与 TODO
- 禁止做：改动监督主流程行为（默认配置）

下一个会话只做的一件事：
确认 TF_008 label 语义范围并接入真实 MAE/ViT 实现与评估指标。

---

更新（2026-02-11）：
- Completed: `python -m src.preprocess_unsup --config configs/unsup_mae.yaml` 跑通 TF_008 ROI 预处理。
- Completed: `python -m src.pretrain_mae --config configs/unsup_mae.yaml` 冒烟训练 1 step 成功并保存 checkpoint。

---

更新（2026-02-11）：
- Completed: 引入本地 MONAI 源码到 `MONAI/`，通过 `src/__init__.py` 自动注入到 `PYTHONPATH`。
- Completed: 监督流程模块替换为 MONAI 风格（dataset/transforms、UNet、DiceCE loss、推理可选滑窗、后处理 LCC）。
- Completed: 预处理在无 CEJ 点时仍生成空热图与空 points.json 以支持冒烟训练。
- Completed: `scripts/run_all.sh` 改为使用 TF_008 数据准备（不再生成合成数据），默认 raw_dir 改为 `data/raw_tf008`。
- Completed: 技术路线文档补充 MONAI + TF_008 冒烟 TODO。
- Completed: `python -m src.train --config configs/default.yaml` 在现有 processed 数据上训练通过（padding 修复生效）。
- Blocked: 监督冒烟失败，`data/TF_008` 缺失 `ToothFairy3F_008_volume.nii`（A 体数据）。需要补齐真实 A 体文件后重跑。

更新（2026-02-11）：
- Completed: TF_008 数据补齐后 `scripts/run_all.sh` 已跑通（preprocess/train/infer/eval/viz）。
- Completed: 修复 `src/eval.py` 中 `_ensure_spacing` 定义顺序导致的 NameError。

更新（2026-02-11）：
- Completed: 清理 `outputs/` 与 `data/processed/` 后，`scripts/run_all.sh` 重新全流程跑通（TF_008）。

更新（2026-02-11）：
- Completed: `scripts/run_all_unsup.sh` 在 TF_008 冒烟跑通，并产出 `outputs/unsup/pretrain/checkpoints/last.pt`。

一句话目标：
基于 MONAI 替换监督流程并用 TF_008 冒烟跑通。

已确认关键决策：
- 目标用户：项目内 CEJ pipeline 使用者
- 技术栈：Python + PyTorch + MONAI（本地源码注入）
- 关键约束：保持入口与结构；不使用合成数据；TF_008 作为监督/无监督数据；CPU 兜底
- 已选方案/库：MONAI UNet + MONAI transforms + 现有 ROI/热图流程

当前进度：
- 已完成：监督模块替换与配置更新
- 进行中：监督冒烟（缺 TF_008 A 体文件）
- 未开始：无监督对齐与 run_all_unsup 冒烟

明确约束：
- 必须做：补齐 TF_008 体数据后跑通 `scripts/run_all.sh`
- 禁止做：使用合成数据替代 TF_008

下一个会话只做的一件事：
补齐 TF_008 A 体数据并重跑监督冒烟，通过后进入无监督对齐。

---

更新（2026-03-01）：
- Completed: 移除本地 `MONAI/` 源码目录，改为环境内 `monai` 包依赖。
- Completed: 删除 `src/__init__.py` 中本地 MONAI `PYTHONPATH` 注入逻辑。
- Completed: 冒烟验证改为“无本地 MONAI 目录”前提执行。
- Completed: 冒烟通过：`pytest tests`、监督链路 `preprocess/train/infer/eval/viz`、无监督链路 `preprocess_unsup/pretrain_mae`。

更新（2026-03-01, visualization）：
- Completed: `src.viz` 增加交互式 3D 可视化（Plotly HTML）：MPR 三正交切片、牙体表面、GT/Pred 热图等值面、预测曲线点云、GT 点误差着色。
- Completed: 默认关闭 2D、开启 3D（`configs/default.yaml`）。

更新（2026-03-01, docs）：
- Completed: README、memory-bank、doc/specs 已同步到当前可视化方案：默认 3D viewer、2D 可选、输出索引为 `outputs/viz/3d/index.html`。
