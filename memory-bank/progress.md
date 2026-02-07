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
