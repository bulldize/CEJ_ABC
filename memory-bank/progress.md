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
