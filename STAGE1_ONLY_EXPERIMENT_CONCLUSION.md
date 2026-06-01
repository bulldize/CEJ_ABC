# Stage1-only 实验结论

日期：2026-06-01

## 决策

后续默认训练路径不再使用 Stage2。

当前证据支持以 Stage1-only 作为主线。Stage2 在固定 16 train / 3 holdout split 上没有显示独立增益；Data Learning Curve 显示当前性能仍明显受数据量限制。

## 证据

### 同一 16/3 split 上 Stage1-only vs Stage2

报告：
`/root/cej_isolated_runs/C/stage1_16train_direct_eval_20260601_073819/summary/direct_eval_report.json`

固定验证集：
`ToothFairy3F_040`、`ToothFairy3F_044`、`ToothFairy3F_052`，共 88 颗牙。

| 实验 | 阈值 | 状态 | Mean mm | P95 mm | SR@1mm | failed/bad teeth |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Stage1-only best | 0.20 | PASS_USABLE | 0.617 | 1.435 | 0.852 | 4 |
| Stage2 16/3 primary | 0.30 | FAIL | 0.652 | 1.516 | 0.854 | 3 |
| Stage2 16/3 max20/min7 | 0.20 | FAIL | 0.659 | 1.621 | 0.849 | 5 |

Stage1-only 在该 split 上与 Stage2 接近或更好，因此没有理由把 Stage2 保留为默认训练环节。

### Stage1-only 数据量学习曲线

报告：
`/root/cej_isolated_runs/C/stage1_data_learning_curve_20260601_075445/summary/data_learning_curve_report.json`

CSV：
`/root/cej_isolated_runs/C/stage1_data_learning_curve_20260601_075445/summary/learning_curve.csv`

| 训练数据量 | cases / teeth | 阈值 | 状态 | Mean mm | P95 mm | SR@1mm | failed/bad teeth |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 20% | 3 / 91 | 0.25 | FAIL | 3.635 | 19.579 | 0.596 | 23 |
| 40% | 6 / 187 | 0.25 | FAIL | 2.678 | 4.240 | 0.678 | 15 |
| 60% | 10 / 313 | 0.20 | FAIL | 2.504 | 3.867 | 0.749 | 12 |
| 80% | 13 / 404 | 0.25 | FAIL | 0.909 | 3.517 | 0.760 | 10 |
| 100% | 16 / 497 | 0.25 | PASS_USABLE | 0.628 | 1.417 | 0.846 | 3 |

80% 到 100% 阶段，P95 改善 2.100 mm，SR@1mm 提升 0.086。这不是 plateau。

## 结论

瓶颈不是 Stage2，而是数据覆盖不足。

下一步主线：

1. 继续使用 Stage1-only `exp03_skeleton_aux_loss`。
2. 优先增加标注病例，尤其是与当前 holdout 失败牙相似的病例。
3. 同时测试更强的数据增强，把它作为低成本的数据扩展代理。
4. Stage2 仅作为 ablation 归档，不作为默认训练流程。

## 注意

本曲线只使用了一组 nested case 顺序和一个固定 3-case holdout。它已经足够支撑当前 pipeline 决策；若要做更稳健的数据充分性估计，可以后续再补一组不同 subset 顺序或另一个固定 holdout。
