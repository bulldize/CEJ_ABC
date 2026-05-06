# CEJ 本地推理 Runtime

本仓库已经瘦身为 CEJ 本地推理 runtime，只保留这条链路：

```text
原始 CBCT + 原始牙齿 segmentation + CEJ checkpoint -> CEJ 推理曲线 -> 3D 可视化
```

本地 runtime 根目录在 Git 仓库外、项目根目录内：

```text
/Users/bulldize/Desktop/华西口腔_CEJ_ABC/runtime/cej/
  models/
  raw/
  processed/
  outputs/
  configs/
```

大体积 runtime 资产不会进入 Git。checkpoint 放在 `runtime/cej/models`，原始病例放在 `runtime/cej/raw`，自动生成的 tooth ROI 放在 `runtime/cej/processed`，推理和可视化结果放在 `runtime/cej/outputs`。

## 运行

```bash
cd /Users/bulldize/Desktop/华西口腔_CEJ_ABC/codes
source .venv/bin/activate

python -m src.infer --config /Users/bulldize/Desktop/华西口腔_CEJ_ABC/runtime/cej/configs/local_runtime.yaml
python -m src.viz --config /Users/bulldize/Desktop/华西口腔_CEJ_ABC/runtime/cej/configs/local_runtime.yaml
```

如果 `runtime/cej/processed` 为空，`src.infer` 会自动从 raw CBCT 和 segmentation 生成 inference-only tooth ROI。推理会输出 tooth-level 的 `H_pred.nii.gz`、`C_pred.nii.gz`、`C_pred_fit.nii.gz`、`curve_pred_dense_points.npy`，以及 case-level 的 `Y_pred.nii.gz` 和 `H_pred.nii.gz`。

`src.viz` 支持没有 GT/manual points 的 prediction-only 可视化。3D 索引输出到：

```text
/Users/bulldize/Desktop/华西口腔_CEJ_ABC/runtime/cej/outputs/viz/3d/index.html
```

## Raw 病例格式

推荐使用 case-directory 格式：

```text
runtime/cej/raw/{case_id}/
  A.nii.gz
  B.nii.gz
```

`A.nii.gz` 是 CBCT，`B.nii.gz` 是牙齿 segmentation。`B.nii.gz` 中 `>=100` 的牙髓标签会自动映射回牙位标签 `label % 100`。

`src/datasets/raw_cases.py` 仍支持 ToothFairy3 的 `imagesTr/labelsTr` 格式，但新的本地 runtime 数据必须放在 `/Users/bulldize/Desktop/华西口腔_CEJ_ABC/runtime/cej` 下。
