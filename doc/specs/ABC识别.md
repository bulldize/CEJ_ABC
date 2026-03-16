PRD｜华西项目：CBCT 牙槽骨脊（ABC）曲线自动提取
1. 目标与范围
1.1 MVP 目标

在 CEJ 曲线已知的前提下，自动从每位患者 CBCT 三维灰度体数据及牙体/牙槽骨语义分割掩模中提取 ABC（Alveolar Bone Crest）空间曲线。输出闭合、平滑的 3D 曲线，并回贴到原 CBCT 张量中，用于后续 CEJ–ABC 骨损量化。

关键策略（Mandatory）：

单牙 ROI 为基本单元：每颗牙 t 对应裁剪后的 3D ROI（A_t、T_t、B_t）。

ABC 提取为无监督规则驱动：使用牙齿和骨掩模的几何关系及空间特征，无需人工标注。

全口输出 Y 由各牙 ROI 回贴生成。

1.2 MVP 输出张量定义

输出张量 Y ∈ {0,1,2}^{X×Y×Z}：

0：背景

1：牙体（tooth）

2：ABC 曲线体素（规则提取）

1.3 医学核验导出（Slicer 对照文件）

ABC_medical_compare_full.nii.gz：

1=牙齿

2=CEJ曲线

3=ABC曲线（算法提取）

ABC_model_compare_full.nii.gz：

1=牙齿

2=CEJ曲线

3=ABC预测曲线

配套 export_meta.json：包含体素统计、曲线长度、最大骨损距离、连通性等指标。

2. 输入数据规范
2.1 输入张量与元信息

CBCT 灰度体 A ∈ ℝ^{X×Y×Z}

牙体掩模 T ∈ {0,1}^{X×Y×Z}

牙槽骨掩模 B ∈ {0,1}^{X×Y×Z}

CEJ 曲线 C_CEJ（3D坐标）

要求：

ROI 内数据与原 CBCT 坐标对齐

spacing / affine 信息必须提供，用于物理单位计算

2.2 ROI 裁剪与回贴

以牙体实例掩模 T_t 的 3D bounding box 扩展 padding（默认 8 mm）裁剪 ROI。

记录 ROI ↔ 原体偏移/仿射映射，用于回贴 ABC 曲线到全口张量 Y==2。

2.3 强度与空间标准化

体素重采样到近各向同性 spacing（默认 0.5 mm）

强度归一化（percentile clip + z-score 或 min-max）

3. ABC 提取算法与规则先验
3.1 算法总体流程

输入 ROI：A_t、T_t、B_t、C_CEJ

构建牙齿局部坐标系（PCA对齐牙长轴）

无监督规则提取 ABC 曲线：

EDT + PDL Morphology：牙齿-骨距离约束，筛选内槽壁，按角度 θ 取最大 Z

Ray-Casting + Gradient Snapping：沿牙长轴投射射线，结合骨掩模和 CBCT 灰度梯度微调

Skeletonization + Ridge Extraction：拓扑骨架化，提取主曲率脊点

对每颗牙生成闭合平滑曲线 C_ABC_t（B-spline 或三次样条插值）

回贴到全口 CBCT 坐标系，写入输出张量 Y

3.2 几何先验

内槽壁选择：

D_T(v) = EuclideanDistance(v, T_t)
W_inner = {v ∈ ∂B_t | D_T(v) ≤ 1.5 mm}

最终 ABC 点：

ABC(θ) = argmax_{v ∈ W_inner(θ)} Z(v)

可选灰度梯度权重：

G(v) = ||∇A(v)||
Ĝ(v) = (G(v)-min)/(max-min+ε)
ABC(θ) ← ABC(θ) * Ĝ(v)
3.3 曲线输出

每颗牙输出闭合、平滑 ABC 曲线 C_ABC_t

稠密采样（默认 0.2 mm）用于可视化和后续 CEJ–ABC 骨损量化

4. 输出与可视化
4.1 输出张量
Y ∈ {0,1,2}^{X×Y×Z}
0=背景
1=牙体
2=ABC曲线体素
4.2 可视化产物

3D HTML Viewer：MPR 三正交切片 + 牙体表面 + CEJ + ABC 曲线

2D Slice 抽检：牙体边界+ABC曲线

规则先验热力图：W_inner / PDL 范围

误差分布图（GT 可选）：CEJ-ABC 最大骨损或可选手工 ABC 点

Slicer 对照文件（Mandatory）：

ABC_medical_compare_full.nii.gz

ABC_model_compare_full.nii.gz

评估报告 CSV：每颗牙曲线长度、最大骨损距离、连通性指标

5. 参数配置默认值
参数	默认值	单位	说明
ROI Padding	8.0	mm	ROI 扩展
Target Spacing	0.5	mm	各向同性重采样
Dense Step	0.2	mm	曲线稠密采样
EDT PDL Threshold	1.5	mm	内槽壁筛选
Ray-Casting Step	0.2	mm	射线采样步长
Gradient Weight	on/off	—	是否启用灰度梯度权重
Skeleton Method	on/off	—	是否使用 Skeletonization
可视化 Tube Radius	0.0	mm	Slicer 曲线显示半径
6. 检查与验收基准

曲线连通性：每颗牙闭合连续

物理合理性：曲线位于牙体表面附近，高于 CEJ

断裂段数量 ≤1 个（可配置）

回贴验证：ABC 曲线回贴全口 CBCT 张量，位置正确

自动生成报告：曲线长度、最大骨损距离、连通性指标完整

7. 评估指标

Point-to-Curve Distance：手工 CEJ 或可选 ABC 点到预测 ABC 曲线距离

MeanDist

P95Dist

Success Rate @ τ：误差小于 τ 的点比例（τ=0.5/1.0/1.5 mm）

辅助指标：断裂段数量、曲线 z 方向漂移

8. 交付产物

全口 ABC 张量 Y

每颗牙 ABC 曲线文件（PLY/OBJ/CSV）

Slicer 对照文件（compare NIfTI）

自动评估报告 CSV

可视化 HTML Viewer

export_meta.json：体素统计、曲线长度、骨损指标
