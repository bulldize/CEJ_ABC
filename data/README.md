# 数据文件夹说明与分析报告

**更新时间**: 2026-02-04
**状态**: ✅ **Ready for Development**

本目录包含了项目训练与验证所需的核心数据。本报告汇总了数据的结构分析、内容验证及空间映射逻辑。

---

## 1. 验证结论 (Verification Status)

| 验证项 | 状态 | 结论 | 备注 |
|---|---|---|---|
| **牙实例掩码** | ✅ 通过 | 遵循 FDI 标准 (牙体 11-48, 牙髓 111-148) | 预处理需执行 `label % 100` 合并 |
| **坐标一致性** | ✅ 通过 | 确认使用 **RAS (mm)** 物理坐标系 | Excel 边界与 NIfTI 物理空间完美吻合 |
| **空间标准化** | ✅ 通过 | 分辨率 `(0.3, 0.3, 0.3)` mm | 高分辨率各向同性，无需重采样 |

---

## 2. NIfTI 数据详解 (CBCT & Label)

位于 `TF_008/` 目录下，包含：
- **CBCT 影像**: `ToothFairy3F_008_volume.nii`
- **标签数据**: `ToothFairy3F_008label.nii`

### 2.1 规格参数
- **维度 (Shape)**: `(410, 410, 270)`
- **体素间距 (Spacing)**: `0.3mm` (Isotropic)
- **仿射矩阵 (Affine)**:
  ```text
  [[-0.3,  0. ,  0. , -0. ],
   [ 0. , -0.3,  0. , -0. ],
   [ 0. ,  0. ,  0.3,  0. ],
   [ 0. ,  0. ,  0. ,  1. ]]
  ```
  *注：负对角线元素表示坐标轴方向（标准 NIfTI 格式），无需手动干预。*

### 2.2 数据内容示例
- **CBCT 灰度 (Float64)**:
  ```text
  [[ -50.75,   41.27,  103.28],
   [-159.78,   37.26,   46.27],
   [-388.84,   -0.74,   23.26]]
  ```
- **Label 标签 (Float64)**:
  ```text
  [[7.0, 0.0, 0.0],
   [7.0, 0.0, 0.0],
   [7.0, 0.0, 0.0]]
  ```
  *7.0=牙槽骨, 0.0=背景*

---

## 3. 标注数据详解 (Excel)

### 3.1 稀疏点标注 (`cej_points_ras.xlsx`)
*(原名: points.xlsx)*

- **坐标系**: **RAS 物理坐标 (mm)**
- **用途**: CEJ 曲线的关键点 Ground Truth。
- **数据示例**:
  | 牙位 (Source) | 点位 (ID) | x (mm) | y (mm) | z (mm) |
  |---|---|---|---|---|
  | 27.mrk.json | 1 | 33.916 | 48.282 | 15.935 |
  | 27.mrk.json | 2 | 43.506 | 49.699 | 19.576 |
  | ... | ... | ... | ... | ... |

### 3.2 边界验证数据 (`scan_boundary_ras.xlsx`)
*(原名: CBCT_8_Boundary_Points.xlsx)*

- **用途**: 验证坐标系与物理范围。
- **验证结果**:
    - 记录的 8 个顶点范围 (`123x123x81 mm`) 与 NIfTI 物理尺寸完全一致。
    - 证实了 Excel 坐标记录的是体素边缘 (Edges)，与 NIfTI 中心 (Centers) 存在半个体素 (0.15mm) 的理论偏差，验证了坐标系的精确性。

---

## 4. 核心逻辑指南 (Development Guide)

### 4.1 坐标转换 (Coordinate Transform)
要在训练中正确使用标注点，**必须**执行以下转换：
1.  **输入**: `cej_points_ras.xlsx` 中的 `(x, y, z)` (RAS mm)。
2.  **转换**: 使用 NIfTI 提供的 `affine` 矩阵及其逆矩阵。
3.  **公式**:
    ```python
    # 伪代码
    nifti_img = nib.load(path)
    affine = nifti_img.affine
    physical_point = [x, y, z] # 来自 Excel
    
    # 映射到体素索引 (i, j, k)
    voxel_index = np.linalg.inv(affine) @ [*physical_point, 1]
    voxel_index = voxel_index[:3] # 取前三维
    ```
    *注意：严禁手动翻转坐标轴（如 -x），Affine 矩阵会自动处理所有方向问题。*

### 4.2 数据预处理 (Preprocessing)
1.  **Label 清洗**:
    - 合并: `label[label > 100] -= 100` (将 111 变为 11)。
    - 过滤: `label[label < 10] = 0` (清除牙槽骨等非牙体结构)。
    - 二值化: 生成 `Tooth_Mask` 用于 ROI 提取。

---

## 5. 现有资产
- `src/test.py`: 基础 NIfTI 可视化脚本。
- `data/`: 包含完整数据与验证表格。