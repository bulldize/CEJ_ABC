import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, IntSlider
import ipywidgets as widgets
import time


def get_unique_values_iterative(memmap_data, batch_size=1000000, verbose=True):
    """
    使用迭代器方法遍历memmap数组，返回所有出现的唯一值列表

    参数:
    - memmap_data: memory-mapped数组
    - batch_size: 每次处理的元素数量
    - verbose: 是否显示进度信息

    返回:
    - unique_values: 排序后的唯一值列表
    """
    if not isinstance(memmap_data, (np.memmap, np.ndarray)):
        raise ValueError("输入数据必须是numpy数组或memmap数组")

    # 获取数组信息
    total_elements = np.prod(memmap_data.shape)

    if verbose:
        print(f"数组形状: {memmap_data.shape}")
        print(f"数组类型: {memmap_data.dtype}")
        print(f"总元素数: {total_elements:,}")
        print(f"批次大小: {batch_size:,}")
        print("开始遍历数组...")

    # 使用集合存储唯一值
    unique_set = set()

    # 将数组展平为一维视图（不复制数据）
    flat_view = memmap_data.reshape(-1)

    start_time = time.time()

    # 分批处理
    for i in range(0, total_elements, batch_size):
        end_idx = min(i + batch_size, total_elements)

        # 获取当前批次的数据
        batch = flat_view[i:end_idx]

        # 将当前批次的唯一值添加到集合中
        unique_set.update(np.unique(batch))

        # 计算并显示进度
        if verbose and (i % (batch_size * 10) == 0 or end_idx == total_elements):
            elapsed = time.time() - start_time
            percent = (end_idx / total_elements) * 100
            print(f"进度: {end_idx:,}/{total_elements:,} ({percent:.1f}%) - "
                  f"已发现 {len(unique_set)} 种数值 - "
                  f"已用时: {elapsed:.1f}秒")

    end_time = time.time()

    if verbose:
        print(f"遍历完成！总用时: {end_time - start_time:.2f}秒")
        print(f"总共发现 {len(unique_set)} 种不同的数值")

    # 转换为排序后的列表
    unique_values = sorted(list(unique_set))

    return unique_values
# 加载NIfTI文件
def load_and_visualize_nifti(nifti_path, TYPE):
    # 加载文件
    img = nib.load(nifti_path)
    data = img.get_fdata()
    unique_values = get_unique_values_iterative(data)

    print(f"图像形状: {data.shape}")
    print(f"体素大小: {img.header.get_zooms()}")
    print(f"数据类型: {data.dtype}")

    # 获取三个方向的切片
    axial_slices = data.shape[2]
    sagittal_slices = data.shape[0]
    coronal_slices = data.shape[1]

    # 创建交互式可视化
    fig, axes = plt.subplots(figsize=(8, 8))

    def update_slice(axial_idx=axial_slices // 2):
        # 清除之前的图像

        axes.clear()

        # 显示三个方向的切片
        axes.imshow(data[:, :, axial_idx].T, cmap='gray', origin='lower')
        axes.set_title(f'Axial Slice {axial_idx}')
        axes.axis('off')

        # axes[1].imshow(data[sagittal_idx, :, :].T, cmap='gray', origin='lower')
        # axes[1].set_title(f'Sagittal Slice {sagittal_idx}')
        # axes[1].axis('off')
        #
        # axes[2].imshow(data[:, coronal_idx, :].T, cmap='gray', origin='lower')
        # axes[2].set_title(f'Coronal Slice {coronal_idx}')
        # axes[2].axis('off')

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./results/{TYPE}/{axial_idx}.png')

    for axial_idx in range(axial_slices):
        update_slice(axial_idx)
    # 在Jupyter中使用交互控件
    # interact(update_slice,
    #          axial_idx=IntSlider(min=0, max=axial_slices - 1, value=axial_slices // 2),
    #          sagittal_idx=IntSlider(min=0, max=sagittal_slices - 1, value=sagittal_slices // 2),
    #          coronal_idx=IntSlider(min=0, max=coronal_slices - 1, value=coronal_slices // 2))

    return data, img

# 使用示例
data, img = load_and_visualize_nifti('./data/TF_008/ToothFairy3F_008label.nii/ToothFairy3F_008label.nii', TYPE='label')
# data, img = load_and_visualize_nifti('./data/TF_008/ToothFairy3F_008_volume.nii/ToothFairy3F_008_volume.nii', TYPE='cbct')