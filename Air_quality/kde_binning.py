import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def kde_adaptive_binning(y, num_bins=50, bandwidth='scott', min_density_threshold=0.1, plot=True):
    """
    使用核密度估计(KDE)进行自适应分箱

    参数:
    y : pd.Series 或 np.ndarray
        标签数据
    num_bins : int
        目标分箱数量，最终分箱数可能会有所不同
    bandwidth : str 或 float
        KDE的带宽参数，可以是'scott'、'silverman'或具体数值
    min_density_threshold : float
        最小密度阈值，用于确定是否需要更细的分箱
    plot : bool
        是否绘制密度图和分箱结果

    返回:
    bin_counts : pd.Series
        每个区间的样本数量
    bins : np.ndarray
        区间边界
    """
    # 确保y是一维数组
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = np.asarray(y)

    y_values = y_values.reshape(-1, 1)  # KDE需要2D数组

    # 计算数据范围
    y_min, y_max = np.min(y_values), np.max(y_values)
    y_range = y_max - y_min

    # 拟合KDE模型
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(y_values)

    # 在整个范围内生成均匀分布的点进行密度估计
    x_grid = np.linspace(y_min - 0.05 * y_range, y_max + 0.05 * y_range, 1000).reshape(-1, 1)
    log_dens = kde.score_samples(x_grid)
    density = np.exp(log_dens)

    # 归一化密度
    density = density / np.max(density)

    # 根据密度自适应地确定分箱边界
    bins = [y_min]
    current_pos = y_min

    # 计算平均步长
    avg_step = y_range / num_bins

    while current_pos < y_max:
        # 找到当前位置的密度
        idx = np.argmin(np.abs(x_grid.flatten() - current_pos))
        current_density = density[idx]

        # 根据密度调整步长：密度高的区域步长小，密度低的区域步长大
        if current_density > min_density_threshold:
            # 密度越高，步长越小，最小为avg_step的0.5倍
            step = avg_step * (1 - 0.5 * (current_density - min_density_threshold) / (1 - min_density_threshold))
        else:
            # 密度低于阈值，步长变大，最大为avg_step的2倍
            step = avg_step * (1 + (min_density_threshold - current_density) / min_density_threshold)

        # 确保步长在合理范围内
        step = max(avg_step * 0.5, min(step, avg_step * 2))

        # 更新位置
        current_pos += step
        if current_pos < y_max:  # 避免添加超出范围的边界
            bins.append(current_pos)

    # 确保最后一个边界是y_max
    if bins[-1] < y_max:
        bins.append(y_max)

    # 转换为numpy数组
    bins = np.array(bins)

    # 使用pandas的cut函数进行分箱
    y_binned = pd.cut(y, bins, include_lowest=True)
    bin_counts = y_binned.value_counts().sort_index()

    # 可视化
    if plot:
        plt.figure(figsize=(12, 6))

        # 绘制KDE曲线
        plt.subplot(2, 1, 1)
        plt.plot(x_grid, density, 'r-', label='KDE')
        plt.title('核密度估计')
        plt.ylabel('密度')

        # 绘制分箱结果
        plt.subplot(2, 1, 2)
        plt.hist(y, bins=bins, alpha=0.5, density=True)
        plt.plot(x_grid, density, 'r-', label='KDE')

        # 标记分箱边界
        for b in bins:
            plt.axvline(b, color='k', linestyle='--', alpha=0.3)

        plt.title(f'KDE自适应分箱结果 (分箱数: {len(bins) - 1})')
        plt.xlabel('值')
        plt.ylabel('频率/密度')
        plt.tight_layout()
        plt.show()

    return bin_counts, bins


def calculate_kde_sampling_rates(y, num_bins=50, bandwidth='scott', min_density_threshold=0.1, plot=False):
    """
    根据KDE自适应分箱计算每个区间需要补充的样本数量

    参数:
    y : pd.Series
        标签数据
    num_bins, bandwidth, min_density_threshold, plot :
        与kde_adaptive_binning函数参数相同

    返回:
    sampling_counts : pd.Series
        每个区间需要补充的样本数量
    bins : np.ndarray
        标签区间
    max_samples_in_bin : int
        最大区间样本数
    """
    # 计算自适应分箱
    bin_counts, bins = kde_adaptive_binning(y, num_bins, bandwidth, min_density_threshold, plot)

    # 找到最大区间样本数
    max_samples_in_bin = bin_counts.max()

    # 计算每个区间需要补充的样本数
    sampling_counts = max_samples_in_bin - bin_counts

    return sampling_counts, bins, max_samples_in_bin