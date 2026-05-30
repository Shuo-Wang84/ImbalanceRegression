import matplotlib

matplotlib.use('TkAgg')  # 防崩溃护身符
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# ================= 数据准备 =================
regions = ['All', 'Many-shot', 'Med-shot', 'Few-shot']
x = np.arange(len(regions))

# 混凝土强度数据集 MSE 数据
mse_base = [168.323, 114.019, 198.947, 312.995]
mse_smoter = [160.913, 131.897, 176.230, 241.617]
mse_smogn = [170.868, 143.508, 190.878, 253.374]
mse_sirn = [171.166, 123.654, 185.511, 309.089]
mse_lds = [76.427, 78.416, 68.180, 94.223]
mse_ssdir_old = [44.425, 40.272, 47.713, 51.817]
mse_ssdir_new = [43.791, 50.754, 26.062, 40.036]

# 混凝土强度数据集 MAE 数据
mae_base = [10.426, 8.334, 12.047, 15.434]
mae_smoter = [10.119, 8.816, 11.167, 13.238]
mae_smogn = [10.448, 9.187, 11.506, 13.476]
mae_sirn = [10.493, 8.561, 11.821, 15.073]
mae_lds = [6.549, 6.882, 6.035, 7.128]
mae_ssdir_old = [5.158, 4.760, 5.455, 5.892]
mae_ssdir_new = [5.223, 5.748, 3.735, 4.993]


# 统一的绘图函数
def plot_metric(data_dict, metric_name, base_data, new_data, y_max):
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制各基线折线
    ax.plot(x, data_dict['Base'], marker='o', markersize=10, linewidth=3, color='#ff6666', label='Base')
    ax.plot(x, data_dict['SMOTER'], marker='^', markersize=10, linewidth=2.5, linestyle='--', color='#f5a623',
            label='SMOTER')
    ax.plot(x, data_dict['SMOGN'], marker='s', markersize=10, linewidth=2.5, linestyle='--', color='#a682c0',
            label='SMOGN')
    ax.plot(x, data_dict['SIRN'], marker='*', markersize=12, linewidth=2.5, linestyle='--', color='#7bbce7',
            label='SIRN')
    ax.plot(x, data_dict['LDS'], marker='D', markersize=9, linewidth=2.5, linestyle=':', color='#999999', label='LDS')
    ax.plot(x, data_dict['SSDIR (Old)'], marker='v', markersize=10, linewidth=2.5, linestyle='-.', color='#a2d192',
            label='SSDIR (Old)')

    # 突出显示我们的方法
    ax.plot(x, data_dict['SSDIR (New)'], marker='*', markersize=16, linewidth=3.5, linestyle='-.', color='#008000',
            label='SSDIR (New)')

    # 标注 Few-shot 区域的具体数值
    # 对于数值较大的数据集，动态调整标签位置偏移量
    offset_up = y_max * 0.03
    offset_down = y_max * 0.05

    ax.text(3, base_data[3] + offset_up, f'{base_data[3]:.3f}', ha='center', va='bottom', fontsize=14, weight='bold',
            bbox=dict(facecolor='#ff6666', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    ax.text(3, new_data[3] - offset_down, f'{new_data[3]:.3f}', ha='center', va='top', fontsize=14, weight='bold',
            bbox=dict(facecolor='#a2d192', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))

    # 标注改善率文本框
    improvement = (base_data[3] - new_data[3]) / base_data[3] * 100
    props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='navy')
    ax.text(0.02, 0.45, f'Few-shot\nImprovement:\n{improvement:.1f}%', transform=ax.transAxes, fontsize=13,
            weight='bold', verticalalignment='center', bbox=props)

    # 图表格式设置
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=13)
    ax.set_ylabel(metric_name, fontsize=14, weight='bold')
    ax.set_xlabel('Data Regions', fontsize=14, weight='bold')
    ax.set_title(f'Concrete Strength Dataset ({metric_name})', fontsize=18, weight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, y_max)
    ax.legend(loc='upper left', fontsize=12, shadow=True, fancybox=True)
    plt.tight_layout()
    plt.show()


# 将数据打包并调用绘图
mse_dict = {'Base': mse_base, 'SMOTER': mse_smoter, 'SMOGN': mse_smogn, 'SIRN': mse_sirn, 'LDS': mse_lds,
            'SSDIR (Old)': mse_ssdir_old, 'SSDIR (New)': mse_ssdir_new}
mae_dict = {'Base': mae_base, 'SMOTER': mae_smoter, 'SMOGN': mae_smogn, 'SIRN': mae_sirn, 'LDS': mae_lds,
            'SSDIR (Old)': mae_ssdir_old, 'SSDIR (New)': mae_ssdir_new}

# 针对混凝土强度数据集巨大的量级，调整 y_max
plot_metric(mse_dict, 'MSE', mse_base, mse_ssdir_new, 350.0)  # 绘制 MSE
plot_metric(mae_dict, 'MAE', mae_base, mae_ssdir_new, 18.0)  # 绘制 MAE