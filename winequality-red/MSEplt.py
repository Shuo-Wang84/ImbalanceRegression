import matplotlib
matplotlib.use('TkAgg')  # 强制使用 Tkinter 独立窗口后端渲染
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# ================= 数据准备 =================
regions = ['All', 'Many-shot', 'Med-shot', 'Few-shot']
x = np.arange(len(regions))

# MSE 数据
mse_base = [0.766, 0.655, 2.384, 5.458]
mse_smoter = [0.654, 0.563, 2.008, 4.441]
mse_smogn = [0.642, 0.524, 2.513, 5.069]
mse_sirn = [0.998, 2.160, 0.998, 3.530]
mse_lds = [0.452, 0.380, 1.092, 3.774]
mse_ssdir_old = [0.411, 0.343, 0.650, 0.942]
mse_ssdir_new = [0.468, 0.383, 0.569, 0.466]

# MAE 数据
mae_base = [0.667, 0.623, 1.350, 2.296]
mae_smoter = [0.610, 0.569, 1.254, 2.074]
mae_smogn = [0.590, 0.539, 1.438, 2.194]
mae_sirn = [2.109, 1.001, 0.869, 1.878]
mae_lds = [0.512, 0.592, 0.839, 1.893]
mae_ssdir_old = [0.486, 0.492, 0.667, 0.488]
mae_ssdir_new = [0.529, 0.491, 0.572, 0.606]


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
    ax.text(3, base_data[3] + y_max * 0.03, f'{base_data[3]:.3f}', ha='center', va='bottom', fontsize=14, weight='bold',
            bbox=dict(facecolor='#ff6666', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    ax.text(3, new_data[3] - y_max * 0.05, f'{new_data[3]:.3f}', ha='center', va='top', fontsize=14, weight='bold',
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
    ax.set_title(f'Wine Quality Dataset ({metric_name})', fontsize=18, weight='bold', pad=20)
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

plot_metric(mse_dict, 'MSE', mse_base, mse_ssdir_new, 6.0)  # 绘制 MSE
plot_metric(mae_dict, 'MAE', mae_base, mae_ssdir_new, 2.5)  # 绘制 MAE