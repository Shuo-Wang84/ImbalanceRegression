import matplotlib

matplotlib.use('TkAgg')  # 加上这句，完美避开 PyCharm 报错
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# ================= 数据准备 =================
regions = ['All', 'Many-shot', 'Med-shot', 'Few-shot']
x = np.arange(len(regions))

# 空气质量数据集 MSE 数据
mse_base = [0.195, 0.140, 0.502, 1.814]
mse_smoter = [0.195, 0.147, 0.451, 1.568]
mse_smogn = [0.213, 0.171, 0.422, 1.602]
mse_sirn = [0.194, 0.159, 0.393, 1.215]
mse_lds = [0.255, 0.569, 0.613, 0.978]
mse_ssdir_old = [0.213, 0.166, 0.487, 0.851]
mse_ssdir_new = [0.274, 0.251, 0.312, 0.985]

# 空气质量数据集 MAE 数据
mae_base = [0.279, 0.239, 0.520, 1.058]
mae_smoter = [0.276, 0.249, 0.482, 0.967]
mae_smogn = [0.301, 0.273, 0.463, 0.988]
mae_sirn = [0.273, 0.245, 0.457, 0.787]
mae_lds = [0.344, 0.627, 0.601, 0.792]
mae_ssdir_old = [0.339, 0.287, 0.511, 0.662]
mae_ssdir_new = [0.396, 0.392, 0.399, 0.760]


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
    ax.set_title(f'Air Quality Dataset ({metric_name})', fontsize=18, weight='bold', pad=20)
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

# 针对空气质量数据集，调整 y_max 的自适应高度
plot_metric(mse_dict, 'MSE', mse_base, mse_ssdir_new, 2.1)  # 绘制 MSE
plot_metric(mae_dict, 'MAE', mae_base, mae_ssdir_new, 1.3)  # 绘制 MAE