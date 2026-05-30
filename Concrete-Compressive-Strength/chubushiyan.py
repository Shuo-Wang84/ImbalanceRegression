import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置后端和字体，防止报错和中文乱码
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# =================================================================
# 1. 严格提取“混凝土强度 (Concrete Strength)”的 MAE 均值数据
# =================================================================
methods = ['Base', 'SMOTER', 'SMOGN', 'SIRN', 'LDS', 'SSDIR(MLP)', 'SSDIR(XGboost)']

# 这里使用的是你表格右侧的 MAE 均值
mae_all = [10.426, 10.119, 10.448, 10.493, 6.549, 5.158, 5.259]
mae_many = [8.334, 8.816, 9.187, 8.561, 6.882, 4.760, 4.694]
mae_med = [12.047, 11.167, 11.506, 11.821, 6.035, 5.455, 5.264]
mae_few = [15.434, 13.238, 13.476, 15.073, 7.128, 5.892, 5.112]

# 将标题和对应的数据打包
datasets = {
    'All (总体)': mae_all,
    'Many-shot (多数派区间)': mae_many,
    'Med-shot (中等频次区间)': mae_med,
    'Few-shot (少数派区间)': mae_few
}

# =================================================================
# 2. 沿用高级清爽的马卡龙/莫兰迪配色方案
# =================================================================
colors = [
    '#7CB9E8',  # Base
    '#A3C1AD',  # SMOTER
    '#F4C2C2',  # SMOGN
    '#FCE883',  # SIRN
    '#CDB8D1',  # LDS
    '#FF9F00',  # SSDIR(MLP): 活力明橙
    '#FF6B6B'  # SSDIR(XGboost): 珊瑚红
]

# =================================================================
# 3. 创建 2x2 的子图画布
# =================================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
axes = axes.flatten()

# =================================================================
# 4. 循环遍历绘制 4 个独立区间的 MAE 柱状图
# =================================================================
for i, (title, data) in enumerate(datasets.items()):
    ax = axes[i]
    x = np.arange(len(methods))

    # 绘制柱状图
    bars = ax.bar(x, data, color=colors, edgecolor='#555555', alpha=0.9, width=0.6, linewidth=1.2)

    # 设置标题和 Y 轴 (改为了 MAE)
    ax.set_title(f"Concrete Strength: {title}", fontsize=16, weight='bold', pad=15)
    ax.set_ylabel('平均绝对误差 (MAE)', fontsize=13)

    # 设置 X 轴标签
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=12, weight='bold')

    # 添加网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Y轴上限自适应
    ax.set_ylim(0, max(data) * 1.15)

    # 在每根柱子上方精准标注 MAE 具体数值
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

# =================================================================
# 5. 全局排版优化并保存
# =================================================================
plt.tight_layout(pad=3.0)

save_path = 'concrete_strength_mae_4subplots.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 混凝土强度数据集(MAE)的 2x2 子图高清图表已生成并保存为: {save_path}")

plt.show()