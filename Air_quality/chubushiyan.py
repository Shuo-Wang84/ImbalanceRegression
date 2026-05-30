import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置后端和字体，防止报错和中文乱码
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# =================================================================
# 1. 严格采用用户提供的真实“空气质量 (Air Quality)” MSE 均值数据
# =================================================================
methods = ['Base', 'SMOTER', 'SMOGN', 'SIRN', 'LDS', 'SSDIR(MLP)', 'SSDIR(XGboost)']

# 按区间严格剥离真实 MSE 数据
mse_all = [0.195, 0.195, 0.213, 0.194, 0.255, 0.213, 0.184]
mse_many = [0.140, 0.147, 0.171, 0.159, 0.569, 0.166, 0.133]
mse_med = [0.502, 0.451, 0.422, 0.393, 0.613, 0.487, 1.697]
mse_few = [1.814, 1.568, 1.602, 1.215, 0.978, 0.851, 1.485]

# 将标题和对应的数据打包
datasets = {
    'All (总体)': mse_all,
    'Many-shot (多数派区间)': mse_many,
    'Med-shot (中等频次区间)': mse_med,
    'Few-shot (少数派区间)': mse_few
}

# =================================================================
# 2. 沿用高级清爽的马卡龙/莫兰迪配色方案 (确保论文视觉统一)
# =================================================================
colors = [
    '#7CB9E8',  # Base: 柔和浅蓝
    '#A3C1AD',  # SMOTER: 浅灰绿
    '#F4C2C2',  # SMOGN: 樱花粉红
    '#FCE883',  # SIRN: 柔和淡黄
    '#CDB8D1',  # LDS: 淡香芋紫
    '#FF9F00',  # SSDIR(MLP): 活力明橙 (极度突出)
    '#FF6B6B'  # SSDIR(XGboost): 珊瑚红 (突出)
]

# =================================================================
# 3. 创建 2x2 的子图画布
# =================================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
axes = axes.flatten()

# =================================================================
# 4. 循环遍历绘制 4 个独立区间的柱状图
# =================================================================
for i, (title, data) in enumerate(datasets.items()):
    ax = axes[i]
    x = np.arange(len(methods))

    # 绘制单张子图的柱子
    bars = ax.bar(x, data, color=colors, edgecolor='#555555', alpha=0.9, width=0.6, linewidth=1.2)

    # 设置标题和 Y 轴 (专门标明是空气质量数据集)
    ax.set_title(f"Air Quality: {title}", fontsize=16, weight='bold', pad=15)
    ax.set_ylabel('均方误差 (MSE)', fontsize=13)

    # 设置 X 轴标签，倾斜 35 度防止重叠
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=12, weight='bold')

    # 添加背景虚线网格，增加学术感
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # 动态为当前子图留出顶部空间 (最高柱子高度的 1.15 倍，防止文字重叠)
    ax.set_ylim(0, max(data) * 1.15)

    # 在每根柱子上方精准标注具体数值
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 垂直向上偏移 4 个像素
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

# =================================================================
# 5. 全局排版优化并保存
# =================================================================
plt.tight_layout(pad=3.0)

save_path = 'air_quality_mse_4subplots.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 空气质量数据集 2x2 子图版的高清图表已生成并保存为: {save_path}")

plt.show()