import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置后端和字体，防止报错和中文乱码
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# =================================================================
# 1. 严格采用用户提供的真实 MSE 均值数据
# =================================================================
methods = ['Base', 'SMOTER', 'SMOGN', 'SIRN', 'LDS', 'SSDIR(MLP)', 'SSDIR(XGboost)']

# 按区间剥离真实数据，方便独立画图
mse_all = [0.375, 0.354, 0.498, 0.396, 0.350, 0.306, 0.259]
mse_many = [0.176, 0.228, 0.371, 0.196, 0.530, 0.173, 0.178]
mse_med = [0.540, 0.461, 0.588, 0.415, 0.817, 0.412, 0.344]
mse_few = [1.093, 0.794, 1.014, 0.812, 0.932, 0.656, 0.661]

# 将标题和对应的数据打包
datasets = {
    'All (总体)': mse_all,
    'Many-shot (多数派区间)': mse_many,
    'Med-shot (中等频次区间)': mse_med,
    'Few-shot (少数派区间)': mse_few
}

# =================================================================
# 2. 参考用户图片的高级清爽配色方案
# =================================================================
# 采用高明度、低饱和度的高级学术色系，清新且对比鲜明
colors = [
    '#7CB9E8',  # Base: 柔和浅蓝
    '#A3C1AD',  # SMOTER: 浅灰绿
    '#F4C2C2',  # SMOGN: 樱花粉红
    '#FCE883',  # SIRN: 柔和淡黄
    '#CDB8D1',  # LDS: 淡香芋紫
    '#FF9F00',  # SSDIR(MLP): 活力明橙 (突出)
    '#FF6B6B'  # SSDIR(XGboost): 珊瑚红 (突出)
]

# =================================================================
# 3. 创建 2x2 的子图画布
# =================================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
axes = axes.flatten()  # 将 2x2 的矩阵拉平为一维数组，方便遍历

# =================================================================
# 4. 循环遍历绘制 4 个独立区间的柱状图
# =================================================================
for i, (title, data) in enumerate(datasets.items()):
    ax = axes[i]
    x = np.arange(len(methods))

    # 绘制单张子图的柱子
    bars = ax.bar(x, data, color=colors, edgecolor='#555555', alpha=0.9, width=0.6, linewidth=1.2)

    # 设置标题和 Y 轴
    ax.set_title(title, fontsize=16, weight='bold', pad=15)
    ax.set_ylabel('均方误差 (MSE)', fontsize=13)

    # 设置 X 轴标签，倾斜 35 度防止重叠
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=12, weight='bold')

    # 添加背景虚线网格，增加学术感
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # 动态为当前子图留出顶部空间 (最高柱子高度的 1.15 倍)
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
# 自动调整子图间距，防止文字重叠
plt.tight_layout(pad=3.0)

save_path = 'california_mse_4subplots.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 2x2 子图版的高清图表已生成并保存为: {save_path}")

plt.show()