import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置字体与后端
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ==========================================
# 左图：不平衡分类的特征空间 (存在坚硬的决策断层)
# ==========================================
np.random.seed(42)
# 多数类 (蓝色，数量多)
X_maj = np.random.normal(loc=[-1.5, -1.5], scale=0.6, size=(200, 2))
# 少数类 (红色，数量极少)
X_min = np.random.normal(loc=[1.5, 1.5], scale=0.4, size=(15, 2))

ax1.scatter(X_maj[:, 0], X_maj[:, 1], c='#3498DB', edgecolors='white', s=60, label='多数样本类别 (Class 0)')
ax1.scatter(X_min[:, 0], X_min[:, 1], c='#E74C3C', edgecolors='black', s=80, label='少数样本类别 (Class 1)')

# 绘制一条极其清晰的线性决策边界
x_bounds = np.array([-3, 3])
y_bounds = -x_bounds + 0.5
ax1.plot(x_bounds, y_bounds, color='#2C3E50', linestyle='--', linewidth=2.5, label='坚硬的物理决策边界')

ax1.set_title('图(a) 分类任务特征空间：存在明确的物理断层', fontsize=15, weight='bold', pad=15)
ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.5, 3.5)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)

# ==========================================
# 右图：不平衡回归的特征空间 (连续渐变与高密引力)
# ==========================================
# 构建一个回归特征空间：数值 y 由空间位置决定，呈现非线性起伏
x_grid = np.linspace(-3.5, 3.5, 100)
y_grid = np.linspace(-3.5, 3.5, 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
# 背景回归曲面（等高线）
Z_mesh = np.sin(X_mesh) + np.cos(Y_mesh)

# 绘制连续等高线背景
contour = ax2.contourf(X_mesh, Y_mesh, Z_mesh, levels=20, cmap='YlGnBu', alpha=0.4)
cbar = plt.colorbar(contour, ax=ax2)
cbar.set_label('连续回归目标值 (Y)', fontsize=12)

# 多数派样本（聚集在中间平缓区域，数量庞大）
X_reg_many = np.random.normal(loc=[0, 0], scale=0.8, size=(250, 2))
Y_reg_many = np.sin(X_reg_many[:, 0]) + np.cos(X_reg_many[:, 1])

# 少数派极端样本（散落在边缘峰值区域，极度稀缺）
X_reg_few = np.random.normal(loc=[-2, 2.5], scale=0.2, size=(8, 2))
Y_reg_few = np.sin(X_reg_few[:, 0]) + np.cos(X_reg_few[:, 1])

ax2.scatter(X_reg_many[:, 0], X_reg_many[:, 1], c=Y_reg_many, cmap='YlGnBu', edgecolors='black', s=40, label='高密常规区间')
ax2.scatter(X_reg_few[:, 0], X_reg_few[:, 1], c='#E74C3C', edgecolors='black', s=90, marker='*', label='极度稀缺区间')

# 核心标注：高密区域对少数派的“预测干扰/引力”
ax2.annotate('边界模糊：\n海量常规样本的梯度回传\n强行“拉扯”极端值的预测面',
             xy=(-1.5, 2.0), xytext=(0.5, 2.8),
             arrowprops=dict(facecolor='#C0392B', shrink=0.05, width=2, headwidth=10, connectionstyle="arc3,rad=-0.2"),
             fontsize=12, weight='bold', color='#C0392B')

ax2.set_title('图(b) 回归任务特征空间：边界模糊与相邻干涉', fontsize=15, weight='bold', pad=15)
ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-3.5, 3.5)
ax2.legend(loc='lower left', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout(pad=3.0)
save_path = 'feature_space_classification_vs_regression.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()