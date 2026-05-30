import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置字体与后端
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ==========================================
# 左图：流形破坏 (锚点落入真实分布的真空区)
# ==========================================
# 1. 生成呈“弯月形”流形的真实连续特征数据
theta = np.linspace(0, np.pi, 30)
real_x = np.cos(theta) + np.random.normal(0, 0.05, 30)
real_y = np.sin(theta) + np.random.normal(0, 0.05, 30)

# 2. 计算传统单锚点 (几何均值)
anchor_x, anchor_y = np.mean(real_x), np.mean(real_y)

# 3. 绘制真实样本与锚点
ax1.scatter(real_x, real_y, c='#3498DB', s=60, edgecolors='black', label='真实稀疏样本 (复杂流形)', zorder=3)
ax1.scatter(anchor_x, anchor_y, c='#E74C3C', s=200, marker='*', edgecolors='black', label='单锚点 (几何均值)', zorder=4)

# 4. 模拟插值过程 (连线并在连线上生成合成样本)
for rx, ry in zip(real_x, real_y):
    # 画辐射状连线
    ax1.plot([rx, anchor_x], [ry, anchor_y], color='#95A5A6', linestyle='--', linewidth=0.8, alpha=0.6)
    # 在连线上随机插值生成新样本
    alpha = np.random.uniform(0.3, 0.7)
    syn_x = rx + alpha * (anchor_x - rx)
    syn_y = ry + alpha * (anchor_y - ry)
    ax1.scatter(syn_x, syn_y, c='#F39C12', s=40, marker='^', edgecolors='black', zorder=3)

# 仅为图例补充一个合成样本的图例项
ax1.scatter([], [], c='#F39C12', s=40, marker='^', edgecolors='black', label='合成样本 (脱离流形)')

ax1.set_title('图(a) 单锚点策略：无视拓扑结构，抹杀流形多样性', fontsize=14, weight='bold', pad=15)
ax1.annotate('锚点落入数据空白区\n合成样本完全偏离真实流形',
             xy=(anchor_x, anchor_y-0.1), xytext=(anchor_x-0.8, anchor_y-0.6),
             arrowprops=dict(facecolor='#C0392B', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, weight='bold', color='#C0392B')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.5)

# ==========================================
# 右图：异常值敏感 (锚点被拖拽，轨迹整体偏离)
# ==========================================
# 1. 拷贝真实数据，并加入一个极端的“游离异常值”
outlier_x, outlier_y = 2.5, 2.5
real_x_out = np.append(real_x, outlier_x)
real_y_out = np.append(real_y, outlier_y)

# 2. 重新计算被异常值污染的锚点
anchor_x_out, anchor_y_out = np.mean(real_x_out), np.mean(real_y_out)

# 3. 绘制真实样本、异常值与偏移后的锚点
ax2.scatter(real_x, real_y, c='#3498DB', s=60, edgecolors='black', label='真实稀疏样本', zorder=3)
ax2.scatter(outlier_x, outlier_y, c='#2ECC71', s=100, marker='D', edgecolors='black', label='游离异常值', zorder=4)
ax2.scatter(anchor_x_out, anchor_y_out, c='#E74C3C', s=200, marker='*', edgecolors='black', label='严重偏移的锚点', zorder=4)

# 4. 模拟偏离的插值过程
for rx, ry in zip(real_x, real_y):
    ax2.plot([rx, anchor_x_out], [ry, anchor_y_out], color='#95A5A6', linestyle='--', linewidth=0.8, alpha=0.6)
    alpha = np.random.uniform(0.3, 0.7)
    syn_x = rx + alpha * (anchor_x_out - rx)
    syn_y = ry + alpha * (anchor_y_out - ry)
    ax2.scatter(syn_x, syn_y, c='#F39C12', s=40, marker='^', edgecolors='black', zorder=3)

ax2.set_title('图(b) 单锚点策略：对游离异常值极其敏感，轨迹偏离', fontsize=14, weight='bold', pad=15)
ax2.annotate('异常值引发锚点剧烈偏移\n导致全部合成轨迹被强行拉扯',
             xy=(anchor_x_out, anchor_y_out), xytext=(anchor_x_out-1.2, anchor_y_out+0.5),
             arrowprops=dict(facecolor='#C0392B', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, weight='bold', color='#C0392B')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout(pad=3.0)
save_path = 'single_anchor_drawbacks.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()