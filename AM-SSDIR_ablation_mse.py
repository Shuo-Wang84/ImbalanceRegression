import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置字体与后端
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 构建 2x2 的矩阵画布
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# X轴标签与刻度
x_labels = ['All', 'Many-shot', 'Med-shot', 'Few-shot']
x_positions = np.arange(len(x_labels))

#========================================
# 1. California Housing Dataset
#========================================
ax1 = axes[0, 0]
base_ch = [0.375, 0.176, 0.540, 1.093]
fipc_ch = [0.288, 0.148, 0.326, 0.576]
mada_ch = [0.301, 0.179, 0.350, 0.507]
all_ch  = [0.293, 0.161, 0.338, 0.543]

ax1.plot(x_positions, base_ch, marker='o', color='#E74C3C', linestyle='-', linewidth=2, label='Base')
ax1.plot(x_positions, fipc_ch, marker='s', color='#3498DB', linestyle='--', linewidth=2, label='FIPC')
ax1.plot(x_positions, mada_ch, marker='^', color='#F39C12', linestyle='-.', linewidth=2, label='MADA')
ax1.plot(x_positions, all_ch, marker='*', color='#27AE60', linestyle='-.', linewidth=2.5, markersize=10, label='ALL')

ax1.set_title('California housing', fontsize=14, weight='bold')
ax1.set_ylabel('MSE', fontsize=12, weight='bold')
ax1.set_xticks(x_positions)
ax1.set_xticklabels(x_labels)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left', fontsize=11)

# 添加 Few-shot 改善率标签与极端值高亮
ax1.annotate('1.093', xy=(3, 1.093), xytext=(3, 1.15), ha='center', color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='#FFB6C1', ec='none'))
ax1.annotate('0.543', xy=(3, 0.543), xytext=(3, 0.48), ha='center', color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='#A9DFBF', ec='none'))
ax1.annotate('Few-shot\nImprovement:\n50.3%', xy=(0, 0.9), xytext=(0, 0.82), fontsize=11, weight='bold', bbox=dict(boxstyle='round,pad=0.4', fc='#D6EAF8', ec='#5DADE2', alpha=0.8))


#========================================
# 2. Wine Quality Dataset
#========================================
ax2 = axes[0, 1]
base_wq = [0.766, 0.655, 2.384, 5.458]
fipc_wq = [0.395, 0.352, 0.434, 0.782]
mada_wq = [0.458, 0.385, 0.543, 0.566]
all_wq  = [0.468, 0.383, 0.569, 0.466]

ax2.plot(x_positions, base_wq, marker='o', color='#E74C3C', linestyle='-', linewidth=2)
ax2.plot(x_positions, fipc_wq, marker='s', color='#3498DB', linestyle='--', linewidth=2)
ax2.plot(x_positions, mada_wq, marker='^', color='#F39C12', linestyle='-.', linewidth=2)
ax2.plot(x_positions, all_wq, marker='*', color='#27AE60', linestyle='-.', linewidth=2.5, markersize=10)

ax2.set_title('Wine quality', fontsize=14, weight='bold')
ax2.set_ylabel('MSE', fontsize=12, weight='bold')
ax2.set_xticks(x_positions)
ax2.set_xticklabels(x_labels)
ax2.grid(True, linestyle=':', alpha=0.6)

# 添加 Few-shot 改善率标签与极端值高亮
ax2.annotate('5.458', xy=(3, 5.458), xytext=(3, 5.65), ha='center', color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='#FFB6C1', ec='none'))
ax2.annotate('0.466', xy=(3, 0.466), xytext=(3, 0.15), ha='center', color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='#A9DFBF', ec='none'))
ax2.annotate('Few-shot\nImprovement:\n91.5%', xy=(0, 4.0), xytext=(0, 4.2), fontsize=11, weight='bold', bbox=dict(boxstyle='round,pad=0.4', fc='#D6EAF8', ec='#5DADE2', alpha=0.8))


#========================================
# 3. Air Quality Dataset
#========================================
ax3 = axes[1, 0]
base_aq = [0.195, 0.140, 0.502, 1.814]
fipc_aq = [0.197, 0.141, 0.310, 1.152]
mada_aq = [0.325, 0.327, 0.298, 1.196]
all_aq  = [0.274, 0.251, 0.312, 0.985]

ax3.plot(x_positions, base_aq, marker='o', color='#E74C3C', linestyle='-', linewidth=2)
ax3.plot(x_positions, fipc_aq, marker='s', color='#3498DB', linestyle='--', linewidth=2)
ax3.plot(x_positions, mada_aq, marker='^', color='#F39C12', linestyle='-.', linewidth=2)
ax3.plot(x_positions, all_aq, marker='*', color='#27AE60', linestyle='-.', linewidth=2.5, markersize=10)

ax3.set_title('Air quality', fontsize=14, weight='bold')
ax3.set_ylabel('MSE', fontsize=12, weight='bold')
ax3.set_xlabel('Data Regions', fontsize=12, weight='bold')
ax3.set_xticks(x_positions)
ax3.set_xticklabels(x_labels)
ax3.grid(True, linestyle=':', alpha=0.6)

# 添加 Few-shot 改善率标签与极端值高亮
ax3.annotate('1.814', xy=(3, 1.814), xytext=(3, 1.89), ha='center', color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='#FFB6C1', ec='none'))
ax3.annotate('0.985', xy=(3, 0.985), xytext=(3, 0.85), ha='center', color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='#A9DFBF', ec='none'))
ax3.annotate('Few-shot\nImprovement:\n45.7%', xy=(0, 1.3), xytext=(0, 1.35), fontsize=11, weight='bold', bbox=dict(boxstyle='round,pad=0.4', fc='#D6EAF8', ec='#5DADE2', alpha=0.8))


#========================================
# 4. Concrete Strength Dataset
#========================================
ax4 = axes[1, 1]
base_cs = [168.323, 114.019, 198.947, 312.995]
fipc_cs = [51.992, 56.712, 21.128, 56.077]
mada_cs = [45.900, 55.762, 19.816, 40.938]
all_cs  = [43.791, 50.754, 26.092, 40.036]

ax4.plot(x_positions, base_cs, marker='o', color='#E74C3C', linestyle='-', linewidth=2)
ax4.plot(x_positions, fipc_cs, marker='s', color='#3498DB', linestyle='--', linewidth=2)
ax4.plot(x_positions, mada_cs, marker='^', color='#F39C12', linestyle='-.', linewidth=2)
ax4.plot(x_positions, all_cs, marker='*', color='#27AE60', linestyle='-.', linewidth=2.5, markersize=10)

ax4.set_title('Concrete strength', fontsize=14, weight='bold')
ax4.set_ylabel('MSE', fontsize=12, weight='bold')
ax4.set_xlabel('Data Regions', fontsize=12, weight='bold')
ax4.set_xticks(x_positions)
ax4.set_xticklabels(x_labels)
ax4.grid(True, linestyle=':', alpha=0.6)

# 添加 Few-shot 改善率标签与极端值高亮
ax4.annotate('312.995', xy=(3, 312.995), xytext=(3, 330), ha='center', color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='#FFB6C1', ec='none'))
ax4.annotate('40.036', xy=(3, 40.036), xytext=(3, 15), ha='center', color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='#A9DFBF', ec='none'))
ax4.annotate('Few-shot\nImprovement:\n87.2%', xy=(0, 220), xytext=(0, 230), fontsize=11, weight='bold', bbox=dict(boxstyle='round,pad=0.4', fc='#D6EAF8', ec='#5DADE2', alpha=0.8))

plt.tight_layout(pad=4.0)
save_path = 'ablation_study_mse.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()