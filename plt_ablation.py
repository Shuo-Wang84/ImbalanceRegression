import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib

# Set font and style
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 0.8
rcParams['grid.alpha'] = 0.3

# Fix matplotlib backend compatibility issue
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues

# Ablation experiment data
data = {
    'California housing': {
        'base': [0.3759, 0.1761, 0.5401, 1.0930],
        'FIPC': [0.2867, 0.1518, 0.3980, 0.7192],
        'Mixup': [0.3021, 0.1798, 0.4172, 0.7097],
        'ALL': [0.3061, 0.1731, 0.4121, 0.6561]
    },
    'Wine quality': {
        'base': [0.7665, 0.6556, 2.3849, 5.4584],
        'FIPC': [0.4311, 0.3660, 0.6443, 1.4276],
        'Mixup': [0.4741, 0.3952, 0.7841, 1.2327],
        'ALL': [0.4110, 0.3436, 0.6504, 0.9424]
    },
    'Air quality': {
        'base': [0.1959, 0.1405, 0.5027, 1.8149],
        'FIPC': [0.1862, 0.1495, 0.5440, 1.1954],
        'Mixup': [0.2440, 0.1915, 0.5042, 1.4973],
        'ALL': [0.2138, 0.1666, 0.4876, 0.8519]
    },
    'Concrete strength': {
        'base': [168.3235, 114.0193, 198.9478, 312.9959],
        'FIPC': [47.7877, 32.8201, 52.6587, 59.9426],
        'Mixup': [45.4117, 29.1585, 52.1942, 55.1270],
        'ALL': [44.4250, 40.2720, 47.7134, 51.8170]
    }
}

# Configuration labels and colors - 使用更加区分明显的颜色
configurations = ['base', 'FIPC', 'Mixup', 'ALL']
regions = ['ALL', 'Many-shot', 'Med-shot', 'Few-shot']

# 更加鲜明的颜色对比
colors = ['#FF4444', '#2196F3', '#FF9800', '#4CAF50']  # 红、蓝、橙、绿
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']  # 圆形、方形、三角形、菱形
marker_sizes = [10, 10, 10, 10]  # 增大标记尺寸

# 创建x轴位置，为每条线添加小的偏移以避免重叠
base_x = np.arange(len(regions))
offsets = [-0.15, -0.05, 0.05, 0.15]  # 为四条线设置不同的x偏移

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 增大图形尺寸
fig.suptitle('Ablation Study Results: MSE Performance Trends Across Different Configurations',
             fontsize=18, fontweight='bold', y=0.95)

# Dataset list
datasets = list(data.keys())

# Create line plot for each dataset
for idx, dataset in enumerate(datasets):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    # Plot lines for each configuration with offset
    for i, config in enumerate(configurations):
        values = data[dataset][config]
        x_positions = base_x + offsets[i]  # 添加x轴偏移

        # 绘制线条，增加线宽和透明度
        line = ax.plot(x_positions, values,
                       color=colors[i],
                       linestyle=line_styles[i],
                       marker=markers[i],
                       markersize=marker_sizes[i],
                       linewidth=3.5,  # 增加线宽
                       label=config,
                       alpha=0.9,  # 增加不透明度
                       markerfacecolor=colors[i],
                       markeredgecolor='white',
                       markeredgewidth=1.5,
                       zorder=5 - i)  # 控制绘制顺序

        # 添加数值标签在关键点
        for j, (x_pos, value) in enumerate(zip(x_positions, values)):
            if j == 3:  # Few-shot区间
                ax.annotate(f'{value:.2f}',
                            (x_pos, value),
                            textcoords="offset points",
                            xytext=(0, 15),
                            ha='center',
                            fontsize=10,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.4',
                                      facecolor=colors[i],
                                      alpha=0.7,
                                      edgecolor='white'))

    # Customize subplot
    ax.set_title(f'{dataset}', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Data Regions', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=12, fontweight='bold')

    # 设置x轴刻度和标签
    ax.set_xticks(base_x)
    ax.set_xticklabels(regions, fontsize=11)

    # 增强网格显示
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Set y-axis to start from 0 for better comparison
    y_max = max([max(data[dataset][config]) for config in configurations])
    ax.set_ylim(0, y_max * 1.15)

    # 设置x轴范围以显示偏移
    ax.set_xlim(-0.3, len(regions) - 0.7)

    # Add legend (only on first subplot) with better positioning
    if idx == 0:
        ax.legend(loc='upper left', frameon=True, fancybox=True,
                  shadow=True, fontsize=11,
                  bbox_to_anchor=(0.02, 0.98))

    # 添加轻微的背景色区分
    ax.axvspan(-0.5, 3.5, alpha=0.03, color='blue')

    # Add improvement text box with better styling
    base_few = data[dataset]['base'][3]
    all_few = data[dataset]['ALL'][3]
    improvement = ((base_few - all_few) / base_few) * 100

    textstr = f'Few-shot\nImprovement:\n{improvement:.1f}%'
    props = dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                 alpha=0.8, edgecolor='navy', linewidth=1)
    ax.text(0.02, 0.75, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontweight='bold')

    # 增强坐标轴样式
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# Adjust layout with more spacing
plt.tight_layout(rect=[0, 0.04, 1, 0.92])

# Add overall description with better formatting
fig.text(0.5, 0.02,
         'Note: Lower MSE values indicate better performance. Lines are horizontally offset for clarity. '
         'FIPC: Feature-aware Imbalanced Prediction Calibration; Mixup: Data augmentation; ALL: Complete SSDIR framework',
         ha='center', fontsize=10, style='italic', weight='bold')

# Save figures with higher DPI
plt.savefig('ablation_study_line_plots_improved.png', dpi=400, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('ablation_study_line_plots_improved.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Display figure (use try-except to handle backend issues)
try:
    plt.show()
except Exception as e:
    print(f"Display warning: {e}")
    print("Figures have been saved successfully as PNG and PDF files.")

# Print detailed analysis
print("\n=== Ablation Experiment Trend Analysis (Improved Visualization) ===")
for dataset in datasets:
    print(f"\n{dataset}:")
    base_values = data[dataset]['base']
    all_values = data[dataset]['ALL']

    print(f"  Performance Trend (base → ALL):")
    for i, region in enumerate(regions):
        improvement = ((base_values[i] - all_values[i]) / base_values[i]) * 100
        trend = "↓" if improvement > 0 else "↑"
        print(f"    {region}: {base_values[i]:.4f} → {all_values[i]:.4f} ({improvement:+.1f}%) {trend}")

print("\n=== Visualization Improvements ===")
print("✓ Added horizontal offset to separate overlapping lines")
print("✓ Increased line width and marker size for better visibility")
print("✓ Used more contrasting colors and distinct line styles")
print("✓ Enhanced marker styling with white edges")
print("✓ Improved annotation positioning and styling")
print("✓ Better legend and grid formatting")
print("\nFigures saved as:")
print("- ablation_study_line_plots_improved.png (high resolution)")
print("- ablation_study_line_plots_improved.pdf (vector format)")