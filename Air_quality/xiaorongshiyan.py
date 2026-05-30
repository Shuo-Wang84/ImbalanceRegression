import matplotlib

matplotlib.use('Agg')  # 设置非交互式后端，避免tostring_rgb错误

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# 设置图表样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用支持英文的字体

# Air Quality数据集消融实验数据
data = {
    'BASE': {
        'MSE': {'All': 0.2085, 'Many': 0.1496, 'Med': 0.7892, 'Few': 1.7355},
        'MAE': {'All': 0.3046, 'Many': 0.2676, 'Med': 0.6841, 'Few': 1.1043}
    },
    'BASE+RS': {
        'MSE': {'All': 0.1862, 'Many': 0.1316, 'Med': 0.5138, 'Few': 1.3281},
        'MAE': {'All': 0.2705, 'Many': 0.2433, 'Med': 0.5269, 'Few': 0.9648}
    },
    'BASE+DE': {
        'MSE': {'All': 0.2062, 'Many': 0.1456, 'Med': 0.5493, 'Few': 2.3029},
        'MAE': {'All': 0.3051, 'Many': 0.2653, 'Med': 0.5464, 'Few': 1.0365}
    },
    'BASE+RS+DE': {
        'MSE': {'All': 0.2217, 'Many': 0.1811, 'Med': 0.4619, 'Few': 1.1967},
        'MAE': {'All': 0.3373, 'Many': 0.3095, 'Med': 0.5055, 'Few': 0.8468}
    }
}

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Air Quality Dataset: Ablation Study Results', fontsize=18, fontweight='bold', y=0.95)

# 定义颜色和标记
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
markers = ['o', 's', '^', 'D']
linestyles = ['-', '--', '-.', ':']
regions = ['All', 'Many', 'Med', 'Few']
methods = list(data.keys())

# 绘制MSE - All regions
for i, method in enumerate(methods):
    mse_values = [data[method]['MSE'][region] for region in regions]
    ax1.plot(regions, mse_values, marker=markers[i], color=colors[i],
             linewidth=2.5, markersize=8, label=method, linestyle=linestyles[i])

ax1.set_title('MSE Across All Regions', fontweight='bold', pad=20)
ax1.set_xlabel('Region', fontweight='bold')
ax1.set_ylabel('MSE', fontweight='bold')
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, max([max(data[method]['MSE'].values()) for method in methods]) * 1.1)

# 绘制MAE - All regions
for i, method in enumerate(methods):
    mae_values = [data[method]['MAE'][region] for region in regions]
    ax2.plot(regions, mae_values, marker=markers[i], color=colors[i],
             linewidth=2.5, markersize=8, label=method, linestyle=linestyles[i])

ax2.set_title('MAE Across All Regions', fontweight='bold', pad=20)
ax2.set_xlabel('Region', fontweight='bold')
ax2.set_ylabel('MAE', fontweight='bold')
ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, max([max(data[method]['MAE'].values()) for method in methods]) * 1.1)

# 绘制MSE改进对比（相对于BASE）
base_mse = data['BASE']['MSE']
for i, method in enumerate(methods[1:], 1):  # 跳过BASE
    improvements = [(base_mse[region] - data[method]['MSE'][region]) / base_mse[region] * 100
                    for region in regions]
    ax3.plot(regions, improvements, marker=markers[i], color=colors[i],
             linewidth=2.5, markersize=8, label=method, linestyle=linestyles[i])

ax3.set_title('MSE Improvement over BASE (%)', fontweight='bold', pad=20)
ax3.set_xlabel('Region', fontweight='bold')
ax3.set_ylabel('Improvement (%)', fontweight='bold')
ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='red', linestyle='-', alpha=0.5, linewidth=1)

# 绘制MAE改进对比（相对于BASE）
base_mae = data['BASE']['MAE']
for i, method in enumerate(methods[1:], 1):  # 跳过BASE
    improvements = [(base_mae[region] - data[method]['MAE'][region]) / base_mae[region] * 100
                    for region in regions]
    ax4.plot(regions, improvements, marker=markers[i], color=colors[i],
             linewidth=2.5, markersize=8, label=method, linestyle=linestyles[i])

ax4.set_title('MAE Improvement over BASE (%)', fontweight='bold', pad=20)
ax4.set_xlabel('Region', fontweight='bold')
ax4.set_ylabel('Improvement (%)', fontweight='bold')
ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='red', linestyle='-', alpha=0.5, linewidth=1)

# 调整布局
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 保存图片
plt.savefig('air_quality_ablation_results_english.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('air_quality_ablation_results_english.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

# 显示图片（如果需要的话可以注释掉）
# plt.show()

# 创建性能改进热力图
fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Air Quality Dataset: Performance Improvement Heatmap', fontsize=16, fontweight='bold')

# MSE改进热力图
mse_improvements = []
for method in methods[1:]:
    improvements = [(base_mse[region] - data[method]['MSE'][region]) / base_mse[region] * 100
                    for region in regions]
    mse_improvements.append(improvements)

mse_df = pd.DataFrame(mse_improvements, index=methods[1:], columns=regions)
sns.heatmap(mse_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax5, cbar_kws={'label': 'Improvement (%)'},
            linewidths=0.5, linecolor='white')
ax5.set_title('MSE Improvement over BASE (%)', fontweight='bold', pad=20)
ax5.set_xlabel('Region', fontweight='bold')
ax5.set_ylabel('Method', fontweight='bold')

# MAE改进热力图
mae_improvements = []
for method in methods[1:]:
    improvements = [(base_mae[region] - data[method]['MAE'][region]) / base_mae[region] * 100
                    for region in regions]
    mae_improvements.append(improvements)

mae_df = pd.DataFrame(mae_improvements, index=methods[1:], columns=regions)
sns.heatmap(mae_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax6, cbar_kws={'label': 'Improvement (%)'},
            linewidths=0.5, linecolor='white')
ax6.set_title('MAE Improvement over BASE (%)', fontweight='bold', pad=20)
ax6.set_xlabel('Region', fontweight='bold')
ax6.set_ylabel('Method', fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 保存热力图
plt.savefig('air_quality_improvement_heatmap_english.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('air_quality_improvement_heatmap_english.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

# 显示热力图（如果需要的话可以注释掉）
# plt.show()

print("Air Quality Dataset Ablation Study Analysis:")
print("=" * 50)

# 分析各方法的性能
print("\n1. Overall Performance (All Region):")
for method in methods:
    mse_all = data[method]['MSE']['All']
    mae_all = data[method]['MAE']['All']
    print(f"   {method}: MSE={mse_all:.4f}, MAE={mae_all:.4f}")

# 分析最佳改进
print("\n2. Best Improvements over BASE:")
for region in regions:
    best_mse_method = min(methods[1:], key=lambda m: data[m]['MSE'][region])
    best_mae_method = min(methods[1:], key=lambda m: data[m]['MAE'][region])

    mse_improvement = (base_mse[region] - data[best_mse_method]['MSE'][region]) / base_mse[region] * 100
    mae_improvement = (base_mae[region] - data[best_mae_method]['MAE'][region]) / base_mae[region] * 100

    print(f"   {region} Region:")
    print(f"     Best MSE: {best_mse_method} ({mse_improvement:.1f}% improvement)")
    print(f"     Best MAE: {best_mae_method} ({mae_improvement:.1f}% improvement)")

# 分析各方法的特点
print("\n3. Method Analysis:")
print("   BASE+RS: Random Sampling technique")
print("   BASE+DE: Data Enhancement technique")
print("   BASE+RS+DE: Combined Random Sampling and Data Enhancement")

print("\n4. Key Observations:")
print("   - RS (Random Sampling) shows consistent improvements across most regions")
print("   - DE (Data Enhancement) has mixed effects, sometimes degrading performance")
print("   - Combined RS+DE shows good performance in Few region for MAE")
print("   - Few region benefits most from the proposed techniques")
print("   - Many region shows the most stable performance across all methods")

print("\nGraphs saved as:")
print("- air_quality_ablation_results_english.png/pdf")
print("- air_quality_improvement_heatmap_english.png/pdf")