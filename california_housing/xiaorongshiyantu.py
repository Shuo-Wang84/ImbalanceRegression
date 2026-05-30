import matplotlib
matplotlib.use('Agg')  # 添加这行，使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 消融实验数据
data = {
    'Methods': ['BASE', 'BASE+FIPC', 'BASE+LAMA', 'BASE+FIPC+LAMA'],
    'MSE_All': [0.3759, 0.2867, 0.3021, 0.3061],
    'MSE_Many': [0.1761, 0.1518, 0.1798, 0.1931],
    'MSE_Med': [0.5401, 0.3980, 0.4172, 0.4121],
    'MSE_Few': [1.093, 0.7692, 0.6797, 0.6561],
    'MAE_All': [0.4198, 0.3680, 0.3801, 0.3895],
    'MAE_Many': [0.3063, 0.2839, 0.3036, 0.3238],
    'MAE_Med': [0.5046, 0.4356, 0.4508, 0.4490],
    'MAE_Few': [0.8633, 0.6760, 0.6215, 0.6026]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置图形大小和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('加利福尼亚房价预测消融实验结果对比', fontsize=20, fontweight='bold', y=0.95)

# 定义颜色和标记样式
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
markers = ['o', 's', '^', 'D']
linestyles = ['-', '--', '-.', ':']
regions = ['All', 'Many', 'Med', 'Few']

# 绘制MSE图
ax1.set_title('MSE性能对比', fontsize=16, fontweight='bold', pad=20)
for i, method in enumerate(df['Methods']):
    mse_values = [df.iloc[i]['MSE_All'], df.iloc[i]['MSE_Many'],
                  df.iloc[i]['MSE_Med'], df.iloc[i]['MSE_Few']]
    ax1.plot(regions, mse_values,
             color=colors[i], marker=markers[i], linestyle=linestyles[i],
             linewidth=3, markersize=10, markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=colors[i],
             label=method, alpha=0.8)

ax1.set_xlabel('数据区域', fontsize=14, fontweight='bold')
ax1.set_ylabel('MSE值', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_ylim(0, max(df[['MSE_All', 'MSE_Many', 'MSE_Med', 'MSE_Few']].max()) * 1.1)

# 添加数值标注
for i, method in enumerate(df['Methods']):
    mse_values = [df.iloc[i]['MSE_All'], df.iloc[i]['MSE_Many'],
                  df.iloc[i]['MSE_Med'], df.iloc[i]['MSE_Few']]
    for j, val in enumerate(mse_values):
        ax1.annotate(f'{val:.3f}', (j, val),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))

# 绘制MAE图
ax2.set_title('MAE性能对比', fontsize=16, fontweight='bold', pad=20)
for i, method in enumerate(df['Methods']):
    mae_values = [df.iloc[i]['MAE_All'], df.iloc[i]['MAE_Many'],
                  df.iloc[i]['MAE_Med'], df.iloc[i]['MAE_Few']]
    ax2.plot(regions, mae_values,
             color=colors[i], marker=markers[i], linestyle=linestyles[i],
             linewidth=3, markersize=10, markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=colors[i],
             label=method, alpha=0.8)

ax2.set_xlabel('数据区域', fontsize=14, fontweight='bold')
ax2.set_ylabel('MAE值', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_ylim(0, max(df[['MAE_All', 'MAE_Many', 'MAE_Med', 'MAE_Few']].max()) * 1.1)

# 添加数值标注
for i, method in enumerate(df['Methods']):
    mae_values = [df.iloc[i]['MAE_All'], df.iloc[i]['MAE_Many'],
                  df.iloc[i]['MAE_Med'], df.iloc[i]['MAE_Few']]
    for j, val in enumerate(mae_values):
        ax2.annotate(f'{val:.3f}', (j, val),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# 保存图片
plt.savefig('california_housing_ablation_results.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('california_housing_ablation_results.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

# 显示图片
plt.show()

# 打印改进效果分析
print("\n=== 消融实验结果分析 ===")
print("\n1. MSE改进效果:")
base_mse = df.iloc[0]
for i in range(1, len(df)):
    method = df.iloc[i]['Methods']
    print(f"\n{method} vs BASE:")
    for region in ['All', 'Many', 'Med', 'Few']:
        base_val = base_mse[f'MSE_{region}']
        current_val = df.iloc[i][f'MSE_{region}']
        improvement = (base_val - current_val) / base_val * 100
        print(f"  {region}: {improvement:+.2f}% (从 {base_val:.4f} 到 {current_val:.4f})")

print("\n2. MAE改进效果:")
base_mae = df.iloc[0]
for i in range(1, len(df)):
    method = df.iloc[i]['Methods']
    print(f"\n{method} vs BASE:")
    for region in ['All', 'Many', 'Med', 'Few']:
        base_val = base_mae[f'MAE_{region}']
        current_val = df.iloc[i][f'MAE_{region}']
        improvement = (base_val - current_val) / base_val * 100
        print(f"  {region}: {improvement:+.2f}% (从 {base_val:.4f} 到 {current_val:.4f})")

# 创建性能改进热力图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('相对于BASE方法的性能改进热力图 (%)', fontsize=16, fontweight='bold')

# MSE改进矩阵
mse_improvement = []
mae_improvement = []
method_names = ['BASE+FIPC', 'BASE+LAMA', 'BASE+FIPC+LAMA']

for i in range(1, len(df)):
    mse_row = []
    mae_row = []
    for region in ['All', 'Many', 'Med', 'Few']:
        base_mse_val = df.iloc[0][f'MSE_{region}']
        current_mse_val = df.iloc[i][f'MSE_{region}']
        mse_improvement_val = (base_mse_val - current_mse_val) / base_mse_val * 100
        mse_row.append(mse_improvement_val)

        base_mae_val = df.iloc[0][f'MAE_{region}']
        current_mae_val = df.iloc[i][f'MAE_{region}']
        mae_improvement_val = (base_mae_val - current_mae_val) / base_mae_val * 100
        mae_row.append(mae_improvement_val)

    mse_improvement.append(mse_row)
    mae_improvement.append(mae_row)

# 绘制MSE改进热力图
im1 = ax1.imshow(mse_improvement, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=30)
ax1.set_title('MSE改进率 (%)', fontsize=14, fontweight='bold')
ax1.set_xticks(range(4))
ax1.set_xticklabels(regions)
ax1.set_yticks(range(3))
ax1.set_yticklabels(method_names)

# 添加数值标注
for i in range(3):
    for j in range(4):
        text = ax1.text(j, i, f'{mse_improvement[i][j]:.1f}%',
                        ha="center", va="center", color="black", fontweight='bold')

# 绘制MAE改进热力图
im2 = ax2.imshow(mae_improvement, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=30)
ax2.set_title('MAE改进率 (%)', fontsize=14, fontweight='bold')
ax2.set_xticks(range(4))
ax2.set_xticklabels(regions)
ax2.set_yticks(range(3))
ax2.set_yticklabels(method_names)

# 添加数值标注
for i in range(3):
    for j in range(4):
        text = ax2.text(j, i, f'{mae_improvement[i][j]:.1f}%',
                        ha="center", va="center", color="black", fontweight='bold')

# 添加颜色条
fig.colorbar(im1, ax=ax1, shrink=0.8)
fig.colorbar(im2, ax=ax2, shrink=0.8)

plt.tight_layout()
plt.savefig('california_housing_improvement_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()