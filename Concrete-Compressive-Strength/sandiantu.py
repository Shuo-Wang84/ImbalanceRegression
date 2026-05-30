import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 强制使用TkAgg，避免PyCharm报错
matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# 1. 填入你真实跑出来的 csv 文件路径
# 注意把下面的路径替换为你自己电脑上真实的文件路径
base_df = pd.read_csv('ccs_base_predictions.csv')
ssdir_df = pd.read_csv(r'F:\SGIR\SGIR-main\plt画图\MLP\混凝土\自适应分箱\ccs_ssdir_predictions_bins8.csv')
# ==========================================
# 请在这里填入你真实跑出来的测试集预测数据！
# 这里我用生成的数据模拟 Base 模型的“趋中回归”和 SSDIR 的“贴合对角线”现象
# ==========================================

y_true = base_df['True_Value'].values

# Base 模型：在长尾高值区（>55）严重低估，强行向均值（40左右）靠拢
y_pred_base = base_df['Base_Prediction'].values

# SSDIR (New) 模型：有效克服趋中回归，高值区依然贴合真实值
y_pred_ssdir = ssdir_df['SSDIR_Prediction'].values

# ================= 开始绘图 =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 绘制对角线 (理想预测线)
min_val, max_val = 0, 90
for ax in [ax1, ax2]:
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal y=x (Perfect Prediction)')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('True Values (Target)', fontsize=14, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

# 子图 1: Base 模型散点
ax1.scatter(y_true, y_pred_base, alpha=0.7, color='#ff6666', edgecolor='k', s=50, label='Predicted Points')
ax1.set_ylabel('Predicted Values', fontsize=14, weight='bold')
ax1.set_title('Base Model (Shows Regression to Mean)', fontsize=16, weight='bold', pad=15)
ax1.legend(loc='upper left')

# 在 Base 模型高值区画一个圈，标示出欠拟合严重的区域
circle1 = plt.Circle((70, 45), 15, color='blue', fill=False, linestyle='-.', lw=2)
ax1.add_patch(circle1)
ax1.text(70, 25, 'Severe Under-prediction\nin Few-shot Region', ha='center', color='blue', weight='bold')

# 子图 2: SSDIR 模型散点
ax2.scatter(y_true, y_pred_ssdir, alpha=0.7, color='#008000', edgecolor='k', s=50, label='Predicted Points')
ax2.set_title('SSDIR Framework (Corrected Long-tail)', fontsize=16, weight='bold', pad=15)
ax2.legend(loc='upper left')

# 在 SSDIR 高值区画一个圈，标示出被纠正的区域
circle2 = plt.Circle((70, 70), 15, color='blue', fill=False, linestyle='-.', lw=2)
ax2.add_patch(circle2)
ax2.text(70, 50, 'Prediction Corrected\nin Few-shot Region', ha='center', color='blue', weight='bold')

plt.tight_layout()
plt.show()