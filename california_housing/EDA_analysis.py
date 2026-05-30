import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import fetch_california_housing
import os

# 设置后端和字体，防止 PyCharm 报错和中文乱码
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def analyze_dataset(df, target_col, dataset_name):
    print(f"==========================================")
    print(f"开始分析数据集: {dataset_name}")
    print(f"==========================================")

    # 1. 计算目标变量的偏度 (Skewness) 和 峰度 (Kurtosis)
    target_data = df[target_col]
    s = skew(target_data)
    k = kurtosis(target_data)
    print(f"[{dataset_name}] 目标变量 '{target_col}' 的统计特征:")
    print(f"样本总量: {len(df)}")
    print(f"偏度 (Skewness): {s:.4f}")
    print(f"峰度 (Kurtosis): {k:.4f}")
    print(f"最小值: {target_data.min():.4f}, 最大值: {target_data.max():.4f}")
    print(f"------------------------------------------\n")

    # 2. 绘制并保存特征相关性热力图
    plt.figure(figsize=(12, 10))  # 设置稍大的画布保证特征名字不拥挤
    corr_matrix = df.corr(method='pearson')  # 计算皮尔逊相关系数

    # 画热力图
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                cbar=True, square=True, linewidths=.5,
                annot_kws={'size': 10})  # 调整数字大小

    plt.title(f'{dataset_name} 特征相关性热力图', fontsize=18, pad=20, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()

    # 动态生成保存路径
    save_path = f'{dataset_name}_correlation_heatmap.png'
    plt.savefig(save_path, dpi=300)  # 保存为 300dpi 的高清图片用于论文
    print(f"✅ {dataset_name} 的热力图已成功保存为: {save_path}\n")
    plt.show()


# ================= 主程序入口 =================

if __name__ == "__main__":
    # 【示例 1】：分析加州房价数据集 (直接使用 sklearn 自带的数据拉取)
    print("正在加载加州房价数据集...")
    california = fetch_california_housing()
    df_california = pd.DataFrame(california.data, columns=california.feature_names)
    df_california['MedHouseVal'] = california.target  # 加上目标标签列

    # 调用分析函数
    analyze_dataset(df_california, target_col='MedHouseVal', dataset_name='California_Housing')

    # ========================================================
    # 导师提示：针对你剩下的三个数据集（红酒、空气质量、混凝土），
    # 你只需要用 pandas 读取它们的 csv 文件，然后调用相同的函数即可。
    # 比如：
    # df_wine = pd.read_csv('你的红酒数据集路径.csv')
    # analyze_dataset(df_wine, target_col='红酒的标签列名', dataset_name='Wine_Quality')
    # ========================================================