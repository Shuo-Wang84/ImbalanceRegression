import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import pandas as pd



# 设置中文字体 - 与test.py保持一致
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体的字体名称
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
sns.set_style("whitegrid")
sns.set_palette("husl")





def load_california_housing_data():
    """
    加载加利福尼亚房价数据集
    """
    # 加载数据
    california_housing = fetch_california_housing()
    X = california_housing.data
    y = california_housing.target
    feature_names = california_housing.feature_names

    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    return df, feature_names


def plot_target_density(df, save_path=None):
    """
    绘制目标变量（房价）的密度图
    """
    plt.figure(figsize=(12, 6))

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：基本密度图
    sns.kdeplot(data=df, x='target', fill=True, ax=ax1, color='skyblue', alpha=0.7)
    ax1.set_title('加利福尼亚房价密度分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('房价 (万美元)', fontsize=12)
    ax1.set_ylabel('密度', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 添加统计信息
    mean_price = df['target'].mean()
    median_price = df['target'].median()
    ax1.axvline(mean_price, color='red', linestyle='--', alpha=0.8, label=f'均值: {mean_price:.2f}')
    ax1.axvline(median_price, color='orange', linestyle='--', alpha=0.8, label=f'中位数: {median_price:.2f}')
    ax1.legend()

    # 右图：密度图 + 直方图组合
    sns.histplot(data=df, x='target', kde=True, ax=ax2, color='lightcoral', alpha=0.6)
    ax2.set_title('房价分布：直方图 + 密度曲线', fontsize=14, fontweight='bold')
    ax2.set_xlabel('房价 (万美元)', fontsize=12)
    ax2.set_ylabel('频数 / 密度', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_features_density(df, feature_names, save_path=None):
    """
    绘制所有特征的密度图
    """
    # 计算子图布局
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    # 特征名称中文映射
    feature_chinese = {
        'MedInc': '收入中位数',
        'HouseAge': '房屋年龄',
        'AveRooms': '平均房间数',
        'AveBedrms': '平均卧室数',
        'Population': '人口数量',
        'AveOccup': '平均入住率',
        'Latitude': '纬度',
        'Longitude': '经度'
    }

    for i, feature in enumerate(feature_names):
        if i < len(axes):
            sns.kdeplot(data=df, x=feature, fill=True, ax=axes[i], alpha=0.7)
            chinese_name = feature_chinese.get(feature, feature)
            axes[i].set_title(f'{chinese_name} 密度分布', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(chinese_name, fontsize=10)
            axes[i].set_ylabel('密度', fontsize=10)
            axes[i].grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(len(feature_names), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('加利福尼亚房价数据集 - 所有特征密度分布', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_target_vs_features_density(df, feature_names, save_path=None):
    """
    绘制目标变量与关键特征的联合密度图
    """
    # 选择几个关键特征
    key_features = ['MedInc', 'HouseAge', 'AveRooms', 'Population']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    feature_chinese = {
        'MedInc': '收入中位数',
        'HouseAge': '房屋年龄',
        'AveRooms': '平均房间数',
        'Population': '人口数量'
    }

    for i, feature in enumerate(key_features):
        # 创建联合分布图
        sns.kdeplot(data=df, x=feature, y='target', ax=axes[i], fill=True, cmap='viridis')
        chinese_name = feature_chinese.get(feature, feature)
        axes[i].set_title(f'{chinese_name} vs 房价 联合密度分布', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(chinese_name, fontsize=10)
        axes[i].set_ylabel('房价 (万美元)', fontsize=10)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('关键特征与房价的联合密度分布', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_data_distribution(df):
    """
    分析数据分布特征
    """
    print("=== 加利福尼亚房价数据集分布分析 ===")
    print(f"数据集大小: {df.shape[0]} 样本, {df.shape[1] - 1} 特征")
    print("\n目标变量（房价）统计信息:")
    print(f"均值: {df['target'].mean():.2f} 万美元")
    print(f"中位数: {df['target'].median():.2f} 万美元")
    print(f"标准差: {df['target'].std():.2f} 万美元")
    print(f"最小值: {df['target'].min():.2f} 万美元")
    print(f"最大值: {df['target'].max():.2f} 万美元")
    print(f"偏度: {df['target'].skew():.2f}")
    print(f"峰度: {df['target'].kurtosis():.2f}")

    # 分析数据不平衡性
    q1 = df['target'].quantile(0.25)
    q3 = df['target'].quantile(0.75)
    iqr = q3 - q1

    print(f"\n四分位数分析:")
    print(f"Q1 (25%): {q1:.2f} 万美元")
    print(f"Q3 (75%): {q3:.2f} 万美元")
    print(f"IQR: {iqr:.2f} 万美元")

    # 检测异常值
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df['target'] < lower_bound) | (df['target'] > upper_bound)]
    print(f"异常值数量: {len(outliers)} ({len(outliers) / len(df) * 100:.1f}%)")


def main():
    """
    主函数
    """
    # 加载数据
    print("正在加载加利福尼亚房价数据集...")
    df, feature_names = load_california_housing_data()

    # 分析数据分布
    analyze_data_distribution(df)

    # 设置保存路径
    output_dir = r"F:\SGIR\SGIR-main\plt画图\MLP\加利福尼亚房价预测"

    print("\n正在绘制密度图...")

    # 1. 绘制目标变量密度图
    plot_target_density(df, f"{output_dir}\\california_target_density.png")

    # 2. 绘制所有特征密度图
    plot_features_density(df, feature_names, f"{output_dir}\\california_features_density.png")

    # 3. 绘制联合密度图
    plot_target_vs_features_density(df, feature_names, f"{output_dir}\\california_joint_density.png")

    print("\n所有密度图绘制完成！")
    print("生成的图片:")
    print("1. california_target_density.png - 房价密度分布")
    print("2. california_features_density.png - 所有特征密度分布")
    print("3. california_joint_density.png - 关键特征与房价联合密度分布")


if __name__ == "__main__":
    main()