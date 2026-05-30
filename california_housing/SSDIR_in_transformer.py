import math
import os
import warnings

import shap
# 新增PyTorch相关依赖
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import plotly.graph_objects as go
import columns
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体的字体名称
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置全局参数
NUM_RUNS = 10  # 运行次数
all_bins_values = range(16, 18, 2)
min_mse_results = []
SEED_BASE = 42
iterations = 5  # 自训练迭代次数
# 文件保存路径
output_dir = r"F:\SGIR\SGIR-main\plt画图\Transformer\加利福尼亚房价"


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 输入投影层
        self.input_projection = nn.Linear(1, d_model)  # 修改：每个特征单独投影

        # 位置编码（对于表格数据，使用可学习的位置编码）
        self.pos_encoding = nn.Parameter(torch.randn(input_dim, d_model))

        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size = x.size(0)

        # 将每个特征视为一个token
        # x: (batch_size, input_dim) -> (batch_size, input_dim, 1)
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)

        # 投影到d_model维度
        # (batch_size, input_dim, 1) -> (batch_size, input_dim, d_model)
        x = self.input_projection(x)  # 修改：直接投影，不需要转置

        # 添加位置编码
        # pos_encoding: (input_dim, d_model) -> (1, input_dim, d_model)
        # x: (batch_size, input_dim, d_model)
        x = x + self.pos_encoding.unsqueeze(0)  # 修改：直接广播

        # Layer normalization
        x = self.layer_norm(x)

        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, input_dim, d_model)

        # 全局平均池化
        x = torch.mean(x, dim=1)  # (batch_size, d_model)

        # 输出投影
        output = self.output_projection(x)  # (batch_size, 1)

        return output.squeeze(-1)  # (batch_size,)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out


# 可视化标签分布
def plot_distribution(data, title, filename=None, color='blue', now_bins=bin):
    """
        使用 Plotly 绘制可交互的分布图，鼠标悬停时显示样本数量和标签范围。

        参数:
        - data: pd.Series 或 np.ndarray
            要绘制分布图的数据。
        - title: str
            图的标题。
        - filename: str
            如果提供，将图保存为 HTML 文件。
        - color: str
            直方图的颜色。
        - now_bins: int
            分布的分箱数量。
        """
    # 计算直方图数据
    hist, bin_edges = np.histogram(data, bins=now_bins)

    # 创建 Plotly 的条形图
    fig = go.Figure()

    # 添加直方图
    fig.add_trace(go.Bar(
        x=[f'{bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}' for i in range(len(bin_edges) - 1)],  # 标签范围
        y=hist,  # 每个区间的样本数量
        marker_color=color,
        text=hist,  # 样本数量（用于鼠标悬停显示）
        hoverinfo='text+name',
        name="频率",
        texttemplate='%{text}',  # 显示样本数量
        textposition='outside'
    ))

    # 设置图表样式
    fig.update_layout(
        title=title,
        xaxis_title="标签范围",
        yaxis_title="样本数量（频率）",
        bargap=0.1,  # 柱形之间的间距
        template="plotly_white",
        xaxis=dict(tickangle=45),  # X轴标签旋转角度
    )

    # 保存为 HTML 文件
    if filename:
        fig.write_html(filename)
    else:
        fig.show()


# 确保目录存在
os.makedirs(output_dir, exist_ok=True)

# ---------------------------- 修改训练逻辑 ----------------------------
# 修改后的 train_model 函数
def train_model(
        model,
        X_train,
        y_train,
        optimizer,
        criterion,  # 接收损失函数
        epochs=100,
        batch_size=64,
        use_fixed_seed=False
):
    X_tensor = torch.FloatTensor(X_train.values)
    y_tensor = torch.FloatTensor(y_train.values)
    dataset = TensorDataset(X_tensor, y_tensor)

    if use_fixed_seed:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED_BASE)
        )
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # # 记录训练前的参数
    # before_weights = model.mlp[0].weight.data.clone()

    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)  # 正确使用传入的 criterion
            loss.backward()
            optimizer.step()

    # # 记录训练后的参数
    # after_weights = model.mlp[0].weight.data.clone()
    #
    # # 计算参数变化量
    # delta = (after_weights - before_weights).norm().item()
    # print(f"模型参数变化量（L2范数）: {delta:.6f}")


# ---------------------------- 修改预测逻辑 ----------------------------
def predict(model, X):
    model.eval()
    with torch.no_grad():
        # 将输入数据转换为张量
        X_test_tensor = torch.FloatTensor(X.values)
        # 前向传播获取预测结果
        pred_tensor = model(X_test_tensor)
        # 转换为NumPy数组（与y_test格式一致）
        pred = pred_tensor.numpy()
    return pred


def plot_correlation_matrix(data, title="Feature Correlation Matrix", output_path=None):
    """
    计算特征之间的相关性矩阵并绘制热图。

    参数:
    - data: pd.DataFrame
        包含数值型特征的数据集。
    - title: str
        热图的标题。
    - output_path: str
        如果提供，将热图保存到指定路径，否则直接显示图像。
    """
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()  # 计算相关性矩阵
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    # 如果提供了保存路径，则保存图像；否则直接显示
    if output_path:
        plt.savefig(output_path)
        print(f"相关性矩阵热图保存到: {output_path}")
    else:
        plt.show()
    plt.close()


# def plot_feature_importance(model, feature_names, title="Feature Importance", output_path=None):
#     """
#     绘制特征重要性图。
#
#     参数:
#     - model: 已训练的模型（如 RandomForestRegressor 或其他支持 feature_importances_ 的模型）。
#     - feature_names: list
#         特征名称列表。
#     - title: str
#         图的标题。
#     - output_path: str
#         如果提供，将图保存到指定路径，否则直接显示。
#     """
#     # 从模型中提取特征重要性
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1]  # 按重要性从高到低排序
#     sorted_feature_names = [feature_names[i] for i in indices]
#
#     # 绘制柱状图
#     plt.figure(figsize=(12, 8))
#     plt.bar(range(len(importances)), importances[indices], align='center', color='skyblue')
#     plt.xticks(range(len(importances)), sorted_feature_names, rotation=45, ha='right')
#     plt.xlabel('Features')
#     plt.ylabel('Importance Score')
#     plt.title(title)
#     plt.tight_layout()
#
#     # 保存或显示图像
#     if output_path:
#         plt.savefig(output_path)
#         print(f"特征重要性图已保存到: {output_path}")
#     else:
#         plt.show()
#     plt.close()

def divide_regions(y, bins, sample_counts):
    """
    根据区间样本数将每个区间划分为 Many、Med、Few，并将每个样本分配到对应区域。
    """
    # 1. 计算分位点
    thresholds = sample_counts.quantile([0.30, 0.70])
    few_threshold = thresholds[0.30]
    many_threshold = thresholds[0.70]

    region_map = {}
    for interval, count in sample_counts.items():
        if count > many_threshold:
            region_map[interval] = "Many"
        elif count <= few_threshold:
            region_map[interval] = "Few"
        else:
            region_map[interval] = "Med"
    return region_map


def assign_region_labels(y, bins, region_map):
    """
    根据region_map将每个样本分配到对应的区域标签。
    """
    y_binned = pd.cut(y, bins, include_lowest=True)
    region_labels = y_binned.map(region_map)
    return region_labels


def calculate_metrics_by_region(y_test, y_pred, region_labels):
    """
    计算每个区域的 MSE 和 MAE。

    参数:
    y_test : pd.Series
        测试集标签。
    y_pred : np.ndarray
        测试集预测值。
    region_labels : pd.Series
        每个样本对应的区域标签 ('Many', 'Med', 'Few')。

    返回:
    metrics : dict
        每个区域的 MSE 和 MAE。
    """
    metrics = {}
    for region in ['Many', 'Med', 'Few']:
        mask = region_labels == region
        if mask.sum() > 0:
            mse = mean_squared_error(np.array(y_test)[mask], np.array(y_pred)[mask])
            mae = mean_absolute_error(np.array(y_test)[mask], np.array(y_pred)[mask])
            metrics[region] = {'MSE': mse, 'MAE': mae}
        else:
            metrics[region] = {'MSE': None, 'MAE': None}
    return metrics


# 添加基尼系数计算函数
def gini_coefficient(y):
    """计算基尼系数"""
    y = np.array(y, dtype=float)

    # 处理特殊情况
    if len(y) == 0:
        return 0
    if np.all(y == y[0]):
        return 0  # 所有值相同

    # 确保非负值
    if np.min(y) < 0:
        y = y - np.min(y)

    # 排序
    y_sorted = np.sort(y)
    n = len(y)

    # 使用累积和计算
    cumsum = np.cumsum(y_sorted)

    # 基尼系数公式
    gini = (2 * np.sum((np.arange(1, n + 1) * y_sorted))) / (n * cumsum[-1]) - (n + 1) / n

    return max(0, gini)  # 确保非负


def plot_lorenz_curve_comparison(y_original, y_confident, title="洛伦兹曲线对比", output_path=None):
    """绘制两个数据集的洛伦兹曲线对比"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # 计算基尼系数
    gini_original = gini_coefficient(y_original)
    gini_confident = gini_coefficient(y_confident)

    # 绘制原始数据的洛伦兹曲线
    def plot_single_lorenz(y, ax, title, color):
        y = np.array(y)
        y_sorted = np.sort(y)
        n = len(y)

        # 计算累积比例
        cumulative_pop = np.arange(1, n + 1) / n
        cumulative_wealth = np.cumsum(y_sorted) / np.sum(y_sorted)

        # 添加原点
        cumulative_pop = np.insert(cumulative_pop, 0, 0)
        cumulative_wealth = np.insert(cumulative_wealth, 0, 0)

        # 绘图
        ax.plot(cumulative_pop, cumulative_wealth, color=color, linewidth=2, label='洛伦兹曲线')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='完全平等线')

        # 填充基尼系数对应的面积
        ax.fill_between(cumulative_pop, cumulative_wealth, cumulative_pop,
                        alpha=0.3, color=color)

        ax.set_xlabel('累积样本比例')
        ax.set_ylabel('累积目标值比例')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # 绘制原始数据
    plot_single_lorenz(y_original, ax1, f'原始数据\n基尼系数: {gini_original:.4f}', 'blue')

    # 绘制高置信度数据
    plot_single_lorenz(y_confident, ax2, f'高置信度数据\n基尼系数: {gini_confident:.4f}', 'green')

    # 绘制对比图
    y_orig_sorted = np.sort(y_original)
    y_conf_sorted = np.sort(y_confident)

    n_orig = len(y_orig_sorted)
    n_conf = len(y_conf_sorted)

    # 原始数据洛伦兹曲线
    cumulative_pop_orig = np.arange(1, n_orig + 1) / n_orig
    cumulative_wealth_orig = np.cumsum(y_orig_sorted) / np.sum(y_orig_sorted)
    cumulative_pop_orig = np.insert(cumulative_pop_orig, 0, 0)
    cumulative_wealth_orig = np.insert(cumulative_wealth_orig, 0, 0)

    # 高置信度数据洛伦兹曲线
    cumulative_pop_conf = np.arange(1, n_conf + 1) / n_conf
    cumulative_wealth_conf = np.cumsum(y_conf_sorted) / np.sum(y_conf_sorted)
    cumulative_pop_conf = np.insert(cumulative_pop_conf, 0, 0)
    cumulative_wealth_conf = np.insert(cumulative_wealth_conf, 0, 0)

    ax3.plot(cumulative_pop_orig, cumulative_wealth_orig, 'blue', linewidth=2,
             label=f'原始数据 (Gini={gini_original:.4f})')
    ax3.plot(cumulative_pop_conf, cumulative_wealth_conf, 'green', linewidth=2,
             label=f'高置信度数据 (Gini={gini_confident:.4f})')
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=1, label='完全平等线')

    ax3.set_xlabel('累积样本比例')
    ax3.set_ylabel('累积目标值比例')
    ax3.set_title('洛伦兹曲线对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示
    else:
        plt.show()

    return gini_original, gini_confident


def analyze_imbalance_improvement(y_original, y_confident, bins_value, iteration=None, output_dir=None):
    """分析不平衡改善情况"""
    gini_original = gini_coefficient(y_original)
    gini_confident = gini_coefficient(y_confident)

    # 计算改善程度
    improvement = (gini_original - gini_confident) / gini_original * 100 if gini_original > 0 else 0

    print(f"\n=== 不平衡分析结果 ===")
    if iteration is not None:
        print(f"迭代 {iteration} 轮:")
    print(f"原始数据基尼系数: {gini_original:.4f}")
    print(f"高置信度数据基尼系数: {gini_confident:.4f}")
    print(f"不平衡改善程度: {improvement:.2f}%")

    # 绘制对比图
    if output_dir:
        if iteration is not None:
            output_path = os.path.join(output_dir, f'bins_{bins_value}_iteration_{iteration}_gini_comparison.png')
            title = f'Bins={bins_value}, 迭代{iteration} - 基尼系数对比'
        else:
            output_path = os.path.join(output_dir, f'bins_{bins_value}_gini_comparison.png')
            title = f'Bins={bins_value} - 基尼系数对比'

        plot_lorenz_curve_comparison(y_original, y_confident, title, output_path)

    return {
        'gini_original': gini_original,
        'gini_confident': gini_confident,
        'improvement_percent': improvement
    }

def add_perturbation(X, feature_importance, scale=0.1):
    perturbation = np.random.normal(0, scale, X.shape)
    perturbation *= feature_importance
    return X + perturbation


def calculate_prediction_variance(predictions):
    """
    计算每个样本的预测方差。
    predictions: shape = (n_samples, n_repeat)
    返回: shape = (n_samples,0)
    """
    return np.var(predictions, axis=0)


def plot_feature_importance_heatmap(feature_importance, feature_names=None, title="Feature Importance Heatmap",
                                    save_path=None):
    """
    绘制特征重要性热力图

    参数:
    - feature_importance: np.ndarray, 特征重要性数组
    - feature_names: list, 特征名称列表
    - title: str, 图表标题
    - save_path: str, 保存路径
    """
    plt.figure(figsize=(12, 8))

    # 如果没有提供特征名称，使用默认名称
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]

    # 创建热力图数据（将一维数组转换为二维）
    importance_matrix = feature_importance.reshape(1, -1)

    # 绘制热力图
    sns.heatmap(importance_matrix,
                annot=True,  # 显示数值
                fmt='.4f',  # 数值格式
                cmap='YlOrRd',  # 颜色映射
                xticklabels=feature_names,
                yticklabels=['Importance'],
                cbar_kws={'label': 'SHAP Importance Value'})

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存到: {save_path}")
        plt.close()  # 关闭图形，不显示

    else:
        plt.show()


def plot_feature_importance_bar(feature_importance, feature_names=None, title="Feature Importance Bar Chart",
                                save_path=None):
    """
    绘制特征重要性柱状图（作为热力图的补充）

    参数:
    - feature_importance: np.ndarray, 特征重要性数组
    - feature_names: list, 特征名称列表
    - title: str, 图表标题
    - save_path: str, 保存路径
    """
    plt.figure(figsize=(12, 6))

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]

    # 按重要性排序
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_importance = feature_importance[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices.tolist()]  # 转换为 Python 列表

    # 创建颜色映射
    colors = plt.cm.YlOrRd(np.linspace(0.3, 1, len(sorted_importance)))

    bars = plt.bar(range(len(sorted_importance)), sorted_importance, color=colors)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('SHAP Importance Value', fontsize=12)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')

    # 在柱状图上添加数值标签
    for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"柱状图已保存到: {save_path}")
        plt.close()  # 关闭图形，不显示

    else:
        plt.show()


def new_select_high_confidence_samples(model, X_unlabeled, n_perturb=5, top_k=1, threshold=0.05,
                                       plot_importance=False, feature_names=None, save_plots=False):
    model.eval()
    # 计算SHAP特征重要性
    explainer = shap.DeepExplainer(ModelWrapper(model), torch.tensor(X_unlabeled.values, dtype=torch.float32))
    shap_values = explainer.shap_values(torch.tensor(X_unlabeled.values, dtype=torch.float32))
    # shap_values 形状：[样本数, 特征数]
    feature_importance = np.abs(shap_values).mean(axis=0)
    # 找到最关键特征的索引

    # 可视化特征重要性
    if plot_importance:
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]

        # 绘制热力图
        heatmap_path = os.path.join(output_dir, "feature_importance_heatmap.png") if save_plots else None
        plot_feature_importance_heatmap(feature_importance, feature_names,
                                            "SHAP Feature Importance Heatmap", heatmap_path)

        # 绘制柱状图
        bar_path = os.path.join(output_dir, "feature_importance_bar.png") if save_plots else None
        plot_feature_importance_bar(feature_importance, feature_names,
                                        "SHAP Feature Importance Bar Chart", bar_path)


    top_indices = np.argsort(feature_importance)[-top_k:]
    # 生成多组扰动样本
    perturbed_samples_list = []
    for _ in range(n_perturb):
        noise = np.random.normal(loc=0.0, scale=0.1, size=X_unlabeled.shape)
        # 只对非最关键特征添加扰动
        noise[:, top_indices] = 0
        perturbed = X_unlabeled + noise
        perturbed_samples_list.append(perturbed)
    # 堆叠扰动样本，形状：[n_perturb, 样本数, 特征数]
    perturbed_samples = np.stack(perturbed_samples_list, axis=0)
    # 计算每组扰动下的预测
    preds = []
    for i in range(n_perturb):
        with torch.no_grad():
            pred = model(torch.tensor(perturbed_samples[i], dtype=torch.float32)).cpu().numpy().squeeze()
        preds.append(pred)
    preds = np.stack(preds, axis=0)  # [n_perturb, 样本数]
    # 计算每个样本的预测方差
    pred_var = np.var(preds, axis=0)
    # 根据方差筛选高置信度样本
    high_conf_indices = np.where(pred_var < threshold)[0]

    return high_conf_indices, feature_importance  # 同时返回特征重要性


def calculate_bins_sample(y, num_bins=50):
    """
        统计每个区间的样本数量。
        参数:
        y : pd.Series
            标签数据。
        num_bins : int
            区间数。
        返回:
        bin_counts : pd.Series
            每个区间的样本数量。
        bins : np.ndarray
            区间边界。
        """
    bins = np.linspace(y.min(), y.max(), num_bins + 1)
    y_binned = pd.cut(y, bins, include_lowest=True)
    bin_counts = y_binned.value_counts().sort_index()

    return bin_counts, bins


def calculate_sampling_rates(y, num_bins=50):
    """
    根据标签数据计算每个区间需要补充的样本数量。

    参数:
    y : pd.Series
        标签数据。
    num_bins : int
        将标签划分为多少个等宽区间。

    返回:
    sampling_counts : pd.Series
        每个区间需要补充的样本数量。
    bins : np.ndarray
        标签区间。
    max_samples_in_bin : int
        最大区间样本数。
    """
    # 计算标签的区间
    bins = np.linspace(y.min(), y.max(), num_bins + 1)
    y_binned = pd.cut(y, bins, include_lowest=True)

    # 计算每个区间的样本数量
    bin_counts = y_binned.value_counts().sort_index()

    # 找到最大区间样本数
    max_samples_in_bin = bin_counts.max()

    # 计算每个区间需要补充的样本数
    sampling_counts = max_samples_in_bin - bin_counts

    # # 输出每个区间的样本数量和需要补充的数量
    # for interval, count, needed in zip(bin_counts.index, bin_counts, sampling_counts):
    #     print(f"区间: {interval}, 当前样本数量: {count}, 需要补充的样本数量: {needed}")

    return sampling_counts, bins, max_samples_in_bin


def perform_sampling(X, y, sampling_counts, bins):
    """
    根据每个区间需要补充的样本数量对数据进行采样。

    参数：
    X : pd.DataFrame
        特征数据。
    y : pd.Series
        标签数据。
    sampling_counts : pd.Series
        每个区间需要补充的样本数量。
    bins : np.ndarray
        标签区间。

    返回：
    X_sampled, y_sampled : pd.DataFrame, pd.Series
        采样后的特征和标签。
    """
    # np.random.seed(SEED + current_iter)
    y_binned = pd.cut(y, bins, include_lowest=True)

    X_sampled = []
    y_sampled = []

    # 遍历每个区间，按照需要补充的数量进行采样
    for interval, needed_samples in sampling_counts.items():
        if needed_samples <= 0:
            continue  # 如果不需要补充样本，跳过

        # 获取属于该区间的样本索引
        mask = y_binned == interval
        indices = np.where(mask)[0]

        if len(indices) == 0:
            print(f"区间 {interval} 没有样本，无法采样")
            continue

        # 如果需要补充的样本数大于当前区间样本数，则只采当前区间的全部样本
        if needed_samples > len(indices):
            needed_samples = len(indices)
            # print(f"区间 {interval} 的需要补充样本数大于当前区间样本数，仅采样 {needed_samples} 条样本")

        # 进行采样
        resampled_indices = np.random.choice(
            indices,
            size=needed_samples,
            replace=False,
        )
        X_sampled.append(X.iloc[resampled_indices])
        y_sampled.append(y.iloc[resampled_indices])

    # 将采样结果合并为一个 DataFrame 和 Series
    if len(X_sampled) > 0:
        X_sampled = pd.concat(X_sampled).reset_index(drop=True)
        y_sampled = pd.concat(y_sampled).reset_index(drop=True)
    else:
        X_sampled = pd.DataFrame(columns=X.columns)
        y_sampled = pd.Series(dtype=y.dtype)

    return X_sampled, y_sampled


def label_anchored_mixup(X, y, sampling_counts, bins, alpha=0.2):
    """
    使用标签锚点混合数据增强来改善标签不平衡问题。
    参数:
    X : pd.DataFrame
        特征数据。
    y : pd.Series
        标签数据。
    sampling_prob : pd.Series
        每个区间的采样概率。
    bins : np.ndarray
        标签区间。
    num_samples : int
        希望生成的增强样本数量。
    alpha : float
       混合策略中的Beta分布参数，控制混合比例。
    返回:
    X_augmented, y_augmented : pd.DataFrame, pd.Series
        增强后的特征和标签。
    """
    # np.random.seed(SEED + current_iter)
    y_binned = pd.cut(y, bins, include_lowest=True)
    X_augmented = []
    y_augmented = []

    bin_counts = y_binned.value_counts().sort_index()

    # print(f"最大数量是：{max_samples_per_bin}")

    # 确保similar_indices不超出当前X的索引范围
    valid_indices = X.index  # 当前X的所有有效索引
    y_binned = y_binned[y_binned.index.isin(valid_indices)]  # 仅保留有效索引

    for interval, needed_samples in sampling_counts.items():
        if needed_samples <= 0:
            continue  # 如果不需要补充样本，跳过

        center = (interval.left + interval.right) / 2

        # 进一步筛选，确保similar_indices在X中存在
        similar_indices = y_binned[y_binned == interval].index

        if len(similar_indices) == 0:
            continue

        # 确保X_virtual是包含所有特征的向量（形状为(8,)）
        X_virtual = X.loc[similar_indices].mean(axis=0).values.flatten()  # 修正为一维数组
        y_virtual = center

        # print(f"当前区间:{interval}，当前的数量是：{len(similar_indices)}，计划使用标签锚定的采样数量是：{num_samples_bin}，"
        #       f"实际采样数量：{real_num_samples_bin}")

        for _ in range(needed_samples):
            # 从similar_indices中随机选择，且确保idx_real在X中存在
            valid_real_indices = list(similar_indices.intersection(X.index))
            if not valid_real_indices:
                continue  # 跳过无效索引
            idx_real = np.random.choice(valid_real_indices)
            x_real = X.loc[idx_real].values.flatten()
            y_real = y.loc[idx_real]

            # 检查维度一致性
            if x_real.shape[0] != X.shape[1]:
                raise ValueError(f"特征维度不一致: 预期{X.shape[1]}，实际{x_real.shape[0]}")

            # np.random.seed(SEED)  # 确保每次增强结果一致

            lam = np.random.beta(alpha, alpha)

            x_mix = lam * X_virtual + (1 - lam) * x_real  # 向量运算
            # 修改混合计算部分
            y_real_value = y.at[idx_real]  # 确保获取标量
            y_mixed = lam * y_virtual + (1 - lam) * y_real_value

            # 检查计算结果的合法性
            if not np.isfinite(y_mixed).all():  # 确保处理标量
                continue  # 跳过非法值

            X_augmented.append(x_mix.reshape(1, -1))  # 形状变为(1,8)
            y_augmented.append(y_mixed)

    # 合并所有样本并确保形状正确
    if X_augmented:
        X_augmented = np.vstack(X_augmented)  # 形状为(N, 8)
    else:
        X_augmented = np.empty((0, X.shape[1]))  # 若无增强数据，返回空数组

    X_augmented = pd.DataFrame(np.array(X_augmented), columns=X.columns)

    # 修复：确保 y_augmented 是数值类型且无 NaN
    y_augmented = pd.Series(y_augmented)

    y_augmented = pd.to_numeric(y_augmented, errors='coerce')  # 强制转换为数值
    y_augmented = y_augmented.dropna()  # 清除 NaN

    # 同步裁剪 X_augmented，确保长度一致
    X_augmented = X_augmented.iloc[:len(y_augmented)]

    return X_augmented, y_augmented


def plot_label_distribution_with_regions(y, bins, region_map, now_bins, title="Label Distribution with Regions"):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(y, bins=bins, kde=False, color="blue", label="Label distribution", ax=ax, alpha=0.7)
    region_label_added = set()
    for i, interval in enumerate(region_map.keys()):
        region = region_map[interval]
        label = None
        if region not in region_label_added:
            if region == "Many":
                label = "Many-shot region"
            elif region == "Med":
                label = "Medium-shot region"
            elif region == "Few":
                label = "Few-shot region"
            region_label_added.add(region)
        if region == "Many":
            ax.axvspan(bins[i], bins[i + 1], color="lightblue", alpha=0.3, label=label)
        elif region == "Med":
            ax.axvspan(bins[i], bins[i + 1], color="lightyellow", alpha=0.6, label=label)
        elif region == "Few":
            ax.axvspan(bins[i], bins[i + 1], color="lightpink", alpha=0.3, label=label)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Target value", fontsize=12)
    ax.set_ylabel("# of samples", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"当前bins值为：{now_bins}的数据分区.png")


# 定义训练和评估函数
def run_experiment(run_seed):
    # 设置随机种子
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    california = fetch_california_housing()
    data = pd.DataFrame(california.data, columns=california.feature_names)
    data['MedHouseVal'] = california.target

    # 分割数据集
    X = data.drop('MedHouseVal', axis=1)
    y = data['MedHouseVal']
    X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 处理缺失值
    X_full_train.dropna(inplace=True)
    X_test.dropna(inplace=True)

    # 初始化标准化器
    scaler = StandardScaler()

    # X_full_train = train_data[features]
    # y_full_train = train_data[target]
    #
    # X_test = test_data[features]
    # y_test = test_data[target]

    # 模拟半监督场景：训练集中只有20%有标签
    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
        X_full_train, y_full_train,
        train_size=0.3,  # 20%有标签
        random_state=42
    )

    # 标准化有标签数据
    X_labeled = pd.DataFrame(scaler.fit_transform(X_labeled), columns=X_labeled.columns)

    # 标准化无标签数据和测试数据
    X_unlabeled = pd.DataFrame(scaler.transform(X_unlabeled), columns=X_unlabeled.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 处理缺失值（可以根据需要选择不同的方法）
    X_labeled.dropna(inplace=True)
    X_unlabeled.dropna(inplace=True)
    X_test.dropna(inplace=True)

    feature_names = [
        'MedInc',  # 收入中位数
        'HouseAge',  # 房屋年龄
        'AveRooms',  # 平均房间数
        'AveBedrms',  # 平均卧室数
        'Population',  # 人口
        'AveOccup',  # 平均入住率
        'Latitude',  # 纬度
        'Longitude'  # 经度
    ]

    # ----------------- 模型训练与评估 -----------------
    min_mse_results = []
    min_mae_results = []
    for bins in all_bins_values:
        base_model = TransformerRegressor(
            input_dim=X_labeled.shape[1],
            d_model=64,
            nhead=8,
            num_layers=2,  # 减少层数以适应小数据集
            dim_feedforward=128,
            dropout=0.1
        )
        base_model_optimizer = optim.Adam(base_model.parameters(), lr=1e-3, weight_decay=1e-4)
        base_model_criterion = nn.MSELoss()

        pseudo_model = TransformerRegressor(
            input_dim=X_labeled.shape[1],
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1
        )
        pseude_model_optimizer = optim.Adam(pseudo_model.parameters(), lr=1e-3, weight_decay=1e-4)
        pseudo_model_criterion = nn.MSELoss()

        augment_model = TransformerRegressor(
            input_dim=X_labeled.shape[1],
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1
        )
        augment_model_optimizer = optim.Adam(augment_model.parameters(), lr=1e-3, weight_decay=1e-4)
        augment_model_criterion = nn.MSELoss()

        model = TransformerRegressor(
            input_dim=X_labeled.shape[1],
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1
        )
        final_optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        final_criterion = nn.MSELoss()

        # model.fit(X_labeled, y_labeled)
        train_model(model, X_labeled, y_labeled, optimizer=final_optimizer, criterion=final_criterion, epochs=200,
                    use_fixed_seed=True)

        # 计算base模型均方误差 (MSE)
        # base_model.fit(X_labeled, y_labeled)
        train_model(base_model, X_labeled, y_labeled, optimizer=base_model_optimizer, criterion=base_model_criterion,
                    epochs=200, use_fixed_seed=True)
        # y_test_pred = base_model.predict(X_test)
        y_test_pred = predict(base_model, X_test)

        # # 验证代码
        # print("y_test 形状:", y_test.shape)
        # print("y_test_pred 形状:", y_test_pred.shape)
        # print("y_test 数据类型:", type(y_test))
        # print("y_test_pred 数据类型:", type(y_test_pred))

        mse = mean_squared_error(y_test, y_test_pred)

        # 计算模型的平均绝对误差（MAE）
        mae = mean_absolute_error(y_test, y_test_pred)

        # 划分区间
        bins_sample, bins_edges = calculate_bins_sample(y_labeled, bins)
        train_data_region_map = divide_regions(y_labeled, bins, bins_sample)

        plot_label_distribution_with_regions(
            y=y_labeled,
            bins=bins_edges,
            region_map=train_data_region_map,
            now_bins=bins,
            title="Label Distribution with Regions (Many/Med/Few)"
        )

        # 设置bins的参数
        all_bins = bins
        # 初始化mse、mae为正无穷大
        min_mse = float('inf')
        min_mae = float('inf')
        print(f"当前bins设置为：{all_bins}")

        # 计算测试集的分箱样本数
        test_bins_sample, test_bins = calculate_bins_sample(y_test, all_bins)
        test_region_map = divide_regions(y_test, bins, test_bins_sample)
        region_labels = assign_region_labels(y_test, test_bins, test_region_map)

        # 计算每个区域的 MSE 和 MAE
        base_region_metrics = calculate_metrics_by_region(y_test, y_test_pred, region_labels)
        # 输出结果
        for region, metrics in base_region_metrics.items():
            print(
                f"当前bins为{all_bins}，base模型区域{region}: MSE = {metrics['MSE']:.4f}, MAE = {metrics['MAE']:.4f}")

        print(f"当前bins为{all_bins} base模型的ALL均方误差 (MSE): {mse} base模型的ALL (MAE): {mae}")

        # 整个数据集标签分布
        print(f"整个数据集标签数量: {len(y_full_train)}")
        plot_distribution(y_full_train, f'当前bins为{all_bins} 整个数据集标签分布',
                          os.path.join(output_dir, f'当前bins为{all_bins} 整个数据集标签分布.html'), now_bins=all_bins)

        # 初始数据集标签分布
        print(f"有标签训练数据集标签数量: {len(y_labeled)}")
        plot_distribution(y_labeled, f'当前bins为{all_bins} 有标签训练数据集标签数量',
                          os.path.join(output_dir, f'当前bins为{all_bins} 有标签训练数据集标签数量.html'),
                          now_bins=all_bins)

        # # 不平衡数据集标签分布
        # print(f"不平衡数据集标签数量: {len(G_imb_y)}")
        # plot_distribution(G_imb_y, f'当前bins为{all_bins} 不平衡数据集标签分布',
        #                   os.path.join(output_dir, f'当前bins为{all_bins} 不平衡数据集标签分布.html'), now_bins=all_bins)

        # 在初始数据分析部分添加
        print(f"\n=== 初始数据不平衡分析 ===")
        initial_gini = gini_coefficient(y_labeled)
        print(f"初始标签数据基尼系数: {initial_gini:.4f}")

        # 存储每次迭代的基尼系数结果
        gini_results = []

        for i in range(iterations):
            # 使用模型预测未标记数据的标签
            X_unlbl_pred = predict(model, X_unlabeled)

            print(f"所预测的无标签数据集数量: {len(X_unlbl_pred)}")
            plot_distribution(X_unlbl_pred, f'当前bins为{all_bins} 所预测的无标签数据集数量',
                              os.path.join(output_dir, f'当前bins为{all_bins} 所预测的无标签数据集数量.html'),
                              now_bins=all_bins)

            # 调用改进后的置信度筛选函数
            high_confidence_mask = new_select_high_confidence_samples(model, X_unlabeled, n_perturb=5, top_k=1,
                                                                      threshold=0.05)

            # 调用函数并生成热力图
            high_conf_indices, feature_importance = new_select_high_confidence_samples(
                model=model,
                X_unlabeled=X_unlabeled,
                plot_importance=True,  # 启用可视化
                feature_names=feature_names,  # 提供特征名称
                save_plots=True  # 保存图片
            )

            # 打印特征重要性
            print("特征重要性排序:")
            for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
                print(f"{i + 1}. {name}: {importance:.4f}")

            # 确保 high_confidence_mask 是布尔数组
            X_conf = X_unlabeled.iloc[high_conf_indices].reset_index(drop=True)
            y_conf = pd.Series(
                predict(model, X_conf),  # 使用 .values 转换为 ndarray，直接传入 DataFrame（模型已记录特征名称）
                index=X_conf.index
            )

            # 输出结果
            print(f"迭代{i + 1}轮后 高置信度预测数据的数量: {len(y_conf)}")
            plot_distribution(y_conf, f'当前bins为{all_bins} 迭代{i + 1}轮后 高置信度的数据集标签分布',
                              os.path.join(output_dir,
                                           f'当前bins为{all_bins} 迭代{i + 1}轮后，高置信度的数据集标签分布.html'),
                              color='yellow', now_bins=all_bins)

            # 反向采样的过程
            sampling_counts, bins, max_samples_in_bin = calculate_sampling_rates(y_labeled, num_bins=all_bins)
            X_sampled, y_sampled = perform_sampling(X_conf, y_conf, sampling_counts, bins)

            X_initial_balance = pd.concat([X_labeled, X_sampled])
            y_initial_balance = pd.concat([y_labeled, y_sampled])

            # 在计算出y_conf后添加基尼系数分析
            if len(y_initial_balance) > 0:
                gini_analysis = analyze_imbalance_improvement(
                    y_labeled, y_initial_balance,
                    bins_value=all_bins,
                    iteration=i + 1,
                    output_dir=output_dir
                )
                gini_results.append(gini_analysis)

            # 计算使用未标记数据的均方误差（MSE）
            train_model(pseudo_model, pd.concat([X_labeled, X_sampled]), pd.concat([y_labeled, y_sampled]),
                        optimizer=pseude_model_optimizer, criterion=pseudo_model_criterion, epochs=200,
                        use_fixed_seed=False)

            # pseudo_model.fit(pd.concat([X_labeled, X_sampled]), pd.concat([y_labeled, y_sampled]))
            y_useunlabel_test_pred = predict(pseudo_model, X_test)

            # 计算每个区域的 MSE 和 MAE
            useunlabel_region_metrics = calculate_metrics_by_region(y_test, y_useunlabel_test_pred, region_labels)
            # 输出结果
            for region, metrics in useunlabel_region_metrics.items():
                print(
                    f"迭代{i + 1}轮后，使用未标记数据的区域{region}: MSE = {metrics['MSE']:.4f}, MAE = {metrics['MAE']:.4f}")

            useunlabel_mse = mean_squared_error(y_test, y_useunlabel_test_pred)
            # 计算平均绝对误差（MAE）
            useunlabel_mae = mean_absolute_error(y_test, y_useunlabel_test_pred)
            print(
                f"迭代{i + 1}轮后，ALL使用未标记数据的模型的均方误差 (MSE): {useunlabel_mse},使用未标记数据的模型的(MAE): {useunlabel_mae}")

            plot_distribution(y_sampled, f'当前bins为{all_bins} 迭代{i + 1}轮后 反向采样后采样的样本分布',
                              os.path.join(output_dir,
                                           f'当前bins为{all_bins} 迭代{i + 1}轮后，反向采样后采样的样本分布.html'),
                              now_bins=all_bins)
            print(f"迭代{i + 1}轮后，反向采样的数据的数量: {len(y_sampled)}")

            # 将高置信度的数据和不平衡数据混合
            X_combined = pd.concat([X_labeled, X_sampled])
            y_combined = pd.concat([y_labeled, y_sampled])
            print(f"迭代{i + 1}轮后 高置信度的数据和不平衡数据混合后数据的数量: {len(y_combined)}")
            plot_distribution(y_combined,
                              f'当前bins为{all_bins} 迭代{i + 1}轮后 高置信度的数据和不平衡数据混合后标签分布',
                              os.path.join(output_dir,
                                           f'当前bins为{all_bins} 迭代{i + 1}轮后，高置信度的数据和不平衡数据混合后标签分布.html'),
                              now_bins=all_bins)

            # 通过计算采样率后使用标签锚点混合数据增强方法来成新的数据
            X_real = pd.concat([X_labeled, X_conf]).reset_index(drop=True)
            y_real = pd.concat([y_labeled, y_conf]).reset_index(drop=True)
            lblsampling_counts, lblbins, max_samples_in_conbined = calculate_sampling_rates(y_combined,
                                                                                            num_bins=all_bins)
            X_augmented, y_augmented = label_anchored_mixup(X_real, y_real, lblsampling_counts, lblbins, alpha=0.2)

            print(f"标签锚点混合增强后数据集标签数量: {len(y_augmented)}")
            y_augmented = pd.Series(y_augmented)

            # 计算使用数据增强的均方误差（MSE）
            # augment_model.fit(
            #     pd.concat([X_labeled, X_augmented], ignore_index=True),
            #     pd.concat([y_labeled, y_augmented], ignore_index=True))
            train_model(augment_model, pd.concat([X_labeled, X_augmented], ignore_index=True),
                        pd.concat([y_labeled, y_augmented], ignore_index=True), optimizer=augment_model_optimizer,
                        criterion=augment_model_criterion, epochs=200, use_fixed_seed=False)

            # for name, param in augment_model.named_parameters():
            #     print(f"{name} 可训练:", param.requires_grad)
            # 应全为 True

            # # 模型参数
            # for name, param in augment_model.named_parameters():
            #     print(f"迭代{i}轮，augment_model的模型参数：")
            #     print(f"Layer: {name}")
            #     print("Parameters:")
            #     print(param.data)  # 打印参数的数值
            #     print("Gradient:")
            #     print(param.grad)  # 打印梯度（如果有的话）
            #     print("------------------------------")

            y_usemixupdate_test_pred = predict(augment_model, X_test)

            # 计算每个区域的 MSE 和 MAE
            usemixupdata_region_metrics = calculate_metrics_by_region(y_test, y_usemixupdate_test_pred, region_labels)
            # 输出结果
            for region, metrics in usemixupdata_region_metrics.items():
                print(
                    f"迭代{i + 1}轮后，使用数据增强方法后的区域{region}: MSE = {metrics['MSE']:.4f}, MAE = {metrics['MAE']:.4f}")

            usemixup_mse = mean_squared_error(y_test, y_usemixupdate_test_pred)
            usemixup_mae = mean_absolute_error(y_test, y_usemixupdate_test_pred)
            print(
                f"迭代{i + 1}轮后，ALL使用数据增强方法的模型的均方误差 (MSE): {usemixup_mse} 使用数据增强方法的模型的(MAE): {usemixup_mae}")

            plot_distribution(y_augmented, f'当前bins为{all_bins} 迭代{i + 1}轮后，标签锚点混合增强后数据集标签分布',
                              os.path.join(output_dir,
                                           f'当前bins为{all_bins} 迭代{i + 1}轮后，标签锚点混合增强后数据集标签分布.html'),
                              color='green', now_bins=all_bins)

            y_aug_combined = pd.concat([y_labeled, y_augmented])
            print(f"迭代{i + 1}轮后 数据增强后和不平衡数据混合后数据的数量: {len(y_aug_combined)}")

            # 将增强数据与原始数据混合
            X_final = pd.concat([X_combined, X_augmented])
            y_final = pd.concat([y_combined, y_augmented])
            plot_distribution(y_final, f'当前bins为{all_bins} 迭代{i + 1}轮后，最终将所有数据混合后数据集标签分布',
                              os.path.join(output_dir,
                                           f'当前bins为{all_bins} 迭代{i + 1}轮后，最终将所有数据混合后数据集标签分布.html'),
                              color='red', now_bins=all_bins)

            # 计算使用所有的均方误差（MSE）
            # model.fit(X_final, y_final)
            train_model(model, X_final, y_final, optimizer=final_optimizer, criterion=final_criterion, epochs=200,
                        use_fixed_seed=False)

            y_useall_test_pred = predict(model, X_test)

            # 计算每个区域的 MSE 和 MAE
            useall_region_metrics = calculate_metrics_by_region(y_test, y_useall_test_pred, region_labels)
            # 输出结果
            for region, metrics in useall_region_metrics.items():
                print(
                    f"迭代{i + 1}轮后，使用所有方法后的区域{region}: MSE = {metrics['MSE']:.4f}, MAE = {metrics['MAE']:.4f}")

            useall_mse = mean_squared_error(y_test, y_useall_test_pred)
            useall_mae = mean_absolute_error(y_test, y_useall_test_pred)
            print(
                f"迭代{i + 1}轮后，ALL使用所有方法的模型的均方误差 (MSE): {useall_mse} 使用所有方法的模型的(MAE): {useall_mae}")

            # 更新最低的MSE
            if useall_mse < min_mse:
                min_mse = useall_mse
            if useall_mae < min_mae:
                min_mae = useall_mae

            # 重置索引以确保唯一性
            X_labeled = X_labeled.reset_index(drop=True)
            y_labeled = pd.Series(y_labeled).reset_index(drop=True)
            X_conf = X_conf.reset_index(drop=True)
            y_conf = pd.Series(y_conf).reset_index(drop=True)
            X_augmented = pd.DataFrame(X_augmented, columns=X_labeled.columns).reset_index(drop=True)
            y_augmented = pd.Series(y_augmented).reset_index(drop=True)

        # 在函数结束前添加总结
        if gini_results:
            print(f"\n=== Bins={all_bins} 基尼系数改善总结 ===")
            avg_improvement = np.mean([r['improvement_percent'] for r in gini_results])
            print(f"平均不平衡改善程度: {avg_improvement:.2f}%")

            # 绘制改善趋势图
            plt.figure(figsize=(10, 6))
            iterations_list = range(1, len(gini_results) + 1)
            original_ginis = [r['gini_original'] for r in gini_results]
            confident_ginis = [r['gini_confident'] for r in gini_results]

            plt.plot(iterations_list, [initial_gini] * len(iterations_list), 'r--',
                     label=f'原始数据基尼系数 ({initial_gini:.4f})', linewidth=2)
            plt.plot(iterations_list, confident_ginis, 'g-o',
                     label='高置信度数据基尼系数', linewidth=2)

            plt.xlabel('迭代次数')
            plt.ylabel('基尼系数')
            plt.title(f'Bins={all_bins} - 基尼系数变化趋势')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if output_dir:
                plt.savefig(os.path.join(output_dir, f'bins_{all_bins}_gini_trend.png'),
                            dpi=300, bbox_inches='tight')
            plt.show()

        # 保存每个all_bisn值下的最低MSE
        min_mse_results.append(min_mse)
        min_mae_results.append(min_mae)

        # # 绘制预测方差分布
        # plt.hist(variances, bins=all_bins)
        # plt.xlabel("Prediction Variance")
        # plt.ylabel("Frequency")
        # plt.title("Distribution of Prediction Variances")
        # plt.show()

    # # 绘制不同 all_bins 与最低 MSE 的折线图
    # plt.figure(figsize=(15, 6))
    # plt.plot(all_bins_values, min_mse_results, marker='o')
    # plt.title(f"第{}all_bins 对 MSE 的影响")
    # plt.xlabel('all_bins')
    # plt.ylabel('MSE')
    # plt.grid(True)
    # plt.show()

    return min_mse_results, min_mae_results  # 返回每个bins的最小MSE和MAE列表


# ----------------- 运行实验并收集结果 -----------------
all_run_mse = []
all_run_mae = []

for run in range(NUM_RUNS):
    run_seed = SEED_BASE + run
    print(f"\n=== 第 {run + 1}/{NUM_RUNS} 次实验（Seed: {run_seed}）===")

    # 运行单次实验并保存结果
    mse_results, mae_results = run_experiment(run_seed)  # 接收两个返回值
    all_run_mse.append(mse_results)
    all_run_mae.append(mae_results)

# ----------------- 计算统计指标 -----------------
# 转换为NumPy数组以便计算
all_run_mse = np.array(all_run_mse)
all_run_mae = np.array(all_run_mae)

# 计算均值和标准差
mean_mse = np.mean(all_run_mse, axis=0)
std_mse = np.std(all_run_mse, axis=0)
mean_mae = np.mean(all_run_mae, axis=0)  # 新增MAE统计
std_mae = np.std(all_run_mae, axis=0)

# # 调用函数绘制训练数据特征的相关性矩阵
# plot_correlation_matrix(
#     X_full_train[features],  # 特征数据
#     title="训练数据特征相关性矩阵",
#     output_path=os.path.join(output_dir, "训练数据特征相关性矩阵.png")
# )
# print("特征相关性矩阵绘制完成")

# 输出结果
print("\n=== 最终结果 ===")
for idx, bins in enumerate(all_bins_values):
    print(f"Bins={bins}:")
    print(f"  MSE均值 = {mean_mse[idx]:.4f} ± {std_mse[idx]:.4f}")
    print(f"  MAE均值 = {mean_mae[idx]:.4f} ± {std_mae[idx]:.4f}")  # 新增MAE输出

# ----------------- 可视化结果 -----------------
plt.figure(figsize=(18, 8))

# MSE图
plt.subplot(1, 2, 1)
plt.errorbar(all_bins_values, mean_mse, yerr=std_mse, fmt='-o', capsize=5, color='blue')
plt.title('MSE随Bins变化（10次实验均值±标准差）')
plt.xlabel('Bins')
plt.ylabel('MSE')
plt.grid(True)

# MAE图
plt.subplot(1, 2, 2)
plt.errorbar(all_bins_values, mean_mae, yerr=std_mae, fmt='-o', capsize=5, color='red')
plt.title('MAE随Bins变化（10次实验均值±标准差）')
plt.xlabel('Bins')
plt.ylabel('MAE')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
plt.show()
