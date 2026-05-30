from sklearn.compose import ColumnTransformer

from smoter import SMOTER
import smogn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

SEED_BASE = 42


class TabularBaseNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[64, 32], dropout_rate=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加 Dropout
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加 Dropout
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze()


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
        # 检查预测结果
        assert not np.isnan(pred).any(), "预测结果中包含 NaN！"
    return pred


# ----------------- 定义 SMOTER 增强函数 -----------------
def apply_smoter(X, y):
    """
    使用 SMOTER 方法对数据进行增强。

    参数:
    X : pd.DataFrame
        特征数据。
    y : pd.Series
        标签数据。

    返回:
    X_resampled, y_resampled : pd.DataFrame, pd.Series
        增强后的特征和标签。
    """
    smoter = SMOTER(random_state=42)  # 初始化 SMOTER
    X_resampled, y_resampled = smoter.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)


# ----------------- 定义 SMOGN 增强函数 -----------------
def apply_smogn(X, y):
    # 强制对齐索引
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).rename('Y1')  # 确保列名正确

    # 合并数据并检查
    data = pd.concat([X, y], axis=1)
    if data.isnull().sum().sum() > 0:
        raise ValueError("输入数据包含 NaN，请检查预处理步骤！")

    # 调用 SMOGN
    smogn_data = smogn.smoter(
        data,
        y='Y1',
        rel_thres=0.1,  # 提高相关性阈值
        rel_method="auto",
        k=3,  # 减少近邻数，避免噪声过大
        pert=0.01  # 减少噪声强度
    )

    # 检查输出数据
    if smogn_data.isnull().sum().sum() > 0:
        smogn_data = smogn_data.dropna()  # 删除 NaN
        if smogn_data.empty:
            raise ValueError("SMOGN 增强后数据为空，请调整参数！")

    X_resampled = smogn_data.drop(columns='Y1')
    y_resampled = smogn_data['Y1']
    return X_resampled, y_resampled


def calculate_bins_sample(y, num_bins=10):
    bins = np.linspace(y.min(), y.max(), num_bins + 1)
    y_binned = pd.cut(y, bins, include_lowest=True)
    bin_counts = y_binned.value_counts().sort_index()
    return bin_counts, bins


def divide_regions(y, bins, sample_counts):
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
    y_binned = pd.cut(y, bins, include_lowest=True)
    region_labels = y_binned.map(region_map)
    return region_labels


def calculate_metrics_by_region(y_true, y_pred, region_labels):
    metrics = {}
    for region in ['Many', 'Med', 'Few']:
        mask = region_labels == region
        if mask.sum() > 0:
            mse = mean_squared_error(np.array(y_true)[mask], np.array(y_pred)[mask])
            mae = mean_absolute_error(np.array(y_true)[mask], np.array(y_pred)[mask])
            metrics[region] = {'MSE': mse, 'MAE': mae}
        else:
            metrics[region] = {'MSE': None, 'MAE': None}
    return metrics


# ----------------- 运行实验并添加 Baseline -----------------
def run_experiment_with_baselines(run_seed):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    data = pd.read_excel(r"e:\PycharmProject\SSIR\energy_efficiency\ENB2012_data.xlsx")

    # 选择特征和目标
    feature_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    target_col = 'Y1'  # 预测加热负荷

    X = data[feature_cols]
    y = data[target_col]

    # 识别类别特征
    categorical_features = ['Orientation', 'Glazing Area Distribution']
    categorical_features = [col for col in categorical_features if col in X.columns]

    # 独热编码类别特征
    if categorical_features:
        ct = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        X_encoded = ct.fit_transform(X)
        feature_names = ct.get_feature_names_out()
        X = pd.DataFrame(X_encoded, columns=feature_names)
    else:
        X = X.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run_seed)

    # 标准化数据
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
        X_train.reset_index(drop=True),  # 重置索引
        y_train.reset_index(drop=True),  # 重置索引
        train_size=0.3,
        random_state=42
    )

    # 保持 y_train 的列名
    y_labeled = pd.Series(y_labeled, name='Y1', dtype=np.float32)

    # 处理缺失值（可以根据需要选择不同的方法）
    X_labeled.dropna(inplace=True)
    X_test.dropna(inplace=True)
    # 检查是否还有NaN
    assert not X_test.isnull().any().any(), "X_test 仍然包含 NaN！"

    # ----------------- 原始数据的基线 -----------------
    print("\n=== Baseline (Original Data) ===")
    model = TabularBaseNet(input_dim=X_train.shape[1])
    base_optimizer = optim.Adam(model.parameters(), lr=1e-3)
    base_criterion = nn.MSELoss()
    train_model(model, X_labeled, y_labeled, base_optimizer, base_criterion, epochs=100)

    y_pred = predict(model, X_test)
    baseline_mse = mean_squared_error(y_test, y_pred)
    baseline_mae = mean_absolute_error(y_test, y_pred)
    print(f"Original Data: MSE = {baseline_mse:.4f}, MAE = {baseline_mae:.4f}")

    # 新增：分区MSE/MAE
    bin_counts, bins = calculate_bins_sample(y_test, num_bins=12)
    region_map = divide_regions(y_test, bins, bin_counts)
    region_labels = assign_region_labels(y_test, bins, region_map)
    region_metrics = calculate_metrics_by_region(y_test, y_pred, region_labels)
    for region, metrics in region_metrics.items():
        print(f"Original Data 区域{region}: MSE = {metrics['MSE']}, MAE = {metrics['MAE']}")

    # ----------------- 使用 SMOTER 增强 -----------------
    print("\n=== Baseline (SMOTER) ===")
    X_smoter, y_smoter = apply_smoter(X_labeled, y_labeled)
    model_smoter = TabularBaseNet(input_dim=X_smoter.shape[1])
    sr_optimizer_smoter = optim.Adam(model_smoter.parameters(), lr=1e-3)
    sr_criterion = nn.MSELoss()
    train_model(model_smoter, X_smoter, y_smoter, sr_optimizer_smoter, sr_criterion, epochs=100)

    y_smoter_pred = predict(model_smoter, X_test)
    smoter_mse = mean_squared_error(y_test, y_smoter_pred)
    smoter_mae = mean_absolute_error(y_test, y_smoter_pred)
    print(f"SMOTER: MSE = {smoter_mse:.4f}, MAE = {smoter_mae:.4f}")

    region_metrics = calculate_metrics_by_region(y_test, y_smoter_pred, region_labels)
    for region, metrics in region_metrics.items():
        print(f"SMOTER 区域{region}: MSE = {metrics['MSE']}, MAE = {metrics['MAE']}")

    # ----------------- 使用 SMOGN 增强 -----------------
    print("\n=== Baseline (SMOGN) ===")
    X_smogn, y_smogn = apply_smogn(X_labeled, y_labeled)
    model_smogn = TabularBaseNet(input_dim=X_smogn.shape[1])
    optimizer_smogn = optim.Adam(model_smogn.parameters(), lr=1e-3)
    sgn_criterion = nn.MSELoss()
    train_model(model_smogn, X_smogn, y_smogn, optimizer_smogn, sgn_criterion, epochs=100)

    y_smogn_pred = predict(model_smogn, X_test)
    smogn_mse = mean_squared_error(y_test, y_smogn_pred)
    smogn_mae = mean_absolute_error(y_test, y_smogn_pred)
    print(f"SMOGN: MSE = {smogn_mse:.4f}, MAE = {smogn_mae:.4f}")

    region_metrics = calculate_metrics_by_region(y_test, y_smogn_pred, region_labels)
    for region, metrics in region_metrics.items():
        print(f"SMOGN 区域{region}: MSE = {metrics['MSE']}, MAE = {metrics['MAE']}")

    return {
        "Original": {"MSE": baseline_mse, "MAE": baseline_mae},
        "SMOTER": {"MSE": smoter_mse, "MAE": smoter_mae},
        "SMOGN": {"MSE": smogn_mse, "MAE": smogn_mae}
    }


# ----------------- 运行实验 -----------------
run_seed = 42
baseline_results = run_experiment_with_baselines(run_seed)

# 打印最终结果
print("\n=== Final Baseline Results ===")
for method, metrics in baseline_results.items():
    print(f"{method}: MSE = {metrics['MSE']:.4f}, MAE = {metrics['MAE']:.4f}")
