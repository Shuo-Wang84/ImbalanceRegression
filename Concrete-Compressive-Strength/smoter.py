import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class SMOTER:
    def __init__(self, k=5, threshold=0.1, sampling_strategy='auto', random_state=None):
        """
        SMOTER 算法实现

        参数:
        - k: 近邻数量 (默认 5)
        - threshold: 定义少数区域的分位数阈值 (默认 0.1)
        - random_state: 随机种子
        """
        self.k = k
        self.threshold = threshold
        self.sampling_strategy = sampling_strategy  # 'auto', 'balance', 或具体数字
        self.random_state = random_state

    def fit_resample(self, X, y):
        """
        对输入数据进行过采样

        参数:
        - X: 特征数据 (pd.DataFrame 或 np.ndarray)
        - y: 目标值 (pd.Series 或 np.ndarray)

        返回:
        - X_resampled: 增强后的特征数据
        - y_resampled: 增强后的目标值
        """
        # 转换为 NumPy 格式
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # 设置随机种子
        np.random.seed(self.random_state)

        # 步骤 1: 识别少数区域 (上下尾)
        lower_bound = np.quantile(y, self.threshold)
        upper_bound = np.quantile(y, 1 - self.threshold)
        minority_mask = (y <= lower_bound) | (y >= upper_bound)
        X_minority = X[minority_mask]
        y_minority = y[minority_mask]

        # 步骤 2: 计算每个少数样本的 k 近邻
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(X_minority)
        _, indices = nbrs.kneighbors(X_minority)

        # 步骤 3: 生成合成样本
        synthetic_samples = []
        synthetic_targets = []
        for i in range(len(X_minority)):
            # 随机选择一个邻居
            neighbor_idx = np.random.choice(indices[i][1:])  # 排除自身
            # 插值比例
            ratio = np.random.uniform(0, 1)
            # 生成新样本
            new_sample = X_minority[i] + ratio * (X_minority[neighbor_idx] - X_minority[i])
            new_target = y_minority[i] + ratio * (y_minority[neighbor_idx] - y_minority[i])
            synthetic_samples.append(new_sample)
            synthetic_targets.append(new_target)

        # 合并原始数据和合成数据
        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.hstack([y, synthetic_targets])

        return X_resampled, y_resampled