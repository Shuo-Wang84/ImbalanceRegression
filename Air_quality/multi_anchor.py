import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def build_multi_anchors(
    X: pd.DataFrame,
    y: pd.Series,
    bins: Any,
    # 下面三个阈值控制“是否启用多锚点”与“最多多少锚点”
    N_min_bin_for_multi: int = 100,      # 分箱内样本数不足时直接单锚点
    N_min_component: int = 25,          # 每个簇至少需要的样本数
    k_max: int = 3,                     # 每个分箱最多的锚点数量上限
    silhouette_threshold: float = 0.3,
    # 聚类选择：'gmm' 更稳健；'kmeans' 更简单
    method: str = "gmm",
    covariance_type: str = "diag",      # GMM 的协方差类型
    label_mode: str = "median",         # 锚点的标签取值：'median'/'mean'/'quantile'
    random_state: int = 42
) -> Tuple[Dict[pd.Interval, List[Dict[str, Any]]], Dict[pd.Interval, Dict[str, Any]]]:
    """
    根据每个 y 分箱内的样本充足度与聚类结构，自适应选择锚点数量并构造锚点。
    返回:
      - anchors_by_bin: {interval: [ {mu, cov, inv_cov, y_anchor, indices}, ... ]}
      - bin_info: {interval: {n_bin, k_bin, method, bic_list, sil_list}}
    """
    assert isinstance(X, pd.DataFrame), "X 必须是 DataFrame"
    assert isinstance(y, pd.Series), "y 必须是 Series"
    d = X.shape[1]

    y_binned = pd.cut(y, bins, include_lowest=True)
    anchors_by_bin: Dict[pd.Interval, List[Dict[str, Any]]] = {}
    bin_info: Dict[pd.Interval, Dict[str, Any]] = {}

    for interval in y_binned.cat.categories:
        idx_bin = (y_binned == interval)
        X_bin = X.loc[idx_bin]
        y_bin = y.loc[idx_bin]
        n_bin = len(X_bin)

        # 小样本分箱直接单锚点，稳健优先
        if n_bin < N_min_bin_for_multi:
            mu = X_bin.mean().values if n_bin > 0 else np.zeros(d)
            cov = np.cov(X_bin.values, rowvar=False) if n_bin > 1 else np.eye(d)
            cov += np.eye(d) * 1e-6
            inv_cov = np.linalg.inv(cov)
            y_anchor = float(np.median(y_bin)) if n_bin > 0 else float("nan")

            anchors_by_bin[interval] = [{
                "mu": mu, "cov": cov, "inv_cov": inv_cov,
                "y_anchor": y_anchor, "indices": list(X_bin.index)
            }]
            bin_info[interval] = {
                "n_bin": n_bin, "k_bin": 1, "reason": "insufficient_samples",
                "method": method, "bic_list": [], "sil_list": []
            }
            continue

        # 可选锚点上限受样本数约束
        effective_k_max = min(k_max, max(1, n_bin // N_min_component))
        chosen_k = 1
        labels = np.zeros(n_bin, dtype=int)
        bic_list: List[Tuple[int, float]] = []
        sil_list: List[Tuple[int, float]] = []

        # 用 GMM 的 BIC 选择 k；或用 KMeans 的 silhouette 选择 k
        if method.lower() == "gmm":
            best_bic = np.inf
            best_labels = None
            for k in range(1, effective_k_max + 1):
                try:
                    gmm = GaussianMixture(
                        n_components=k,
                        covariance_type=covariance_type,
                        random_state=random_state
                    )
                    gmm.fit(X_bin.values)
                    bic = gmm.bic(X_bin.values)
                    bic_list.append((k, bic))

                    lbls = gmm.predict(X_bin.values)
                    if k > 1 and len(set(lbls)) > 1:
                        sil = silhouette_score(X_bin.values, lbls)
                    else:
                        sil = 0.0
                    sil_list.append((k, sil))

                    if bic < best_bic:
                        best_bic = bic
                        chosen_k = k
                        best_labels = lbls
                except Exception:
                    # 某些 k 会因奇异协方差失败，直接跳过
                    pass
            labels = best_labels if best_labels is not None else np.zeros(n_bin, dtype=int)

        else:  # kmeans
            best_sil = -1.0
            best_labels = None
            for k in range(1, effective_k_max + 1):
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                km.fit(X_bin.values)
                lbls = km.labels_
                if k > 1 and len(set(lbls)) > 1:
                    sil = silhouette_score(X_bin.values, lbls)
                else:
                    sil = 0.0
                sil_list.append((k, sil))

                if k == 1 or sil > best_sil:
                    best_sil = sil if k > 1 else 0.0
                    chosen_k = k
                    best_labels = lbls
            labels = best_labels if best_labels is not None else np.zeros(n_bin, dtype=int)

        # 保证每个簇的样本数足够，否则缩小 k 并重新聚类
        def shrink_k(curr_labels: np.ndarray, target_k: int) -> Tuple[int, np.ndarray]:
            curr_k = target_k
            X_vals = X_bin.values
            while curr_k > 1:
                counts = np.bincount(curr_labels, minlength=curr_k)
                if counts.min() >= N_min_component:
                    break
                curr_k -= 1
                if method.lower() == "gmm":
                    gmm = GaussianMixture(
                        n_components=curr_k,
                        covariance_type=covariance_type,
                        random_state=random_state
                    )
                    gmm.fit(X_vals)
                    curr_labels = gmm.predict(X_vals)
                else:
                    km = KMeans(n_clusters=curr_k, random_state=random_state, n_init=10)
                    km.fit(X_vals)
                    curr_labels = km.labels_
            return curr_k, curr_labels

        chosen_k, labels = shrink_k(labels, chosen_k)

        # 分离度不足则退回单锚点
        if chosen_k > 1:
            try:
                sil = silhouette_score(X_bin.values, labels)
            except Exception:
                sil = 0.0
            if sil < silhouette_threshold:
                chosen_k = 1
                labels = np.zeros(n_bin, dtype=int)

        # 构造锚点
        anchors: List[Dict[str, Any]] = []
        for comp in range(chosen_k):
            comp_mask = (labels == comp)
            comp_idx = X_bin.index[comp_mask]
            X_comp = X.loc[comp_idx]
            y_comp = y.loc[comp_idx]

            mu = X_comp.mean().values
            cov = np.cov(X_comp.values, rowvar=False) if len(X_comp) > 1 else np.eye(d)
            cov += np.eye(d) * 1e-6
            inv_cov = np.linalg.inv(cov)

            if label_mode == "median":
                y_anchor = float(np.median(y_comp))
            elif label_mode == "quantile":
                # 取中位（或可扩展为选择最接近分箱中心的分位）
                y_anchor = float(np.quantile(y_comp, 0.5))
            else:
                y_anchor = float(np.mean(y_comp))

            anchors.append({
                "mu": mu, "cov": cov, "inv_cov": inv_cov,
                "y_anchor": y_anchor, "indices": list(comp_idx)
            })

        anchors_by_bin[interval] = anchors
        bin_info[interval] = {
            "n_bin": n_bin, "k_bin": chosen_k, "method": method,
            "bic_list": bic_list, "sil_list": sil_list
        }

    return anchors_by_bin, bin_info