import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import shap
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import os

# 设置中文字体 - 更可靠的方法
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class ConfidenceVisualization:
    def __init__(self, output_dir=r"F:\SGIR\SGIR-main\plt画图\MLP\加利福尼亚房价预测"):
        self.output_dir = output_dir
        self.feature_names = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]

    def load_and_prepare_data(self):
        """加载和预处理数据"""
        # 加载加利福尼亚房价数据
        california = fetch_california_housing()
        X, y = california.data, california.target

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    def train_model(self, X_train, y_train):
        """训练模型"""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def calculate_shap_values(self, model, X_sample):
        """计算SHAP值"""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return shap_values, explainer

    def generate_perturbations(self, X_sample, important_features, noise_level=0.1, n_perturbations=50):
        """生成扰动样本"""
        perturbations = []
        for _ in range(n_perturbations):
            perturbed = X_sample.copy()
            for feature_idx in important_features:
                noise = np.random.normal(0, noise_level)
                perturbed[feature_idx] += noise
            perturbations.append(perturbed)
        return np.array(perturbations)

    def calculate_confidence(self, model, X_sample, perturbations):
        """计算置信度"""
        # 原始预测
        original_pred = model.predict(X_sample.reshape(1, -1))[0]

        # 扰动样本预测
        perturbed_preds = model.predict(perturbations)

        # 计算方差作为不确定性度量
        variance = np.var(perturbed_preds)

        # 置信度 = 1 / (1 + variance)
        confidence = 1 / (1 + variance)

        return original_pred, perturbed_preds, variance, confidence

    def plot_confidence_process(self, save_path=None):
        """绘制完整的置信度计算流程"""
        # 准备数据
        X_train, X_test, y_train, y_test, scaler = self.load_and_prepare_data()
        model = self.train_model(X_train, y_train)

        # 选择一个测试样本
        sample_idx = 0
        X_sample = X_test[sample_idx]

        # 计算SHAP值
        shap_values, explainer = self.calculate_shap_values(model, X_sample.reshape(1, -1))
        feature_importance = np.abs(shap_values[0])

        # 选择重要特征（前4个）
        important_indices = np.argsort(feature_importance)[-4:]

        # 生成扰动样本
        perturbations = self.generate_perturbations(X_sample, important_indices)

        # 计算置信度
        original_pred, perturbed_preds, variance, confidence = self.calculate_confidence(
            model, X_sample, perturbations
        )

        # 创建图形
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. SHAP特征重要性热力图
        ax1 = fig.add_subplot(gs[0, :2])
        importance_matrix = feature_importance.reshape(1, -1)
        im1 = ax1.imshow(importance_matrix, cmap='RdYlBu_r', aspect='auto')
        ax1.set_xticks(range(len(self.feature_names)))
        ax1.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax1.set_yticks([])
        ax1.set_title('步骤1: SHAP特征重要性分析', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(im1, ax=ax1, shrink=0.6)

        # 2. 关键特征识别
        ax2 = fig.add_subplot(gs[0, 2:])
        important_names = [self.feature_names[i] for i in important_indices]
        important_values = feature_importance[important_indices]
        colors = plt.cm.Set3(np.linspace(0, 1, len(important_names)))
        bars = ax2.bar(important_names, important_values, color=colors)
        ax2.set_title('步骤2: 关键特征识别 (Top 4)', fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylabel('SHAP重要性')
        ax2.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, value in zip(bars, important_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=10)

        # 3. 扰动样本生成可视化
        ax3 = fig.add_subplot(gs[1, :2])
        # 显示原始样本和部分扰动样本的对比
        sample_data = []
        labels = ['原始样本']
        sample_data.append(X_sample[important_indices])

        # 选择几个扰动样本进行展示
        for i in range(min(5, len(perturbations))):
            sample_data.append(perturbations[i][important_indices])
            labels.append(f'扰动样本{i + 1}')

        sample_matrix = np.array(sample_data)
        im3 = ax3.imshow(sample_matrix, cmap='viridis', aspect='auto')
        ax3.set_xticks(range(len(important_names)))
        ax3.set_xticklabels(important_names, rotation=45, ha='right')
        ax3.set_yticks(range(len(labels)))
        ax3.set_yticklabels(labels)
        ax3.set_title('步骤3: 扰动样本生成', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(im3, ax=ax3, shrink=0.6)

        # 4. 预测方差计算
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.hist(perturbed_preds, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(original_pred, color='red', linestyle='--', linewidth=2, label=f'原始预测: {original_pred:.3f}')
        ax4.axvline(np.mean(perturbed_preds), color='green', linestyle='--', linewidth=2,
                    label=f'扰动均值: {np.mean(perturbed_preds):.3f}')
        ax4.set_xlabel('预测值')
        ax4.set_ylabel('频次')
        ax4.set_title(f'步骤4: 预测方差计算\n方差: {variance:.4f}', fontsize=14, fontweight='bold', pad=20)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 置信度分类结果
        ax5 = fig.add_subplot(gs[2, :2])
        confidence_levels = ['低置信度\n(<0.7)', '中等置信度\n(0.7-0.85)', '高置信度\n(>0.85)']
        confidence_colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']

        # 确定当前样本的置信度级别
        if confidence < 0.7:
            current_level = 0
        elif confidence < 0.85:
            current_level = 1
        else:
            current_level = 2

        # 绘制置信度分类
        bars = ax5.bar(confidence_levels, [0.7, 0.85, 1.0],
                       color=[confidence_colors[i] if i != current_level else 'red'
                              for i in range(3)], alpha=0.7)

        # 标记当前样本的置信度
        ax5.axhline(confidence, color='red', linestyle='-', linewidth=3,
                    label=f'当前样本置信度: {confidence:.3f}')

        ax5.set_ylabel('置信度阈值')
        ax5.set_title('步骤5: 置信度分类结果', fontsize=14, fontweight='bold', pad=20)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. 置信度计算公式和结果展示
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')

        # 创建文本框显示计算过程
        formula_text = f"""
        置信度计算流程总结:

        1. 特征重要性: SHAP分析
        2. 关键特征: {len(important_indices)}个最重要特征
        3. 扰动生成: {len(perturbations)}个扰动样本
        4. 预测方差: {variance:.4f}
        5. 置信度公式: 1/(1+方差)

        最终置信度: {confidence:.4f}
        置信度等级: {confidence_levels[current_level].replace(chr(10), ' ')}

        解释: 方差越小，预测越稳定，置信度越高
        """

        # 添加背景框
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8)
        ax6.text(0.05, 0.95, formula_text, transform=ax6.transAxes, fontsize=12,
                 verticalalignment='top', bbox=bbox_props, family='monospace')

        ax6.set_title('步骤6: 置信度计算总结', fontsize=14, fontweight='bold')

        # 添加总标题
        fig.suptitle('机器学习模型置信度计算完整流程可视化', fontsize=18, fontweight='bold', y=0.98)

        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"置信度计算流程图已保存至: {save_path}")

        plt.tight_layout()
        # plt.show()  # 注释掉这一行
        plt.close()  # 添加这一行来释放内存

        return {
            'confidence': confidence,
            'variance': variance,
            'original_prediction': original_pred,
            'feature_importance': feature_importance,
            'important_features': important_names
        }

    def plot_simple_confidence_overview(self, save_path=None):
        """绘制简化的置信度概览图"""
        # 准备数据
        X_train, X_test, y_train, y_test, scaler = self.load_and_prepare_data()
        model = self.train_model(X_train, y_train)

        # 计算多个样本的置信度
        n_samples = min(20, len(X_test))
        confidences = []
        predictions = []

        for i in range(n_samples):
            X_sample = X_test[i]
            shap_values, _ = self.calculate_shap_values(model, X_sample.reshape(1, -1))
            feature_importance = np.abs(shap_values[0])
            important_indices = np.argsort(feature_importance)[-4:]

            perturbations = self.generate_perturbations(X_sample, important_indices, n_perturbations=30)
            original_pred, perturbed_preds, variance, confidence = self.calculate_confidence(
                model, X_sample, perturbations
            )

            confidences.append(confidence)
            predictions.append(original_pred)

        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 置信度分布
        ax1.hist(confidences, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--',
                    label=f'平均置信度: {np.mean(confidences):.3f}')
        ax1.set_xlabel('置信度')
        ax1.set_ylabel('样本数量')
        ax1.set_title('样本置信度分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 置信度vs预测值散点图
        scatter = ax2.scatter(predictions, confidences, c=confidences, cmap='RdYlGn', alpha=0.7)
        ax2.set_xlabel('预测值')
        ax2.set_ylabel('置信度')
        ax2.set_title('预测值 vs 置信度')
        plt.colorbar(scatter, ax=ax2)
        ax2.grid(True, alpha=0.3)

        # 3. 置信度等级饼图
        low_conf = sum(1 for c in confidences if c < 0.7)
        med_conf = sum(1 for c in confidences if 0.7 <= c < 0.85)
        high_conf = sum(1 for c in confidences if c >= 0.85)

        labels = ['低置信度(<0.7)', '中等置信度(0.7-0.85)', '高置信度(≥0.85)']
        sizes = [low_conf, med_conf, high_conf]
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']

        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('置信度等级分布')

        # 4. 置信度趋势
        ax4.plot(range(len(confidences)), sorted(confidences, reverse=True),
                 marker='o', linestyle='-', color='blue', alpha=0.7)
        ax4.axhline(0.7, color='orange', linestyle='--', alpha=0.7, label='低置信度阈值')
        ax4.axhline(0.85, color='green', linestyle='--', alpha=0.7, label='高置信度阈值')
        ax4.set_xlabel('样本排序（按置信度降序）')
        ax4.set_ylabel('置信度')
        ax4.set_title('样本置信度排序')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('模型置信度分析概览', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"置信度概览图已保存至: {save_path}")

        # plt.show()  # 注释掉这一行
        plt.close()  # 添加这一行来释放内存

        return {
            'confidences': confidences,
            'predictions': predictions,
            'avg_confidence': np.mean(confidences),
            'confidence_distribution': {'low': low_conf, 'medium': med_conf, 'high': high_conf}
        }


# 使用示例
if __name__ == "__main__":
    # 创建可视化对象
    viz = ConfidenceVisualization()

    # 绘制完整的置信度计算流程
    print("正在生成置信度计算流程图...")
    result1 = viz.plot_confidence_process(
        save_path=r"F:\SGIR\SGIR-main\plt画图\MLP\加利福尼亚房价预测\置信度计算流程.png"
    )

    # 绘制置信度概览图
    print("正在生成置信度概览图...")
    result2 = viz.plot_simple_confidence_overview(
        save_path=r"F:\SGIR\SGIR-main\plt画图\MLP\加利福尼亚房价预测\置信度分析概览.png"
    )

    print("\n=== 分析结果 ===")
    print(f"单样本置信度: {result1['confidence']:.4f}")
    print(f"预测方差: {result1['variance']:.4f}")
    print(f"重要特征: {result1['important_features']}")
    print(f"平均置信度: {result2['avg_confidence']:.4f}")
    print(f"置信度分布: {result2['confidence_distribution']}")