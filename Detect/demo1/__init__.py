#demo1

import torch
import numpy as np
import os
import time
import random
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc as calc_auc
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from torch.utils.data import DataLoader
from scipy.spatial.distance import mahalanobis
from scipy import stats
import warnings
import glob
from datetime import datetime
import re

from data_loader import TimeSeriesDataset, custom_collate
from config import DEVICE, WIN_SIZE, get_model_params, EXPERIMENTS
from model_factory import create_model

# ====== 随机性控制 ======
def set_all_seeds(seed=42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)  # 对于 numpy 的新随机生成器
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置 matplotlib 的随机种子（如果使用）
    matplotlib.use('Agg')  # 使用非交互式后端

# 在代码开始处调用
set_all_seeds(42)

# 修改字体设置：完全使用英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# ====== 日志记录系统 ======
class DetectionLogger:
    """检测日志记录器 - 将所有print内容分类保存到CSV"""

    def __init__(self, output_dir='./detection_logs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 创建不同的日志类别
        self.log_categories = {
            'config': [],
            'data_loading': [],
            'model_loading': [],
            'distance_calculation': [],
            'fault_line': [],
            'performance': [],
            'misclassification': [],
            'plotting': [],
            'general': []
        }

        self.current_experiment = None
        self.current_model = None

    def set_experiment_context(self, experiment_name, model_name):
        """设置当前实验和模型上下文"""
        self.current_experiment = experiment_name
        self.current_model = model_name

    def log(self, category, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'experiment': self.current_experiment,
            'model': self.current_model,
            'category': category,
            'level': level,
            'message': message
        }

        # 添加到对应的类别
        if category in self.log_categories:
            self.log_categories[category].append(log_entry)
        else:
            self.log_categories['general'].append(log_entry)

        # 同时打印到控制台（保持原有行为）
        print(f"[{level}] {message}")

    def save_all_logs(self):
        """保存所有日志到CSV文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存分类日志
        for category, logs in self.log_categories.items():
            if logs:  # 只保存有内容的类别
                df = pd.DataFrame(logs)
                csv_path = os.path.join(self.output_dir, f'{category}_logs_{timestamp}.csv')
                df.to_csv(csv_path, index=False, encoding='utf-8')
                self.log('general', f"{category} 日志已保存: {csv_path}")

        # 2. 保存合并的完整日志
        all_logs = []
        for category_logs in self.log_categories.values():
            all_logs.extend(category_logs)

        if all_logs:
            df_all = pd.DataFrame(all_logs)
            csv_all_path = os.path.join(self.output_dir, f'all_logs_{timestamp}.csv')
            df_all.to_csv(csv_all_path, index=False, encoding='utf-8')
            self.log('general', f"完整日志已保存: {csv_all_path}")

    def get_category_logs(self, category):
        """获取特定类别的日志"""
        return self.log_categories.get(category, [])

# ====== 全局日志记录器实例 ======
detection_logger = DetectionLogger()

# ====== 全局配置 ======
DETECTION_CONFIG = {
    'n_components': 1,
    'batch_size': 8,
    'distance_method': 'stable_mahalanobis',  # 修改这里
    'robust_covariance': False,               # 修改这里
    'fault_line_method': 'robust_upper_bound',
    'outlier_threshold': 2.5,
    'confidence_level': 0.95,
    'remove_extreme_k': 5,
    'remove_test_extreme_k': 10,       # 🆕 测试集正常样本去除极端值
    'deviation_threshold': 3.0,
    'safety_margin': 1.0
}
shapes_info = []

def filter_test_extreme_values(test_distances_norm, labels, remove_test_extreme_k):
    """过滤测试集中正常样本的极端值"""
    normal_mask = (labels == 0)
    anomaly_mask = (labels == 1)

    # 只对正常样本进行过滤
    normal_scores = test_distances_norm[normal_mask]
    normal_indices = np.where(normal_mask)[0]

    if len(normal_scores) > remove_test_extreme_k:
        # 找出正常样本中的极端值
        median = np.median(normal_scores)
        mad = stats.median_abs_deviation(normal_scores, scale='normal')
        deviations = np.abs(normal_scores - median) / mad

        # 找出偏离最大的k个样本
        extreme_indices = np.argsort(deviations)[-remove_test_extreme_k:]
        extreme_test_indices = normal_indices[extreme_indices]

        # 创建过滤掩码
        filter_mask = np.ones(len(test_distances_norm), dtype=bool)
        filter_mask[extreme_test_indices] = False

        detection_logger.log('data_loading',
                           f"过滤测试集正常样本极端值: 移除{remove_test_extreme_k}个, "
                           f"剩余{np.sum(filter_mask)}/{len(test_distances_norm)}样本")

        return test_distances_norm[filter_mask], labels[filter_mask], filter_mask
    else:
        return test_distances_norm, labels, np.ones(len(test_distances_norm), dtype=bool)

# ====== 模型名称到文件前缀的映射 ======
MODEL_NAME_MAPPING = {
    'PredictiveSNN_ODE_Model': 'snn_ode',
    'PredictiveLatentODE_Model': 'latent_ode',
    'PredictiveSNN_ODEATT_Model': 'snn_odeatt',
    'PredictiveSNN_Model': 'snn',
    'PredictiveODERNN_Model': 'ode_rnn',
    'PredictiveRNN_Model': 'rnn',
    'PredictiveBiLSTM_Model': 'bilstm',
    'PredictiveDF2M_Model': 'df2m',
    'PredictiveStatLinearAR_Model': 'stat_linear_ar',
    'PredictiveStatLassoAR_Model': 'stat_lasso_ar',
    'PredictiveStatKRR_Model': 'stat_krr',
    'PredictiveStatFPCA_Model': 'stat_fpca',
    'PredictiveStatCumInt_Model': 'stat_cumint',
}


def _safe_model_forward(model, x, m, delta_t):
    """
    统一调用模型 forward，兼容:
    - model(x, m, delta_t)
    - model(x, m, delta_ts=delta_t)
    - model(x, m)
    """
    try:
        outputs = model(x, m, delta_t)
    except TypeError:
        try:
            outputs = model(x, m, delta_ts=delta_t)
        except TypeError:
            outputs = model(x, m)
    return outputs


def _parse_model_outputs(outputs, model_name="UnknownModel"):
    """
    统一解析模型输出，兼容旧4输出和新7输出。
    返回:
        recons, preds, z_mean, z_seq
    约定:
    - 4输出: (recons, preds, z_mean, z_seq)
    - 7输出: (recons, preds, z_mean, mem_seq, mem_reset_seq, spike_or_rate_seq, h_seq)
      检测阶段优先使用 mem_reset_seq 作为时序隐变量
    """
    if not isinstance(outputs, (tuple, list)):
        raise ValueError(
            f"{model_name}.forward() 输出类型错误: 期望 tuple/list，实际为 {type(outputs)}"
        )

    if len(outputs) == 4:
        recons, preds, z_mean, z_seq = outputs
        return recons, preds, z_mean, z_seq

    if len(outputs) >= 7:
        recons = outputs[0]
        preds = outputs[1]
        z_mean = outputs[2]

        # 对你的 SNN_ODE / SNN_ODEATT:
        # outputs[3] = mem_seq
        # outputs[4] = mem_reset_seq
        # 检测特征 z_mean 已经是对 mem_reset_seq 聚合得到
        # 所以这里优先把 z_seq 对齐到 mem_reset_seq
        mem_reset_seq = outputs[4]
        return recons, preds, z_mean, mem_reset_seq

    raise ValueError(
        f"{model_name}.forward() 输出数量不足: 期望至少4项，实际为 {len(outputs)}"
    )


def extract_detection_features(model, x, m, delta_t, model_name="UnknownModel"):
    """
    统一提取检测阶段需要的特征
    返回:
        z_mean: [B, H]
        z_seq : [B, T, H]
    """
    outputs = _safe_model_forward(model, x, m, delta_t)
    _, _, z_mean, z_seq = _parse_model_outputs(outputs, model_name=model_name)
    return z_mean, z_seq


def _get_clean_training_scores(train_distances, normalizer, fault_line_calculator):
    """获取用于计算故障线的清洁训练集分数"""
    distances = np.array(train_distances)

    median = np.median(distances)
    mad = stats.median_abs_deviation(distances, scale='normal')

    if len(distances) > fault_line_calculator.remove_extreme_k:
        deviations = np.abs(distances - median) / mad
        extreme_indices = np.argsort(deviations)[-fault_line_calculator.remove_extreme_k:]

        clean_mask = np.ones(len(distances), dtype=bool)
        clean_mask[extreme_indices] = False
        clean_distances = distances[clean_mask]

        detection_logger.log('fault_line', f"清洁训练集: {len(clean_distances)}/{len(distances)} 样本")

        if normalizer is not None:
            clean_distances_norm = normalizer.transform(clean_distances)
            return clean_distances_norm
        else:
            return clean_distances
    else:
        if normalizer is not None:
            return normalizer.transform(distances)
        else:
            return distances

def validate_config_consistency(config):
    """验证配置一致性"""
    required_keys = ['distance_method', 'fault_line_method']
    for key in required_keys:
        if key not in config:
            detection_logger.log('config', f"Missing required config key: {key}", "ERROR")
            raise ValueError(f"Missing required config key: {key}")

    detection_logger.log('config', "使用配置:")
    for key, value in config.items():
        detection_logger.log('config', f"  {key}: {value}")

class RobustDistanceCalculator:
    """稳健距离计算器 - 仅使用robust_mahalanobis方法"""

    def __init__(self, robust_covariance=True, confidence_level=0.95):
        self.robust_covariance = robust_covariance
        self.confidence_level = confidence_level
        self.scaler = None
        self.train_mean = None
        self.train_cov = None
        self.inv_cov = None
        self.is_fitted = False

    def fit(self, train_features):
        """拟合距离计算器"""
        detection_logger.log('distance_calculation', "使用 robust_mahalanobis 方法拟合距离计算器")

        self.scaler = RobustScaler().fit(train_features)
        train_scaled = self.scaler.transform(train_features)
        self.train_mean = np.median(train_scaled, axis=0)

        if self.robust_covariance:
            try:
                cov_estimator = MinCovDet(support_fraction=0.75, random_state=42, assume_centered=True)
                cov_estimator.fit(train_scaled)
                self.train_cov = cov_estimator.covariance_
                detection_logger.log('distance_calculation', "使用MinCovDet稳健协方差估计")
            except Exception as e:
                detection_logger.log('distance_calculation', f"MinCovDet失败，使用标准协方差: {e}")
                self.train_cov = np.cov(train_scaled, rowvar=False)
        else:
            self.train_cov = np.cov(train_scaled, rowvar=False)

        reg_cov = self.train_cov + np.eye(self.train_cov.shape[0]) * 1e-6

        try:
            self.inv_cov = np.linalg.inv(reg_cov)
        except:
            self.inv_cov = np.linalg.pinv(reg_cov)
            detection_logger.log('distance_calculation', "使用伪逆矩阵")

        self.is_fitted = True
        detection_logger.log('distance_calculation', f"距离计算器拟合完成，特征维度: {train_features.shape[1]}")

    def calculate_distance(self, features):
        """计算特征到训练集的距离"""
        if not self.is_fitted:
            raise ValueError("距离计算器尚未拟合")

        features_scaled = self.scaler.transform(features)
        distances = []

        for i in range(len(features_scaled)):
            try:
                dist = mahalanobis(features_scaled[i], self.train_mean, self.inv_cov)
                distances.append(dist)
            except Exception as e:
                dist = np.linalg.norm(features_scaled[i] - self.train_mean)
                distances.append(dist)

        return np.array(distances)

class StableDistanceCalculator:
    """稳定距离计算器 - 使用标准Scaler和更强的正则化"""

    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.scaler = None
        self.train_mean = None
        self.train_cov = None
        self.inv_cov = None
        self.is_fitted = False

    def fit(self, train_features):
        """拟合距离计算器"""
        detection_logger.log('distance_calculation', "使用 stable_mahalanobis 方法拟合距离计算器")

        # 使用 StandardScaler 而不是 RobustScaler
        self.scaler = StandardScaler().fit(train_features)
        train_scaled = self.scaler.transform(train_features)

        # 使用均值而不是中位数
        self.train_mean = np.mean(train_scaled, axis=0)

        # 计算标准协方差
        self.train_cov = np.cov(train_scaled, rowvar=False)

        # 更强的正则化
        reg_cov = self.train_cov + np.eye(self.train_cov.shape[0]) * 1e-2

        # 使用伪逆提高数值稳定性
        try:
            self.inv_cov = np.linalg.pinv(reg_cov)
            detection_logger.log('distance_calculation', "使用伪逆矩阵提高数值稳定性")
        except Exception as e:
            detection_logger.log('distance_calculation', f"伪逆失败，使用单位矩阵: {e}")
            self.inv_cov = np.eye(reg_cov.shape[0])

        self.is_fitted = True
        detection_logger.log('distance_calculation', f"稳定距离计算器拟合完成，特征维度: {train_features.shape[1]}")

    def calculate_distance(self, features):
        """计算特征到训练集的距离"""
        if not self.is_fitted:
            raise ValueError("距离计算器尚未拟合")

        features_scaled = self.scaler.transform(features)
        distances = []

        for i in range(len(features_scaled)):
            try:
                dist = mahalanobis(features_scaled[i], self.train_mean, self.inv_cov)
                distances.append(dist)
            except Exception as e:
                # 回退到欧氏距离
                dist = np.linalg.norm(features_scaled[i] - self.train_mean)
                distances.append(dist)

        return np.array(distances)

class ConsistentNormalizer:
    """一致的标准化器 - 基于训练集统计量，应用到所有数据集"""

    def __init__(self, method='robust'):
        self.method = method
        self.median = None
        self.iqr = None
        self.mean = None
        self.std = None
        self.is_fitted = False

    def fit(self, train_scores):
        """基于训练集拟合标准化器"""
        detection_logger.log('config', f"使用 {self.method} 方法拟合标准化器")

        if self.method == 'robust':
            self.median = np.median(train_scores)
            q75, q25 = np.percentile(train_scores, [75, 25])
            self.iqr = q75 - q25

            if self.iqr == 0:
                self.iqr = np.std(train_scores) if np.std(train_scores) > 0 else 1.0

            detection_logger.log('config', f"训练集统计量 - 中位数: {self.median:.4f}, IQR: {self.iqr:.4f}")

        elif self.method == 'standard':
            self.mean = np.mean(train_scores)
            self.std = np.std(train_scores)

            if self.std == 0:
                self.std = 1.0

            detection_logger.log('config', f"训练集统计量 - 均值: {self.mean:.4f}, 标准差: {self.std:.4f}")

        self.is_fitted = True
        detection_logger.log('config', "标准化器拟合完成")

    def transform(self, scores):
        """使用训练集统计量转换分数"""
        if not self.is_fitted:
            raise ValueError("标准化器尚未拟合")

        scores = np.array(scores)

        if self.method == 'robust':
            normalized = (scores - self.median) / self.iqr
        elif self.method == 'standard':
            normalized = (scores - self.mean) / self.std

        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1e10, neginf=-1e10)
        return normalized

    def fit_transform(self, train_scores):
        """拟合并转换训练集"""
        self.fit(train_scores)
        return self.transform(train_scores)

class RobustFaultLineCalculator:
    """基于去除极端值后取上界的稳健故障线计算器"""

    def __init__(self, method='robust_upper_bound', confidence_level=0.95,
                 remove_extreme_k=3, deviation_threshold=3.0, safety_margin=1.05):
        self.method = method
        self.confidence_level = confidence_level
        self.remove_extreme_k = remove_extreme_k
        self.deviation_threshold = deviation_threshold
        self.safety_margin = safety_margin

    def calculate(self, normal_distances, normalizer=None):
        """计算稳健故障线"""
        detection_logger.log('fault_line', f"使用 {self.method} 方法计算稳健故障线")
        detection_logger.log('fault_line', f"配置: 移除前{self.remove_extreme_k}个极端值, 偏离阈值={self.deviation_threshold}, 安全边界={self.safety_margin}")

        return self._single_dataset_with_extreme_removal(normal_distances, normalizer)

    def _single_dataset_with_extreme_removal(self, distances, normalizer):
        """单数据集稳健上界方法 - 去除极端值"""
        fault_line_result = self._calculate_with_extreme_removal(distances, normalizer)

        if isinstance(fault_line_result, tuple):
            fault_line = fault_line_result[0]
        else:
            fault_line = fault_line_result

        return fault_line, [fault_line]

    def _calculate_with_extreme_removal(self, distances, normalizer):
        """核心方法：去除极端值后取上界"""
        distances = np.array(distances)

        detection_logger.log('fault_line', "原始距离统计:")
        detection_logger.log('fault_line', f"  样本数: {len(distances)}")
        detection_logger.log('fault_line', f"  范围: [{distances.min():.4f}, {distances.max():.4f}]")
        detection_logger.log('fault_line', f"  中位数: {np.median(distances):.4f}")

        median = np.median(distances)
        mad = stats.median_abs_deviation(distances, scale='normal')

        detection_logger.log('fault_line', f"中心趋势: 中位数={median:.4f}, MAD={mad:.4f}")

        if len(distances) > self.remove_extreme_k:
            deviations = np.abs(distances - median) / mad
            extreme_indices = np.argsort(deviations)[-self.remove_extreme_k:]

            clean_mask = np.ones(len(distances), dtype=bool)
            clean_mask[extreme_indices] = False
            clean_distances = distances[clean_mask]

            detection_logger.log('fault_line', "极端值去除结果:")
            detection_logger.log('fault_line', f"  原始样本数: {len(distances)}")
            detection_logger.log('fault_line', f"  去除样本数: {self.remove_extreme_k}")
            detection_logger.log('fault_line', f"  剩余样本数: {len(clean_distances)}")

            if len(clean_distances) < 0.8 * len(distances):
                detection_logger.log('fault_line', f"警告: 去除过多样本({len(clean_distances)}/{len(distances)})，调整策略")
                threshold_mask = deviations <= self.deviation_threshold
                clean_distances = distances[threshold_mask]
                detection_logger.log('fault_line', f"使用偏离阈值{self.deviation_threshold}，剩余样本: {len(clean_distances)}")
        else:
            clean_distances = distances
            detection_logger.log('fault_line', f"样本数较少({len(distances)})，不进行极端值去除")

        if len(clean_distances) == 0:
            detection_logger.log('fault_line', "警告: 清洁数据为空，使用原始数据")
            clean_distances = distances

        clean_upper_bound = np.max(clean_distances)
        fault_line = clean_upper_bound * self.safety_margin

        if normalizer is not None:
            fault_line_normalized = normalizer.transform([fault_line])[0]
            detection_logger.log('fault_line', "清洁数据集统计:")
            detection_logger.log('fault_line', f"  范围: [{clean_distances.min():.4f}, {clean_distances.max():.4f}]")
            detection_logger.log('fault_line', f"  上界: {clean_upper_bound:.4f}")
            detection_logger.log('fault_line', f"原始故障线: {fault_line:.4f} (上界 × {self.safety_margin})")
            detection_logger.log('fault_line', f"标准化故障线: {fault_line_normalized:.4f}")
            return fault_line_normalized
        else:
            detection_logger.log('fault_line', "清洁数据集统计:")
            detection_logger.log('fault_line', f"  范围: [{clean_distances.min():.4f}, {clean_distances.max():.4f}]")
            detection_logger.log('fault_line', f"  上界: {clean_upper_bound:.4f}")
            detection_logger.log('fault_line', f"最终故障线: {fault_line:.4f} (上界 × {self.safety_margin})")
            return fault_line

def fit_scaler_on_features(model, dataloader, device, feature_type='global', win_size=WIN_SIZE, use_standard_scaler=True):
    """在特征上拟合Scaler（兼容4输出/7输出模型）"""
    model.eval()
    features = []

    model_name = getattr(model, "__class__", type(model)).__name__

    with torch.no_grad():
        for x, m, delta_t, _ in dataloader:
            x, m = x.to(device), m.to(device)
            delta_t = delta_t.to(device)

            z_mean, z_seq = extract_detection_features(
                model, x, m, delta_t, model_name=model_name
            )

            if feature_type == 'global':
                features.append(z_mean.cpu().numpy())
            elif feature_type == 'startup':
                startup_seq = z_seq[:, :win_size, :] if z_seq.shape[1] >= win_size else z_seq
                startup_mean = startup_seq.mean(dim=1).cpu().numpy()
                features.append(startup_mean)
            else:
                raise ValueError(f"未知 feature_type: {feature_type}")

    if not features:
        raise ValueError("No features extracted for scaler fitting.")

    Z = np.vstack(features).astype(np.float64)

    if use_standard_scaler:
        scaler = StandardScaler().fit(Z)
        Z_scaled = scaler.transform(Z)
        detection_logger.log('data_loading', f"拟合StandardScaler完成，特征维度: {Z.shape[1]}")
    else:
        scaler = RobustScaler().fit(Z)
        Z_scaled = scaler.transform(Z)
        detection_logger.log('data_loading', f"拟合RobustScaler完成，特征维度: {Z.shape[1]}")

    detection_logger.log('data_loading', f"标准化后特征范围: [{Z_scaled.min():.4f}, {Z_scaled.max():.4f}]")

    return scaler, Z

def find_denoised_abnormal_paths(data_dir):
    """查找所有增强异常样本"""
    paths = []
    for file in os.listdir(data_dir):
        if (file.startswith('denoised_aug1_sample_') or file.startswith('denoised_aug2_sample_')) and file.endswith('_abnormal.csv'):
            paths.append(os.path.join(data_dir, file))
    if not paths:
        raise FileNotFoundError("未找到匹配的 denoised_aug1_sample_*_abnormal.csv 或 denoised_aug2_sample_*_abnormal.csv 文件")
    return sorted(paths)

def find_aligned_abnormal_paths(data_dir):
    """查找原始异常数据文件"""
    paths = []
    for file in os.listdir(data_dir):
        if file.startswith('denoised_aligned_sample') and file.endswith('_abnormal.csv'):
            paths.append(os.path.join(data_dir, file))
    if not paths:
        raise FileNotFoundError("未找到匹配的原始异常数据文件")
    return sorted(paths)

def handle_df2m_length_adjustment(model_name, arr, target_len):
    """DF2M模型特殊处理：截断数组到目标长度"""
    if model_name == "PredictiveDF2M_Model" and len(arr) > target_len:
        detection_logger.log('data_loading', f"截断数组从 {len(arr)} 到 {target_len}")
        return arr[:target_len]
    return arr

def _load_model_and_verify_input_dim(model_path, model_name, data_dir, apply_poisson_sampling, base_lambda, exp_key, fold):
    """加载模型并验证输入维度 - 简化版本"""
    config_path = f"results/{exp_key}/data_splits/{model_name}_fold_{fold}_data_splits.pth"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到数据划分配置文件: {config_path}")

    saved_split = torch.load(config_path, map_location='cpu', weights_only=True)

    if 'test_paths' in saved_split:
        test_normal_paths = saved_split['test_paths']
        detection_logger.log('model_loading', "使用 'test_paths' 作为测试正常样本路径")
    elif 'test_normal_paths' in saved_split:
        test_normal_paths = saved_split['test_normal_paths']
        detection_logger.log('model_loading', "使用 'test_normal_paths' 作为测试正常样本路径")
    else:
        available_keys = list(saved_split.keys())
        raise KeyError(f"数据划分文件中没有找到测试样本路径。可用键: {available_keys}")

    input_dim = saved_split['input_dim']

    detection_logger.log('model_loading', f"模型 {model_name}:")
    detection_logger.log('model_loading', f"  - 训练时保存的input_dim: {input_dim}")
    detection_logger.log('model_loading', f"  - 实验配置: 泊松采样={apply_poisson_sampling}")

    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")

    if 'model_params' in ckpt:
        model_params = ckpt['model_params'].copy()
        input_dim = ckpt.get('input_dim', model_params.get('input_dim'))
    else:
        input_dim = ckpt.get('input_dim')
        model_params = get_model_params(model_name, input_dim)

    try:
        ds_tmp = TimeSeriesDataset(test_normal_paths, apply_poisson_sampling=apply_poisson_sampling, base_lambda=base_lambda)
        sample_x, _, _, _ = next(iter(DataLoader(
            ds_tmp, batch_size=1, shuffle=False, collate_fn=custom_collate
        )))
        detected_input_dim = sample_x.shape[-1]

        if detected_input_dim != input_dim:
            detection_logger.log('model_loading', f"保存的input_dim={input_dim}但数据集给出{detected_input_dim}，使用数据集值")
            input_dim = detected_input_dim
            model_params['input_dim'] = input_dim

    except Exception as e:
        detection_logger.log('model_loading', f"输入维度验证失败: {e}，使用保存的维度: {input_dim}")

    try:
        model = create_model(model_name, **model_params)

        if 'model_state_dict' in ckpt:
            model_state_dict = ckpt['model_state_dict']
            current_state_dict = model.state_dict()

            missing_keys = [k for k in current_state_dict.keys() if k not in model_state_dict]
            unexpected_keys = [k for k in model_state_dict.keys() if k not in current_state_dict]

            if missing_keys or unexpected_keys:
                detection_logger.log('model_loading', f"状态字典键不匹配 - 缺失: {missing_keys}, 意外: {unexpected_keys}")
                model_state_dict = {k: v for k, v in model_state_dict.items() if k in current_state_dict}

            model.load_state_dict(model_state_dict, strict=False)
            detection_logger.log('model_loading', f"成功加载模型状态: {model_name}")
        else:
            detection_logger.log('model_loading', "检查点中没有model_state_dict，使用随机初始化的模型")

        model = model.to(DEVICE)
        return model, input_dim, model_params, fold

    except Exception as e:
        raise RuntimeError(f"创建模型失败: {e}")

def _prepare_test_data(data_dir, abnormal_data_type, num_anomaly, apply_poisson_sampling, base_lambda, model_name, exp_key, fold=None):
    """准备测试数据 - 使用模型特定的划分文件"""
    config_path = f"results/{exp_key}/data_splits/{model_name}_fold_{fold}_data_splits.pth"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到数据划分配置文件: {config_path}")

    saved_split = torch.load(config_path, map_location='cpu', weights_only=True)

    if 'test_paths' in saved_split:
        test_normal_paths = saved_split['test_paths']
        detection_logger.log('data_loading', "使用 'test_paths' 作为测试正常样本路径")
    elif 'test_normal_paths' in saved_split:
        test_normal_paths = saved_split['test_normal_paths']
        detection_logger.log('data_loading', "使用 'test_normal_paths' 作为测试正常样本路径")
    else:
        available_keys = list(saved_split.keys())
        raise KeyError(f"数据划分文件中没有找到测试样本路径。可用键: {available_keys}")

    if abnormal_data_type == 'augmented_all':
        abnormal_paths = find_denoised_abnormal_paths(data_dir)
        test_abnormal_paths = abnormal_paths
        detection_logger.log('data_loading', f"使用所有增强异常样本 ({len(abnormal_paths)}个)")
    elif abnormal_data_type == 'original_6':
        abnormal_paths = find_aligned_abnormal_paths(data_dir)
        test_abnormal_paths = random.sample(abnormal_paths, min(num_anomaly, len(abnormal_paths)))
        detection_logger.log('data_loading', f"使用原始异常样本 ({len(test_abnormal_paths)}个)")
    else:
        raise ValueError(f"未知的异常数据类型: {abnormal_data_type}")

    all_paths = test_normal_paths + test_abnormal_paths
    labels = np.array([0] * len(test_normal_paths) + [1] * len(test_abnormal_paths))

    detection_logger.log('data_loading', f"总测试样本: {len(all_paths)} (正常: {len(test_normal_paths)}, 异常: {len(test_abnormal_paths)})")

    return all_paths, labels

def analyze_misclassified_samples(model_name, experiment_name, all_paths, labels, preds, scores, fault_line):
    """分析分类失败的样本"""
    detection_logger.log('misclassification', f"{model_name} 分类失败样本分析")

    false_positives = np.where((labels == 0) & (preds == 1))[0]
    false_negatives = np.where((labels == 1) & (preds == 0))[0]

    if len(false_positives) > 0:
        detection_logger.log('misclassification', f"假阳性样本 ({len(false_positives)}个):")
        for idx in false_positives:
            file_name = os.path.basename(all_paths[idx])
            score = scores[idx]
            detection_logger.log('misclassification', f"  文件: {file_name}")
            detection_logger.log('misclassification', f"    分数: {score:.4f} (超过故障线 {fault_line:.4f})")
            detection_logger.log('misclassification', f"    相对故障线倍数: {score/fault_line:.2f}x")

    if len(false_negatives) > 0:
        detection_logger.log('misclassification', f"假阴性样本 ({len(false_negatives)}个):")
        for idx in false_negatives:
            file_name = os.path.basename(all_paths[idx])
            score = scores[idx]
            detection_logger.log('misclassification', f"  文件: {file_name}")
            detection_logger.log('misclassification', f"    分数: {score:.4f} (低于故障线 {fault_line:.4f})")
            detection_logger.log('misclassification', f"    相对故障线倍数: {score/fault_line:.2f}x")

    if len(false_positives) == 0 and len(false_negatives) == 0:
        detection_logger.log('misclassification', "所有样本分类正确")

    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'false_positive_files': [os.path.basename(all_paths[i]) for i in false_positives],
        'false_negative_files': [os.path.basename(all_paths[i]) for i in false_negatives],
        'false_positive_scores': [scores[i] for i in false_positives],
        'false_negative_scores': [scores[i] for i in false_negatives]
    }

def save_all_scores_to_csv(model_name, experiment_name, train_scores, test_scores, labels, fault_line, output_dir='./score_results'):
    """保存所有分数到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)

    train_data = {
        'model': model_name,
        'experiment': experiment_name,
        'dataset_type': 'train',
        'score': train_scores,
        'label': 0,
        'is_anomaly': 0,
        'above_fault_line': (train_scores > fault_line).astype(int)
    }

    test_data = {
        'model': model_name,
        'experiment': experiment_name,
        'dataset_type': 'test',
        'score': test_scores,
        'label': labels,
        'is_anomaly': labels,
        'above_fault_line': (test_scores > fault_line).astype(int)
    }

    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    csv_path = os.path.join(output_dir, f'{model_name}_{experiment_name}_scores.csv')
    df_all.to_csv(csv_path, index=False)
    detection_logger.log('performance', f"所有分数已保存到: {csv_path}")

    return df_all

def plot_score_analysis(model_name, experiment_name, train_scores, test_scores, labels, fault_line,
                       clean_train_scores=None, output_dir='./score_plots'):
    """绘制分数分析图 - 仅保留距离分布直方图"""
    os.makedirs(output_dir, exist_ok=True)

    if clean_train_scores is not None:
        plot_train_scores = clean_train_scores
        detection_logger.log('plotting', f"使用清洁训练集数据绘图，样本数: {len(plot_train_scores)} (原始: {len(train_scores)})")
    else:
        plot_train_scores = train_scores
        detection_logger.log('plotting', f"使用完整训练集数据绘图，样本数: {len(plot_train_scores)}")

    predictions = (test_scores > fault_line).astype(int)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    normal_scores = test_scores[labels == 0]
    anomaly_scores = test_scores[labels == 1]

    all_scores_combined = np.concatenate([plot_train_scores, normal_scores, anomaly_scores])

    if len(all_scores_combined) > 1:
        q25, q75 = np.percentile(all_scores_combined, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            bin_width = 2 * iqr / (len(all_scores_combined) ** (1/3))
            bins = max(20, int((np.max(all_scores_combined) - np.min(all_scores_combined)) / bin_width))
        else:
            bins = 50
    else:
        bins = 50

    detection_logger.log('plotting', f"使用 {bins} 个bins进行直方图绘制")

    ax.hist(plot_train_scores, bins=bins, alpha=0.7, label='Train Normal', color='green', density=True)
    ax.hist(normal_scores, bins=bins, alpha=0.7, label='Test Normal', color='blue', density=True)
    ax.hist(anomaly_scores, bins=bins, alpha=0.7, label='Test Anomaly', color='red', density=True)
    ax.axvline(fault_line, color='black', linestyle='--', linewidth=2, label='Control Limit')

    if clean_train_scores is not None and len(clean_train_scores) < len(train_scores):
        removed_extremes = train_scores[~np.isin(train_scores, clean_train_scores)]
        if len(removed_extremes) > 0:
            ax.hist(removed_extremes, bins=max(10, bins//5), alpha=0.5, label='Randomness',
                    color='orange', density=True, hatch='//')


    ax.set_yscale('log')
    ax.set_xlabel('Standardized Distance')
    ax.set_ylabel('Log Density')
    ax.set_title('Distance Distribution Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)

    log_note = "Y-axis uses log scale to better visualize\ndistributions with extreme density values"
    ax.text(0.02, 0.98, log_note, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7), fontsize=9)



    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name}_{experiment_name}_score_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    detection_logger.log('plotting', f"Score analysis plot saved to: {plot_path}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fault_line': fault_line
    }

def detect_anomalies_unified(data_dir, model_path, model_name, abnormal_data_type='augmented_all',
                           apply_poisson_sampling=None, experiment_name=None, exp_key=None, fold=None):
    """统一的异常检测主函数"""
    detection_logger.set_experiment_context(experiment_name, model_name)

    config = DETECTION_CONFIG.copy()

    # 根据异常数据类型调整配置
    if abnormal_data_type == 'augmented_all':
        # 对于增强异常，标签可能不可靠，禁用过滤
        config['remove_test_extreme_k'] = 0
        detection_logger.log('config', "增强异常检测：禁用测试集过滤（标签可能不可靠）")

    batch_size = config['batch_size']
    base_lambda = 0.5

    if exp_key is None and experiment_name is not None:
        exp_key = experiment_name.split('_')[0]
        detection_logger.log('config', f"从experiment_name中提取exp_key: {exp_key}")

    if experiment_name is None:
        experiment_name = f"{model_name}_{abnormal_data_type}"

    detection_logger.log('general', f"开始检测 - 模型: {model_name}, 实验: {experiment_name}, exp_key: {exp_key}")

    validate_config_consistency(config)

    # 修改这行调用，添加 fold 参数
    model, input_dim, model_params, fold = _load_model_and_verify_input_dim(
        model_path, model_name, data_dir, apply_poisson_sampling, base_lambda, exp_key, fold
    )

    # 修改这行调用，添加 fold 参数
    all_paths, labels = _prepare_test_data(
        data_dir, abnormal_data_type, 6, apply_poisson_sampling, base_lambda, model_name, exp_key, fold
    )

    ds_test = TimeSeriesDataset(all_paths, apply_poisson_sampling=apply_poisson_sampling, base_lambda=base_lambda)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    config_path = f"results/{exp_key}/data_splits/{model_name}_fold_{fold}_data_splits.pth"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到数据划分配置文件: {config_path}")

    saved_split = torch.load(config_path, map_location='cpu', weights_only=True)

    if 'train_paths' in saved_split:
        train_paths = saved_split['train_paths']
        detection_logger.log('data_loading', "使用 'train_paths' 作为训练样本路径")
    else:
        available_keys = list(saved_split.keys())
        raise KeyError(f"数据划分文件中没有找到训练样本路径。可用键: {available_keys}")

    ds_norm = TimeSeriesDataset(train_paths, apply_poisson_sampling=apply_poisson_sampling, base_lambda=base_lambda)
    dl_norm = DataLoader(ds_norm, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    # 根据配置选择距离计算器
    if config['distance_method'] == 'stable_mahalanobis':
        distance_calculator = StableDistanceCalculator(
            confidence_level=config['confidence_level']
        )
    else:
        distance_calculator = RobustDistanceCalculator(
            robust_covariance=config['robust_covariance'],
            confidence_level=config['confidence_level']
        )

    fault_line_calculator = RobustFaultLineCalculator(
        method=config['fault_line_method'],
        confidence_level=config['confidence_level'],
        remove_extreme_k=config.get('remove_extreme_k', 3),
        deviation_threshold=config.get('deviation_threshold', 3.0),
        safety_margin=config.get('safety_margin', 1.05)
    )

    detection_logger.log('general', "拟合Scaler并提取训练集特征...")
    scaler_global, train_features_global = fit_scaler_on_features(
        model, dl_norm, DEVICE, feature_type='global', use_standard_scaler=True
    )

    detection_logger.log('general', "拟合距离计算器...")
    distance_calculator.fit(train_features_global)

    detection_logger.log('general', "计算训练集距离...")
    train_distances = distance_calculator.calculate_distance(train_features_global)

    detection_logger.log('performance', f"训练集正常样本原始分数范围: [{train_distances.min():.4f}, {train_distances.max():.4f}]")

    detection_logger.log('general', "创建一致的标准化器...")
    normalizer = ConsistentNormalizer(method='robust')
    train_distances_norm = normalizer.fit_transform(train_distances)

    detection_logger.log('performance', f"训练集标准化后范围: [{train_distances_norm.min():.4f}, {train_distances_norm.max():.4f}]")

    detection_logger.log('general', "计算稳健故障线...")
    fault_line_result = fault_line_calculator.calculate(
        train_distances,
        normalizer=normalizer
    )

    if isinstance(fault_line_result, tuple):
        fault_line, fold_fault_lines = fault_line_result
    else:
        fault_line = fault_line_result
        fold_fault_lines = [fault_line]

    detection_logger.log('fault_line', f"最终故障线: {fault_line:.4f}")

    detection_logger.log('general', "检测测试集...")
    start_time = time.time()

    test_distances = []
    z_test_list = []

    model.eval()
    model_name_runtime = getattr(model, "__class__", type(model)).__name__

    with torch.no_grad():
        for batch_idx, (x, m, delta_t, _) in enumerate(dl_test):
            x, m = x.to(DEVICE), m.to(DEVICE)
            delta_t = delta_t.to(DEVICE)

            z_mean, z_seq = extract_detection_features(
                model, x, m, delta_t, model_name=model_name_runtime
            )
            z_test_list.append(z_mean.cpu().numpy())

    z_test = np.vstack(z_test_list)
    test_distances = distance_calculator.calculate_distance(z_test)

    elapsed_time = time.time() - start_time

    test_distances = np.array(test_distances)

    target_len = len(labels)
    test_distances = handle_df2m_length_adjustment(model_name, test_distances, target_len)
    z_test = handle_df2m_length_adjustment(model_name, z_test, target_len)

    test_distances_norm = normalizer.transform(test_distances)

    # 🆕 新增：过滤测试集正常样本中的极端值
    remove_test_extreme_k = config.get('remove_test_extreme_k', 0)
    if remove_test_extreme_k > 0:
        test_distances_norm_filtered, labels_filtered, filter_mask = filter_test_extreme_values(
            test_distances_norm, labels, remove_test_extreme_k
        )

        # 更新路径和特征数组
        all_paths_filtered = [all_paths[i] for i in range(len(all_paths)) if filter_mask[i]]
        z_test_filtered = z_test[filter_mask] if len(z_test) == len(filter_mask) else z_test

        # 使用过滤后的数据
        test_distances_norm = test_distances_norm_filtered
        labels = labels_filtered
        all_paths = all_paths_filtered
        z_test = z_test_filtered

        detection_logger.log('performance',
                           f"测试集过滤后统计: {len(labels)}样本 (正常:{np.sum(labels==0)}, 异常:{np.sum(labels==1)})")

    normal_mask = (labels == 0)
    anomaly_mask = (labels == 1)

    if np.sum(normal_mask) > 0:
        detection_logger.log('performance', f"正常样本标准化后范围: [{test_distances_norm[normal_mask].min():.4f}, {test_distances_norm[normal_mask].max():.4f}]")
    if np.sum(anomaly_mask) > 0:
        detection_logger.log('performance', f"异常样本标准化后范围: [{test_distances_norm[anomaly_mask].min():.4f}, {test_distances_norm[anomaly_mask].max():.4f}]")

    preds_final = (test_distances_norm > fault_line).astype(int)

    misclassification_analysis = analyze_misclassified_samples(
        model_name, experiment_name, all_paths, labels, preds_final, test_distances_norm, fault_line
    )

    total_samples = len(all_paths)
    time_per_sample = elapsed_time / total_samples

    test_distances_norm = np.nan_to_num(test_distances_norm, nan=0.0, posinf=1e10, neginf=-1e10)

    try:
        acc = accuracy_score(labels, preds_final)
        prec = precision_score(labels, preds_final, zero_division=0)
        rec = recall_score(labels, preds_final, zero_division=0)
        f1 = f1_score(labels, preds_final, zero_division=0)

        # 🆕 添加F-beta分数（beta=2，更重视召回率）
        from sklearn.metrics import fbeta_score
        f_beta_2 = fbeta_score(labels, preds_final, beta=2, zero_division=0)
        f_beta_0_5 = fbeta_score(labels, preds_final, beta=0.5, zero_division=0)

        auc = roc_auc_score(labels, test_distances_norm)
        precision_vals, recall_vals, _ = precision_recall_curve(labels, test_distances_norm)
        pr_auc = calc_auc(recall_vals, precision_vals)
    except Exception as e:
        detection_logger.log('performance', f"计算指标失败: {e}")
        acc = prec = rec = f1 = f_beta_2 = f_beta_0_5 = auc = pr_auc = 0.5

    df_scores = save_all_scores_to_csv(
        model_name, experiment_name,
        train_distances_norm, test_distances_norm, labels, fault_line
    )

    clean_train_distances_norm = _get_clean_training_scores(train_distances, normalizer, fault_line_calculator)
    plot_metrics = plot_score_analysis(
        model_name, experiment_name,
        train_distances_norm, test_distances_norm, labels, fault_line,
        clean_train_scores=clean_train_distances_norm
    )

    latent_train_path = f"{model_name}_{exp_key}_train_latent.npy"
    latent_test_path = f"{model_name}_{exp_key}_test_latent.npy"

    np.save(latent_train_path, train_features_global)
    np.save(latent_test_path, z_test)

    detection_logger.log('general', "Latent特征已保存:")
    detection_logger.log('general', f"  训练集: {latent_train_path}")
    detection_logger.log('general', f"  测试集: {latent_test_path}")

    shapes_info.append((model_name, train_features_global.shape, z_test.shape))

    tp = np.sum((preds_final == 1) & (labels == 1))
    fp = np.sum((preds_final == 1) & (labels == 0))
    fn = np.sum((preds_final == 0) & (labels == 1))

    detection_logger.log('performance', f"检测结果 ({model_name} - {experiment_name}):")
    detection_logger.log('performance', f"总样本数: {total_samples} (正常:{np.sum(labels==0)}, 异常:{np.sum(labels==1)})")
    detection_logger.log('performance', f"总耗时: {elapsed_time:.4f}s, 单样本: {time_per_sample:.6f}s")
    detection_logger.log('performance', f"故障线: {fault_line:.4f} (基于训练集统计量标准化)")
    detection_logger.log('performance', f"距离方法: {config['distance_method']}")
    detection_logger.log('performance', f"故障线方法: {config['fault_line_method']}")
    detection_logger.log('performance', f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    detection_logger.log('performance', f"F-beta(2) (重召回): {f_beta_2:.4f}, F-beta(0.5) (重精确): {f_beta_0_5:.4f}")  # 新增
    detection_logger.log('performance', f"AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
    detection_logger.log('performance', f"TP: {tp}, FP: {fp}, FN: {fn}")
    detection_logger.log('performance', f"F-beta说明: F-beta(2) 更重视召回率（β=2），F-beta(0.5) 更重视精确率（β=0.5）")  # 新增说明

    if len(misclassification_analysis['false_negatives']) > 0:
        detection_logger.log('performance', f"未检测出的异常样本: {len(misclassification_analysis['false_negatives'])}个")
    else:
        detection_logger.log('performance', "成功检测出所有异常样本")

    return {
        'model_name': model_name,
        'experiment_name': experiment_name,
        'exp_key': exp_key,
        'abnormal_data_type': abnormal_data_type,
        'poisson_sampling': apply_poisson_sampling,
        'total_samples': total_samples,
        'normal_samples': int(np.sum(labels==0)),
        'abnormal_samples': int(np.sum(labels==1)),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'f_beta_2': float(f_beta_2),  # 🆕 添加
        'f_beta_0_5': float(f_beta_0_5),  # 🆕 添加
        'auc': float(auc),
        'pr_auc': float(pr_auc),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'fault_line': float(fault_line),
        'detection_time': float(elapsed_time),
        'time_per_sample': float(time_per_sample),
        'distance_method': config['distance_method'],
        'fault_line_method': config['fault_line_method'],
        'remove_test_extreme_k': remove_test_extreme_k,  # 🆕 添加过滤参数
        'plot_metrics': plot_metrics,
        'score_dataframe': df_scores,
        'misclassification_analysis': misclassification_analysis
    }

def detect_all_models_unified(data_dir, exp_key, model_names=None, abnormal_data_type='augmented_all'):
    """批量检测多个模型 - 统一配置"""
    from config import ACTIVE_MODELS, get_experiment_models

    if model_names is None:
        model_names = ACTIVE_MODELS

    experiment_models = get_experiment_models(exp_key)
    exp_config = EXPERIMENTS.get(exp_key, {})
    apply_poisson_sampling = exp_config.get('poisson', False)

    detection_results = {}

    for model_name in model_names:
        if model_name not in experiment_models:
            detection_logger.log('general', f"跳过 {model_name}，不在实验 {exp_key} 的配置中")
            continue

        model_save_name = experiment_models[model_name]
        model_path = os.path.join('./saved_models', model_save_name)

        if not os.path.exists(model_path):
            detection_logger.log('model_loading', f"模型文件不存在: {model_path}")
            continue

        detection_logger.log('general', f"开始检测 {model_name}")

        try:
            result = detect_anomalies_unified(
                data_dir=data_dir,
                model_path=model_path,
                model_name=model_name,
                abnormal_data_type=abnormal_data_type,
                apply_poisson_sampling=apply_poisson_sampling,
                experiment_name=f"{exp_key}_{model_name}",
                exp_key=exp_key,
                fold=None  # 对于非K折版本，传递 None
            )
            detection_results[model_name] = result
            detection_logger.log('general', f"{model_name} 检测完成")

        except Exception as e:
            detection_logger.log('general', f"检测 {model_name} 失败: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    summary_results = generate_unified_summary(detection_results, exp_key)
    detection_logger.save_all_logs()

    return {
        'detection_summary': detection_results,
        'experiment_key': exp_key,
        'abnormal_data_type': abnormal_data_type,
        'total_models_tested': len(detection_results),
        'summary_results': summary_results
    }

def generate_unified_summary(detection_results, exp_key):
    """生成统一的汇总报告"""
    detection_logger.log('performance', f"开始生成实验 {exp_key} 的汇总报告")

    print(f"\n{'='*80}")
    print(f"实验 {exp_key} 汇总报告")
    print(f"{'='*80}")

    sorted_results = sorted(
        detection_results.values(),
        key=lambda x: x.get('f1', 0),
        reverse=True
    )

    # 扩展表头，添加 F-beta 指标
    print(f"{'模型':<25} {'距离方法':<15} {'故障线方法':<15} {'测试过滤K':<10} "
          f"{'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} "
          f"{'F-beta(2)':<10} {'F-beta(0.5)':<12} {'AUC':<10} {'Time/Sample(s)':<15}")
    print("-" * 140)  # 增加分隔线长度

    summary_data = []

    for result in sorted_results:
        model_name = result['model_name']
        distance_method = result['distance_method']
        fault_line_method = result['fault_line_method']
        test_filter = f"K={result.get('remove_test_extreme_k', 0)}"  # 🆕 测试过滤信息
        fault_line = f"{result['fault_line']:.4f}"
        accuracy = f"{result['accuracy']:.4f}"
        precision = f"{result['precision']:.4f}"
        recall = f"{result['recall']:.4f}"
        f1_score = f"{result['f1']:.4f}"
        f_beta_2 = f"{result.get('f_beta_2', 0):.4f}"  # 新增
        f_beta_0_5 = f"{result.get('f_beta_0_5', 0):.4f}"  # 新增
        auc = f"{result['auc']:.4f}"
        time_per_sample = f"{result['time_per_sample']:.6f}"

        print(f"{model_name:<25} {distance_method:<15} {fault_line_method:<15} {test_filter:<10} "
              f"{accuracy:<10} {precision:<10} {recall:<10} {f1_score:<10} "
              f"{f_beta_2:<10} {f_beta_0_5:<12} {auc:<10} {time_per_sample:<15}")

        summary_data.append({
            '模型': model_name,
            '距离方法': distance_method,
            '故障线方法': fault_line_method,
            '测试过滤K值': result.get('remove_test_extreme_k', 0),  # 🆕
            '故障线': result['fault_line'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1': result['f1'],
            'F-beta(2)': result.get('f_beta_2', 0),  # 新增
            'F-beta(0.5)': result.get('f_beta_0_5', 0),  # 新增
            'AUC': result['auc'],
            'TimePerSample': result['time_per_sample'],
            '总样本数': result['total_samples'],
            '正常样本': result['normal_samples'],
            '异常样本': result['abnormal_samples'],
            'TP': result['tp'],
            'FP': result['fp'],
            'FN': result['fn'],
            '实验键': exp_key
        })

    if summary_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_df = pd.DataFrame(summary_data)
        summary_df_sorted = summary_df.sort_values('F1', ascending=False)

        csv_path = os.path.join('./detection_logs', f'summary_report_{exp_key}_{timestamp}.csv')
        summary_df_sorted.to_csv(csv_path, index=False, encoding='utf-8')
        detection_logger.log('performance', f"汇总报告已保存: {csv_path}")

    if shapes_info:
        print(f"\n{'='*40}")
        print("Latent Shapes 汇总")
        print(f"{'='*40}")
        print(f"{'模型':<25} {'训练集Shape':<20} {'测试集Shape':<20}")
        print("-" * 65)
        for name, train_shape, test_shape in shapes_info:
            print(f"{name:<25} {str(train_shape):<20} {str(test_shape):<20}")

        shapes_data = []
        for name, train_shape, test_shape in shapes_info:
            shapes_data.append({
                '模型': name,
                '训练集Shape': str(train_shape),
                '测试集Shape': str(test_shape),
                '实验键': exp_key
            })

        shapes_df = pd.DataFrame(shapes_data)
        shapes_path = os.path.join('./detection_logs', f'shapes_info_{exp_key}_{timestamp}.csv')
        shapes_df.to_csv(shapes_path, index=False, encoding='utf-8')
        detection_logger.log('general', f"Shapes信息已保存: {shapes_path}")

    return sorted_results

def find_available_runs(exp_key, model_names, pattern='fold'):
    """自动查找可用的运行版本"""
    available_runs = set()

    for model_name in model_names:
        # 使用模型名称映射获取文件前缀
        model_prefix = MODEL_NAME_MAPPING.get(model_name, model_name.lower())

        if pattern == 'fold':
            # 查找K折模型文件
            pattern_str = os.path.join('./saved_models', f'{model_prefix}_{exp_key}_{pattern}*.pth')
        else:
            # 查找普通运行文件
            pattern_str = os.path.join('./saved_models', f'{model_prefix}_{exp_key}_{pattern}*.pth')

        matching_files = glob.glob(pattern_str)
        if pattern == 'fold':
            alt_pattern = os.path.join('./saved_models', f'{model_prefix}_{exp_key}_{pattern}_*.pth')
            matching_files.extend(glob.glob(alt_pattern))

        for file_path in matching_files:
            filename = os.path.basename(file_path)
            # 提取运行编号
            if pattern == 'fold':
                run_match = re.search(rf'{pattern}_?(\d+)\.pth', filename)
            else:
                run_match = re.search(rf'{pattern}(\d+)\.pth', filename)

            if run_match:
                run_id = int(run_match.group(1))
                available_runs.add(run_id)

    # 如果没有找到特定运行的文件，检查基础文件
    if not available_runs:
        detection_logger.log('general', "未找到特定运行文件，检查基础文件")
        for model_name in model_names:
            model_prefix = MODEL_NAME_MAPPING.get(model_name, model_name.lower())
            base_file = os.path.join('./saved_models', f'{model_prefix}_{exp_key}.pth')
            if os.path.exists(base_file):
                detection_logger.log('general', f"找到基础文件: {base_file}")
                return [1]  # 默认使用运行1

    return sorted(available_runs)

def detect_experiment_runs(data_dir, exp_key, model_names=None, abnormal_data_type='augmented_all', runs=None, pattern='fold'):
    """检测并运行实验的不同运行版本，计算指标统计"""
    from config import ACTIVE_MODELS, get_experiment_models

    if model_names is None:
        model_names = ACTIVE_MODELS

    if runs is None:
        runs = find_available_runs(exp_key, model_names, pattern)
        detection_logger.log('general', f"自动检测到可用的{pattern}版本: {runs}")

    experiment_models = get_experiment_models(exp_key)
    exp_config = EXPERIMENTS.get(exp_key, {})
    apply_poisson_sampling = exp_config.get('poisson', False)

    # 存储所有运行的结果
    all_run_results = {}

    for run_id in runs:
        detection_logger.log('general', f"开始运行 {exp_key} 的 {pattern}{run_id}")
        run_results = {}

        for model_name in model_names:
            if model_name not in experiment_models:
                detection_logger.log('general', f"跳过 {model_name}，不在实验 {exp_key} 的配置中")
                continue

            # 🔧 修复：使用模型名称映射直接构建K折模型文件名
            model_prefix = MODEL_NAME_MAPPING.get(model_name, model_name.lower())
            model_run_name = f"{model_prefix}_{exp_key}_{pattern}_{run_id}.pth"
            model_path = os.path.join('./saved_models', model_run_name)

            # 如果特定运行的文件不存在，尝试基础文件名
            if not os.path.exists(model_path):
                detection_logger.log('model_loading', f"{pattern} {run_id} 的模型文件不存在: {model_path}，尝试基础文件")
                # 也尝试基础文件名
                base_model_name = f"{model_prefix}_{exp_key}.pth"
                base_model_path = os.path.join('./saved_models', base_model_name)
                if os.path.exists(base_model_path):
                    model_path = base_model_path
                    detection_logger.log('model_loading', f"使用基础模型文件: {base_model_path}")
                else:
                    detection_logger.log('model_loading', f"模型文件不存在: {model_path} 和 {base_model_path}")
                    continue

            detection_logger.log('general', f"开始检测 {model_name} - {pattern}{run_id}")

            try:
                result = detect_anomalies_unified(
                    data_dir=data_dir,
                    model_path=model_path,
                    model_name=model_name,
                    abnormal_data_type=abnormal_data_type,
                    apply_poisson_sampling=apply_poisson_sampling,
                    experiment_name=f"{exp_key}_{model_name}_{pattern}{run_id}",
                    exp_key=exp_key,
                    fold=run_id  # 传递 fold 参数
                )
                run_results[model_name] = result
                detection_logger.log('general', f"{model_name} - {pattern}{run_id} 检测完成")

            except Exception as e:
                detection_logger.log('general', f"检测 {model_name} - {pattern}{run_id} 失败: {e}", "ERROR")
                import traceback
                traceback.print_exc()

        all_run_results[f'{pattern}{run_id}'] = run_results

    # 计算统计信息
    summary_stats = calculate_run_statistics(all_run_results, exp_key)

    # 生成详细报告
    detailed_report = generate_detailed_run_report(all_run_results, exp_key, pattern)

    detection_logger.save_all_logs()

    return {
        'all_run_results': all_run_results,
        'summary_statistics': summary_stats,
        'detailed_report': detailed_report,
        'experiment_key': exp_key,
        'runs_tested': runs,
        'pattern': pattern
    }

def calculate_run_statistics(all_run_results, exp_key):
    """计算多个运行的统计信息（均值和标准差）"""
    detection_logger.log('performance', f"开始计算实验 {exp_key} 的运行统计信息")

    # 收集所有模型的指标
    model_metrics = {}

    for run_name, run_results in all_run_results.items():
        for model_name, result in run_results.items():
            if model_name not in model_metrics:
                model_metrics[model_name] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'f_beta_2': [],  # 新增
                    'f_beta_0_5': [],  # 新增
                    'auc': [],
                    'pr_auc': [],
                    'fault_line': [],
                    'time_per_sample': []
                }

            model_metrics[model_name]['accuracy'].append(result['accuracy'])
            model_metrics[model_name]['precision'].append(result['precision'])
            model_metrics[model_name]['recall'].append(result['recall'])
            model_metrics[model_name]['f1'].append(result['f1'])
            model_metrics[model_name]['f_beta_2'].append(result.get('f_beta_2', 0))  # 新增
            model_metrics[model_name]['f_beta_0_5'].append(result.get('f_beta_0_5', 0))  # 新增
            model_metrics[model_name]['auc'].append(result['auc'])
            model_metrics[model_name]['pr_auc'].append(result.get('pr_auc', 0))
            model_metrics[model_name]['fault_line'].append(result['fault_line'])
            model_metrics[model_name]['time_per_sample'].append(result['time_per_sample'])

    # 计算统计量
    statistics = {}
    for model_name, metrics in model_metrics.items():
        statistics[model_name] = {}
        for metric_name, values in metrics.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                statistics[model_name][metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'values': values,
                    'n_runs': len(values)
                }
            else:
                statistics[model_name][metric_name] = {
                    'mean': 0,
                    'std': 0,
                    'values': [],
                    'n_runs': 0
                }

    return statistics

def generate_detailed_run_report(all_run_results, exp_key, pattern='fold'):
    """生成详细的运行报告，包含每个指标的均值和标准差"""
    statistics = calculate_run_statistics(all_run_results, exp_key)

    print(f"\n{'='*120}")
    print(f"实验 {exp_key} - {pattern}统计报告")
    print(f"{'='*120}")

    # 按 F-beta(2) 分数均值排序（更重视召回率）
    sorted_models = sorted(
        statistics.items(),
        key=lambda x: x[1].get('f_beta_2', {}).get('mean', 0),
        reverse=True
    )

    # 打印每个模型的详细统计信息
    for model_name, stats in sorted_models:
        print(f"\n📊 模型: {model_name}")
        print("-" * 100)

        # 主要指标 - 添加 F-beta 指标
        main_metrics = ['accuracy', 'precision', 'recall', 'f1', 'f_beta_2', 'f_beta_0_5', 'auc']
        for metric in main_metrics:
            if metric in stats:
                mean_val = stats[metric]['mean']
                std_val = stats[metric]['std']
                # 为 F-beta 指标添加描述
                metric_name = metric.upper()
                if metric == 'f_beta_2':
                    metric_name = 'F-beta(2) (重召回)'
                elif metric == 'f_beta_0_5':
                    metric_name = 'F-beta(0.5) (重精确)'
                print(f"  {metric_name:<20}: {mean_val:.4f} ± {std_val:.4f}")

        # 故障线信息
        if 'fault_line' in stats:
            fault_mean = stats['fault_line']['mean']
            fault_std = stats['fault_line']['std']
            print(f"  故障线              : {fault_mean:.4f} ± {fault_std:.4f}")

        # 额外信息
        print(f"  {pattern}次数            : {stats['f1']['n_runs']}")

        if 'time_per_sample' in stats:
            time_mean = stats['time_per_sample']['mean']
            time_std = stats['time_per_sample']['std']
            print(f"  单样本耗时          : {time_mean:.6f} ± {time_std:.6f} s")

    # 生成汇总表格 - 添加 F-beta 列
    print(f"\n{'='*120}")
    print("汇总表格 - 按 F-beta(2) 分数排序（更重视召回率）")
    print(f"{'='*120}")
    print(f"{'模型':<25} {'运行次数':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} "
          f"{'F-beta(2)':<12} {'F-beta(0.5)':<12} {'AUC':<12} {'故障线':<12}")
    print("-" * 140)

    for model_name, stats in sorted_models:
        n_runs = stats['f1']['n_runs']

        # 获取各个指标的均值和标准差
        acc_mean = stats['accuracy']['mean']
        acc_std = stats['accuracy']['std']
        prec_mean = stats['precision']['mean']
        prec_std = stats['precision']['std']
        rec_mean = stats['recall']['mean']
        rec_std = stats['recall']['std']
        f1_mean = stats['f1']['mean']
        f1_std = stats['f1']['std']
        f_beta_2_mean = stats.get('f_beta_2', {}).get('mean', 0)
        f_beta_2_std = stats.get('f_beta_2', {}).get('std', 0)
        f_beta_0_5_mean = stats.get('f_beta_0_5', {}).get('mean', 0)
        f_beta_0_5_std = stats.get('f_beta_0_5', {}).get('std', 0)
        auc_mean = stats['auc']['mean']
        auc_std = stats['auc']['std']
        fault_mean = stats['fault_line']['mean']
        fault_std = stats['fault_line']['std']

        print(f"{model_name:<25} {n_runs:<8} {acc_mean:.4f}±{acc_std:.4f} {prec_mean:.4f}±{prec_std:.4f} "
              f"{rec_mean:.4f}±{rec_std:.4f} {f1_mean:.4f}±{f1_std:.4f} "
              f"{f_beta_2_mean:.4f}±{f_beta_2_std:.4f} {f_beta_0_5_mean:.4f}±{f_beta_0_5_std:.4f} "
              f"{auc_mean:.4f}±{auc_std:.4f} {fault_mean:.4f}±{fault_std:.4f}")

    # 保存详细报告到CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_data = []

    for model_name, stats in statistics.items():
        row = {
            '模型': model_name,
            f'{pattern}次数': stats['f1']['n_runs']
        }

        # 添加所有指标
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'f_beta_2', 'f_beta_0_5', 'auc', 'pr_auc']:
            if metric in stats:
                row[f'{metric}_均值'] = stats[metric]['mean']
                row[f'{metric}_标准差'] = stats[metric]['std']

        if 'fault_line' in stats:
            row['故障线_均值'] = stats['fault_line']['mean']
            row['故障线_标准差'] = stats['fault_line']['std']

        if 'time_per_sample' in stats:
            row['单样本耗时_均值'] = stats['time_per_sample']['mean']
            row['单样本耗时_标准差'] = stats['time_per_sample']['std']

        report_data.append(row)

    if report_data:
        df_report = pd.DataFrame(report_data)
        # 按 F-beta(2) 分数排序
        df_report = df_report.sort_values('f_beta_2_均值', ascending=False)

        csv_path = os.path.join('./detection_logs', f'{pattern}_statistics_{exp_key}_{timestamp}.csv')
        df_report.to_csv(csv_path, index=False, encoding='utf-8')
        detection_logger.log('performance', f"运行统计报告已保存: {csv_path}")

        # 保存可读版本
        readable_columns = {
            '模型': '模型',
            f'{pattern}次数': f'{pattern}次数',
            'accuracy_均值': '准确率-均值',
            'accuracy_标准差': '准确率-标准差',
            'precision_均值': '精确率-均值',
            'precision_标准差': '精确率-标准差',
            'recall_均值': '召回率-均值',
            'recall_标准差': '召回率-标准差',
            'f1_均值': 'F1分数-均值',
            'f1_标准差': 'F1分数-标准差',
            'f_beta_2_均值': 'F-beta(2)-均值（重召回）',  # 新增
            'f_beta_2_标准差': 'F-beta(2)-标准差',  # 新增
            'f_beta_0_5_均值': 'F-beta(0.5)-均值（重精确）',  # 新增
            'f_beta_0_5_标准差': 'F-beta(0.5)-标准差',  # 新增
            'auc_均值': 'AUC-均值',
            'auc_标准差': 'AUC-标准差',
            '故障线_均值': '故障线-均值',
            '故障线_标准差': '故障线-标准差'
        }

        # 只包含实际存在的列
        available_columns = [col for col in readable_columns.keys() if col in df_report.columns]
        readable_df = df_report[available_columns].copy()
        readable_df.rename(columns={col: readable_columns[col] for col in available_columns}, inplace=True)

        # 格式化数值
        for col in readable_df.columns:
            if '均值' in col or '标准差' in col:
                readable_df[col] = readable_df[col].apply(lambda x: f"{x:.4f}")

        readable_path = os.path.join('./detection_logs', f'{pattern}_statistics_readable_{exp_key}_{timestamp}.csv')
        readable_df.to_csv(readable_path, index=False, encoding='utf-8')
        detection_logger.log('performance', f"可读版运行统计报告已保存: {readable_path}")

    return statistics

# 主执行函数
if __name__ == "__main__":
    data_dir = "./origin_samples/preprocess_data"

    # 自动检测可用的K折版本
    from config import ACTIVE_MODELS
    available_folds = find_available_runs('exp1', ACTIVE_MODELS, pattern='fold')
    detection_logger.log('general', f"检测到可用的K折版本: {available_folds}")

    # 运行exp1实验的所有可用K折版本
    if available_folds:
        detection_logger.log('general', f"开始运行 exp1 的所有K折版本: {available_folds}")
        exp1_fold_results = detect_experiment_runs(
            data_dir=data_dir,
            exp_key='exp1',
            abnormal_data_type='augmented_all',
            #model_names=['PredictiveLatentODE_Model'],  # 只指定这个模型
            runs=available_folds,
            pattern='fold'
        )

        # 打印最终总结
        print(f"\n🎯 实验完成总结:")
        print(f"实验: exp1")
        print(f"测试的K折版本: {available_folds}")
        print(f"总模型数量: {len(exp1_fold_results['summary_statistics'])}")
        print(f"详细报告已保存到 detection_logs 目录")
    else:
        detection_logger.log('general', "未找到可用的K折版本，使用单模型检测")
        # 回退到原来的单模型检测
        exp1_results = detect_all_models_unified(
            data_dir=data_dir,
            exp_key='exp1',
            #model_names=['PredictiveLatentODE_Model'],  # 只指定这个模型
            abnormal_data_type='augmented_all'
        )

    detection_logger.log('general', "所有实验完成！")
