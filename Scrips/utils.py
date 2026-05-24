# utils.py

import json
import logging
import os
from datetime import datetime

import numpy as np
import torch


# ========================================
# 泊松采样相关函数
# ========================================
def generate_inhomogeneous_poisson_times(t_max, lambda_func, max_iter=10000, min_interval=1):
    times = []
    t = 0.0

    t_grid = np.linspace(0, t_max, 100)
    try:
        M = max(lambda_func(tt) for tt in t_grid)
    except Exception as e:
        raise ValueError(f"lambda_func 执行失败: {e}")

    if M <= 0:
        raise ValueError("lambda_func 必须返回正值")

    for _ in range(max_iter):
        w = np.random.exponential(1.0 / M)
        t = t + w
        if t > t_max:
            break

        if np.random.uniform(0, 1) <= lambda_func(t) / M:
            if not times or (t - times[-1]) >= min_interval:
                times.append(t)

    return np.array(times, dtype=np.float32)


def poisson_sampling_series_segmented(data, lambda_func, min_interval=0.05, t_max=None):
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.asarray(data, dtype=np.float32)

    n_total_steps, feat_dim = data_np.shape

    if t_max is None:
        t_max = float(n_total_steps)

    sampled_continuous_times = generate_inhomogeneous_poisson_times(
        t_max=t_max,
        lambda_func=lambda_func,
        min_interval=min_interval,
    )

    sampled_indices = np.round(sampled_continuous_times).astype(int)
    sampled_indices = np.clip(sampled_indices, 0, n_total_steps - 1)
    sampled_indices = np.unique(sampled_indices)
    sampled_indices = np.sort(sampled_indices)

    if sampled_indices.size == 0:
        sampled_indices = np.array([0], dtype=int)

    T_prime = sampled_indices.size
    sampled_data = data_np[sampled_indices]

    delta_t = np.zeros(T_prime, dtype=np.float32)
    delta_t[0] = sampled_indices[0] + 1.0
    for i in range(1, T_prime):
        delta_t[i] = sampled_indices[i] - sampled_indices[i - 1]

    mask = np.ones((T_prime, feat_dim), dtype=np.float32)

    sampled_data = torch.tensor(sampled_data.tolist(), dtype=torch.float32)
    delta_t = torch.tensor(delta_t.tolist(), dtype=torch.float32)
    mask = torch.tensor(mask.tolist(), dtype=torch.float32)

    print(
        f"[POISSON DEBUG] 原始T={n_total_steps} → "
        f"采样后T={T_prime} (保留 {T_prime / n_total_steps:.1%})"
    )

    return sampled_data, delta_t, mask


# ========================================
# 日志与配置
# ========================================
def setup_experiment_logging(exp_key, log_dir='./logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'experiment_{exp_key}_{timestamp}.log')

    logger = logging.getLogger(f'exp_{exp_key}')
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def save_experiment_config(config_dict, exp_key, save_dir='./experiment_configs'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    config_file = os.path.join(save_dir, f'config_{exp_key}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)

    print(f"[UTILS] 配置已保存: {config_file}")
    return config_file


# ========================================
# 模型前向适配
# ========================================
def safe_model_forward(model, x, mask=None, delta_t=None, n_samples=1):
    import inspect

    sig = inspect.signature(model.forward)
    kwargs = {'x': x}

    if 'mask' in sig.parameters:
        kwargs['mask'] = mask
    if 'delta_t' in sig.parameters:
        kwargs['delta_t'] = delta_t
    if 'delta_ts' in sig.parameters and delta_t is not None:
        kwargs['delta_ts'] = delta_t
    if 'n_samples' in sig.parameters:
        kwargs['n_samples'] = n_samples

    outputs = model(**kwargs)

    if not isinstance(outputs, tuple):
        raise TypeError("模型 forward 输出必须是 tuple")

    if len(outputs) == 4:
        recon, pred, z_mean, z_seq = outputs
        spike_seq = None
        h_seq = None
        return recon, pred, z_mean, z_seq, spike_seq, h_seq

    if len(outputs) == 6:
        recon, pred, z_mean, z_seq, spike_seq, h_seq = outputs
        return recon, pred, z_mean, z_seq, spike_seq, h_seq

    if len(outputs) == 7:
        recon, pred, z_mean, z_seq, _mem_reset_seq, seq6, h_seq = outputs
        return recon, pred, z_mean, z_seq, seq6, h_seq

    raise ValueError(f"当前系统仅兼容模型 forward 返回 4、6 或 7 个值，但收到 {len(outputs)} 个值")


# ========================================
# 模型相关函数
# ========================================
def load_model_checkpoint(model_path, model_name, device='cpu'):
    from model_factory import create_model

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_params' in checkpoint:
        model_params = checkpoint['model_params']
        input_dim = checkpoint.get('input_dim', model_params.get('input_dim'))
    else:
        from config import get_model_params
        input_dim = checkpoint.get('input_dim')
        if input_dim is None:
            raise ValueError("检查点中未找到input_dim信息")
        model_params = get_model_params(model_name, input_dim)

    clean_model_params = {k: v for k, v in model_params.items() if k != 'input_dim'}

    model = create_model(model_name, input_dim, **clean_model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"[UTILS] 成功加载模型: {model_path}")
    return model, checkpoint


def print_model_summary(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n=== {model_name} 模型摘要 ===")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {total_params - trainable_params:,}")

    if total_params < 10_000_000:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,} 参数")

    return total_params, trainable_params


def validate_model_compatibility(model_name, input_dim):
    try:
        from config import get_model_params
        from model_factory import create_model

        model_params = get_model_params(model_name, input_dim)
        clean_model_params = {k: v for k, v in model_params.items() if k != 'input_dim'}
        model = create_model(model_name, input_dim, **clean_model_params)

        batch_size, seq_len = 2, 10
        test_input = torch.randn(batch_size, seq_len, input_dim)
        test_mask = torch.ones(batch_size, seq_len, input_dim)
        test_delta_t = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            recon, pred, z_mean, z_seq, spike_seq, h_seq = safe_model_forward(
                model,
                test_input,
                mask=test_mask,
                delta_t=test_delta_t,
                n_samples=1,
            )

        print(f"[OK] {model_name} 兼容性验证通过")
        print(f"   输入: {test_input.shape} -> 输出: {recon.shape}")
        print(f"   z_mean: {z_mean.shape}")
        if z_seq is not None:
            print(f"   z_seq: {z_seq.shape}")
        if spike_seq is not None:
            print(f"   spike_seq: {spike_seq.shape}")
        if h_seq is not None:
            print(f"   h_seq: {h_seq.shape}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"[ERROR] {model_name} 兼容性验证失败: {e}")
        return False


# ========================================
# 结果处理
# ========================================
def setup_results_directory(exp_key, base_dir='./results'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f'{exp_key}_{timestamp}')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    subdirs = ['models', 'plots', 'metrics', 'logs']
    for subdir in subdirs:
        subdir_path = os.path.join(results_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

    print(f"[UTILS] 结果目录已创建: {results_dir}")
    return results_dir


def save_training_metrics(metrics_dict, exp_key, model_name, results_dir):
    metrics_file = os.path.join(results_dir, 'metrics', f'{model_name}_{exp_key}_metrics.json')

    serializable_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            serializable_metrics[key] = value
        elif isinstance(value, (list, tuple)):
            serializable_metrics[key] = [float(x) if isinstance(x, (int, float)) else str(x) for x in value]
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            serializable_metrics[key] = value.detach().cpu().tolist()
        else:
            serializable_metrics[key] = str(value)

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

    print(f"[UTILS] 训练指标已保存: {metrics_file}")
    return metrics_file
