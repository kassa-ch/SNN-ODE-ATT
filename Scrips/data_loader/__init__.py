#data_loader

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import poisson_sampling_series_segmented


def _read_csv_array(path: str) -> np.ndarray:
    """Read numeric wafer CSVs without pandas.

    The first row contains sensor column names, so skip it to preserve the
    original ``pandas.read_csv(...).values`` behavior.
    """
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _kfold_indices(n_samples: int, n_splits: int, random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    splits = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        splits.append((train_idx, val_idx))
        current = stop
    return splits


class TimeSeriesDataset(Dataset):
    """Load CSV time series samples for the existing training pipeline.

    Each item keeps the current return contract unchanged:
    ``(x, mask, delta_t, file_path)``.
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        apply_poisson_sampling: bool = False,
        base_lambda: float = 0.8,
        fixed_random_offset: bool = False,
    ) -> None:
        if not file_paths:
            raise ValueError("file_paths must contain at least one CSV path.")
        if base_lambda <= 0:
            raise ValueError("base_lambda must be positive.")

        self.file_paths = list(file_paths)
        self.apply_poisson_sampling = apply_poisson_sampling
        self.base_lambda = base_lambda
        self.fixed_random_offset = fixed_random_offset

        if self.fixed_random_offset:
            self.random_offsets = {
                path: np.random.uniform(-0.2 * base_lambda, 0.2 * base_lambda)
                for path in file_paths
            }
        else:
            self.random_offsets = None

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        try:
            file_path = self.file_paths[idx]
            data = _read_csv_array(file_path)
            x = torch.tensor(data, dtype=torch.float32)

            # 标准化
            x_mean = x.mean(dim=0, keepdim=True)
            x_std = x.std(dim=0, keepdim=True) + 1e-6
            x = (x - x_mean) / x_std

            t_max = x.shape[0]

            if self.random_offsets is not None:
                random_offset = self.random_offsets[file_path]
            else:
                random_offset = np.random.uniform(-0.2 * self.base_lambda, 0.2 * self.base_lambda)

            lambda_func = lambda t: self.base_lambda + random_offset + 0.4 * np.sin(t / 5)

            delta_t = None
            mask = None

            if self.apply_poisson_sampling:
                sampled_x, delta_t, mask = poisson_sampling_series_segmented(
                    data=x.numpy(),
                    lambda_func=lambda_func,
                    min_interval=0.05,
                    t_max=t_max
                )
                x = sampled_x if isinstance(sampled_x, torch.Tensor) else torch.tensor(sampled_x.tolist(), dtype=torch.float32)
            else:
                delta_t = torch.ones(x.shape[0], dtype=torch.float32)
                mask = torch.ones_like(x)

            if not isinstance(delta_t, torch.Tensor):
                delta_t = torch.tensor(delta_t.tolist(), dtype=torch.float32)
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask.tolist(), dtype=torch.float32)

            # 位置编码
            times = torch.cumsum(delta_t, dim=0)
            max_time = t_max
            pe_sin = torch.sin(times.unsqueeze(1) * 2 * np.pi / max_time)
            pe_cos = torch.cos(times.unsqueeze(1) * 2 * np.pi / max_time)
            pe = torch.cat([pe_sin, pe_cos], dim=1)
            x = torch.cat([x, pe], dim=1)

            if mask.shape[1] != x.shape[1]:
                pad_dim = x.shape[1] - mask.shape[1]
                mask_padding = torch.ones(mask.shape[0], pad_dim, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([mask, mask_padding], dim=1)

            return x.float(), mask.float(), delta_t.float(), file_path

        except Exception as e:
            print(f"[ERROR] 加载文件失败: {self.file_paths[idx]}, 错误: {e}")
            raise


def custom_collate(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    batch: List[(x, mask, delta_t, path)]
        x       : [T_i, D]
        mask    : [T_i, D]
        delta_t : [T_i]
    """
    xs, masks, delta_ts, paths = zip(*batch)

    max_t = max(x.shape[0] for x in xs)
    feat_dim = xs[0].shape[1]

    padded_xs = []
    padded_masks = []
    padded_delta_ts = []

    for x, m, dt in zip(xs, masks, delta_ts):
        T_i = x.shape[0]
        pad_len = max_t - T_i

        if pad_len > 0:
            pad_x = torch.zeros(pad_len, feat_dim, dtype=x.dtype)
            x = torch.cat([x, pad_x], dim=0)
        padded_xs.append(x)

        if pad_len > 0:
            pad_m = torch.zeros(pad_len, feat_dim, dtype=m.dtype)
            m = torch.cat([m, pad_m], dim=0)
        padded_masks.append(m)

        if pad_len > 0:
            pad_dt = torch.zeros(pad_len, dtype=dt.dtype)
            dt = torch.cat([dt, pad_dt], dim=0)
        padded_delta_ts.append(dt)

    return (
        torch.stack(padded_xs),
        torch.stack(padded_masks),
        torch.stack(padded_delta_ts),
        list(paths),
    )


def find_csv_paths(data_dir: str, normal: bool = True) -> List[str]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    suffix = '_normal.csv' if normal else '_abnormal.csv'
    paths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(suffix)
    ]
    if not paths:
        raise FileNotFoundError(f"未找到匹配的{'正常' if normal else '异常'}数据文件，请检查命名及路径。")
    return sorted(paths)


def split_normal_data(
    normal_paths: Sequence[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dict[str, List[str]]]:
    if not normal_paths:
        raise ValueError("正常样本路径列表为空")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > len(normal_paths):
        raise ValueError("n_splits cannot exceed the number of normal samples.")

    normal_paths_array = np.array(normal_paths)

    kfold_splits = []
    for train_idx, val_idx in _kfold_indices(len(normal_paths_array), n_splits, random_state):
        train_paths = normal_paths_array[train_idx].tolist()
        val_paths = normal_paths_array[val_idx].tolist()
        kfold_splits.append({
            'train': train_paths,
            'val': val_paths
        })

    print(f"[DATA] 成功分割 {len(normal_paths)} 个正常样本为 {n_splits} 折")
    for i, split in enumerate(kfold_splits):
        print(f"  - 折 {i + 1}: 训练 {len(split['train'])} 个, 验证 {len(split['val'])} 个")

    return kfold_splits


def get_input_dim(data_dir: str) -> int:
    """
    注意：这里只返回原始 CSV 特征维度，不含位置编码。
    训练建模时优先使用 batch 的 x.shape[-1] 或 config.get_model_params(...).
    """
    normal_paths = find_csv_paths(data_dir, normal=True)
    if not normal_paths:
        raise ValueError("No data files found to determine input dimension.")

    dims = []
    for path in normal_paths:
        arr = _read_csv_array(path)
        dims.append(arr.shape[1])
        print(f"[DATA DEBUG] 文件 {os.path.basename(path)} 维度: {arr.shape}")

    max_d = max(dims)
    anomalies = [(p, d) for p, d in zip(normal_paths, dims) if d != max_d]

    if anomalies:
        print(f"[DATA DEBUG] 存在列数异常文件: {anomalies}")

    print(f"[DATA DEBUG] 最终确定的原始输入维度: {max_d}")
    return max_d
