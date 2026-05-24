import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("PYTORCH_JIT", "0")

import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

try:
    from sklearn.model_selection import KFold, train_test_split
except Exception:
    KFold = None

    def train_test_split(items, test_size=0.25, random_state=42, shuffle=True):
        items = list(items)
        indices = np.arange(len(items))
        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(indices)
        if isinstance(test_size, float):
            n_test = int(np.ceil(len(items) * test_size))
        else:
            n_test = int(test_size)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        return [items[i] for i in train_idx], [items[i] for i in test_idx]

    class _FallbackKFold:
        def __init__(self, n_splits=5, shuffle=False, random_state=None):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def split(self, items):
            n_samples = len(items)
            indices = np.arange(n_samples)
            if self.shuffle:
                rng = np.random.default_rng(self.random_state)
                rng.shuffle(indices)
            fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
            fold_sizes[: n_samples % self.n_splits] += 1
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                test_idx = indices[start:stop]
                train_idx = np.concatenate([indices[:start], indices[stop:]])
                yield train_idx, test_idx
                current = stop

    KFold = _FallbackKFold
from torch.utils.data import DataLoader

from data_loader import TimeSeriesDataset, custom_collate, find_csv_paths
from config import (
    DEVICE,
    ACTIVE_MODELS,
    MODEL_SAVE_DIR,
    get_enhanced_loss_params,
    get_experiment_models,
    get_model_params,
    get_training_params,
)
from enhanced_loss import EnhancedSNNLoss
from model_factory import create_model


# =========================================================
# 手写优化器：绕开 torch.optim
# =========================================================
class ManualOptimizer:
    def __init__(self, params):
        self.params = [p for p in params if p.requires_grad]
        self.param_groups = [{'lr': 0.0}]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self):
        raise NotImplementedError


class ManualAdam(ManualOptimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.param_groups = [{'lr': lr}]
        self.state = {}
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p

            if p not in self.state:
                self.state[p] = {
                    'm': torch.zeros_like(p),
                    'v': torch.zeros_like(p),
                }

            m = self.state[p]['m']
            v = self.state[p]['v']

            m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


class ManualStepLR:
    def __init__(self, optimizer, step_size=20, gamma=0.75):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.step_size > 0 and self.last_epoch % self.step_size == 0:
            for group in self.optimizer.param_groups:
                group['lr'] *= self.gamma
            if hasattr(self.optimizer, 'lr'):
                self.optimizer.lr = self.optimizer.param_groups[0]['lr']


# =========================================================
# forward 适配：统一当前最新版 SNNODE / SNNODEATT
# 统一返回 8 个值：
# recon, pred, z_mean, mem_seq, mem_reset_seq, spike_seq, rate_seq, h_seq
# =========================================================
def model_forward(model, model_name, x, mask=None, delta_t=None, n_samples=1):
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

    # 当前最新版：
    # SNNODE    -> recons, preds, z_mean, mem_seq, mem_reset_seq, spike_seq, h_seq
    # SNNODEATT -> recons, preds, z_mean, mem_seq, mem_reset_seq, rate_seq,  h_seq
    if len(outputs) == 7:
        recon, pred, z_mean, mem_seq, mem_reset_seq, seq6, h_seq = outputs

        event_kind = getattr(model, "event_seq_kind", None)
        if event_kind == "rate":
            spike_seq = None
            rate_seq = seq6
        elif event_kind == "spike":
            spike_seq = seq6
            rate_seq = None
        else:
            model_name_upper = model_name.upper()
            class_name_upper = model.__class__.__name__.upper()
            is_att_model = ('ATT' in model_name_upper) or ('ATT' in class_name_upper)
            if is_att_model:
                spike_seq = None
                rate_seq = seq6
            else:
                spike_seq = seq6
                rate_seq = None

        return recon, pred, z_mean, mem_seq, mem_reset_seq, spike_seq, rate_seq, h_seq

    # 兼容极旧版本：4 输出
    if len(outputs) == 4:
        recon, pred, z_mean, mem_seq = outputs
        mem_reset_seq = None
        spike_seq = None
        rate_seq = None
        h_seq = None
        return recon, pred, z_mean, mem_seq, mem_reset_seq, spike_seq, rate_seq, h_seq

    raise ValueError(
        f"当前训练系统仅兼容 forward 返回 4 个或 7 个值，但收到 {len(outputs)} 个值"
    )


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[SEED] 设置随机种子: {seed}")


def compute_anomaly_scores(model, model_name, data_loader, n_samples=1):
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for batch in data_loader:
            x, mask, delta_t, _ = batch
            x = x.to(DEVICE)
            mask = mask.to(DEVICE)
            delta_t = delta_t.to(DEVICE)

            reconstructions, predictions, _, _, _, _, _, _ = model_forward(
                model, model_name, x, mask=mask, delta_t=delta_t, n_samples=n_samples
            )

            reconstruction_error = F.mse_loss(
                reconstructions * mask,
                x * mask,
                reduction='none'
            )
            sample_scores = torch.mean(reconstruction_error, dim=(1, 2))
            anomaly_scores.extend(sample_scores.detach().cpu().tolist())

    return np.asarray(anomaly_scores, dtype=np.float32)


def compute_train_scores_for_threshold(model, model_name, train_loader, n_samples=1):
    print("[TRAIN] 计算训练集异常分数用于阈值计算...")
    train_scores = compute_anomaly_scores(model, model_name, train_loader, n_samples)

    if train_scores.size > 0:
        print(
            f"[TRAIN] 训练集分数统计: "
            f"min={np.min(train_scores):.4f}, "
            f"max={np.max(train_scores):.4f}, "
            f"mean={np.mean(train_scores):.4f}"
        )
    else:
        print("[TRAIN] 训练集分数为空")

    return train_scores


def validate_epoch(model, val_dl, criterion, model_name, n_samples):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for x, m, delta_t, _ in val_dl:
            x, m, delta_t = x.to(DEVICE), m.to(DEVICE), delta_t.to(DEVICE)

            recon, pred, z_mean, mem_seq, mem_reset_seq, spike_seq, rate_seq, h_seq = model_forward(
                model, model_name, x, mask=m, delta_t=delta_t, n_samples=n_samples
            )

            val_loss, _ = criterion(
                reconstructions=recon,
                predictions=pred,
                x=x,
                mask=m,
                z_mean=z_mean,
                mem_seq=mem_seq,
                mem_reset_seq=mem_reset_seq,
                spike_seq=spike_seq,
                rate_seq=rate_seq,
                h_seq=h_seq,
                delta_ts=delta_t,
            )
            total_val_loss += val_loss.item()

    return total_val_loss / max(len(val_dl), 1)


def evaluate_epoch_with_components(model, data_dl, criterion, model_name, n_samples):
    model.eval()
    total_loss = 0.0
    loss_components = {}

    with torch.no_grad():
        for x, m, delta_t, _ in data_dl:
            x, m, delta_t = x.to(DEVICE), m.to(DEVICE), delta_t.to(DEVICE)

            recon, pred, z_mean, mem_seq, mem_reset_seq, spike_seq, rate_seq, h_seq = model_forward(
                model, model_name, x, mask=m, delta_t=delta_t, n_samples=n_samples
            )

            loss, loss_info = criterion(
                reconstructions=recon,
                predictions=pred,
                x=x,
                mask=m,
                z_mean=z_mean,
                mem_seq=mem_seq,
                mem_reset_seq=mem_reset_seq,
                spike_seq=spike_seq,
                rate_seq=rate_seq,
                h_seq=h_seq,
                delta_ts=delta_t,
            )
            total_loss += loss.item()
            for key, value in loss_info.items():
                loss_components.setdefault(key, []).append(float(value))

    avg_loss = total_loss / max(len(data_dl), 1)
    avg_components = {k: float(np.mean(v)) for k, v in loss_components.items()}
    return avg_loss, avg_components


def save_data_split_config(
    exp_key,
    model_name,
    input_dim,
    train_features_shape,
    val_features_shape,
    test_features_shape,
    train_paths,
    val_paths,
    test_paths,
    apply_poisson_sampling,
    base_lambda,
    run_id=None,
    fold_id=None,
):
    config = {
        'input_dim': input_dim,
        'train_features_shape': tuple(train_features_shape),
        'val_features_shape': tuple(val_features_shape),
        'test_features_shape': tuple(test_features_shape),
        'train_samples': len(train_paths),
        'val_samples': len(val_paths),
        'test_samples': len(test_paths),
        'train_paths': train_paths,
        'val_paths': val_paths,
        'test_paths': test_paths,
        'apply_poisson_sampling': apply_poisson_sampling,
        'base_lambda': base_lambda,
        'data_config_saved': True,
        'run_id': run_id,
        'fold_id': fold_id,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    os.makedirs(f"results/{exp_key}/data_splits", exist_ok=True)

    filename_parts = [model_name]
    if fold_id is not None:
        filename_parts.append(f'fold_{fold_id}')
    filename_parts.append('data_splits.pth')

    config_path = f"results/{exp_key}/data_splits/{'_'.join(filename_parts)}"
    torch.save(config, config_path)
    print(f"[CONFIG] 已保存数据划分配置: {config_path}")
    return config_path


def split_data_kfold_corrected(normal_paths, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for train_val_idx, test_val_idx in kf.split(normal_paths):
        train_paths = [normal_paths[i] for i in train_val_idx]
        test_val_paths = [normal_paths[i] for i in test_val_idx]

        val_paths, test_paths = train_test_split(
            test_val_paths,
            test_size=0.5,
            random_state=random_state,
            shuffle=True
        )

        splits.append((train_paths, val_paths, test_paths))

    print(f"[DATA] 修正的K折交叉验证划分完成: {n_splits}折")
    return splits


def split_train_test_val(normal_paths, test_ratio=0.1, val_ratio=0.1, shuffle=True, random_state=42):
    train_val_paths, test_paths = train_test_split(
        normal_paths,
        test_size=test_ratio,
        random_state=random_state,
        shuffle=shuffle
    )

    val_ratio_adj = val_ratio / (1 - test_ratio)
    train_paths, val_paths = train_test_split(
        train_val_paths,
        test_size=val_ratio_adj,
        random_state=random_state,
        shuffle=shuffle
    )

    print(f"数据划分: 训练集 {len(train_paths)} 个文件, 验证集 {len(val_paths)} 个文件, 测试集 {len(test_paths)} 个文件")
    return train_paths, val_paths, test_paths


def setup_optimizer(model, model_name, lr, weight_decay, odefunc_weight_decay):
    has_odefunc = hasattr(model, 'cell') and hasattr(model.cell, 'ode_func')
    force_manual = os.environ.get("USE_MANUAL_OPTIMIZER", "").lower() in {"1", "true", "yes"}

    if force_manual:
        print(f"[INFO] 使用 ManualAdam；has_odefunc={has_odefunc}")
        optimizer = ManualAdam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        return optimizer, has_odefunc

    try:
        print(f"[INFO] 使用标准 torch.optim.Adam；has_odefunc={has_odefunc}")
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    except Exception as e:
        print(f"[WARNING] torch.optim.Adam 初始化失败，回退到 ManualAdam: {e}")
        optimizer = ManualAdam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    return optimizer, has_odefunc


def train_epoch(model, train_dl, optimizer, criterion, model_name, n_samples, has_odefunc, grad_clip, odefunc_grad_clip):
    model.train()
    total_train_loss = 0.0
    epoch_loss_components = {}

    for x, m, delta_t, _ in train_dl:
        x, m, delta_t = x.to(DEVICE), m.to(DEVICE), delta_t.to(DEVICE)

        recon, pred, z_mean, mem_seq, mem_reset_seq, spike_seq, rate_seq, h_seq = model_forward(
            model, model_name, x, mask=m, delta_t=delta_t, n_samples=n_samples
        )

        loss, loss_info = criterion(
            reconstructions=recon,
            predictions=pred,
            x=x,
            mask=m,
            z_mean=z_mean,
            mem_seq=mem_seq,
            mem_reset_seq=mem_reset_seq,
            spike_seq=spike_seq,
            rate_seq=rate_seq,
            h_seq=h_seq,
            delta_ts=delta_t,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if has_odefunc and hasattr(model, 'cell') and hasattr(model.cell, 'ode_func'):
            torch.nn.utils.clip_grad_norm_(model.cell.ode_func.parameters(), max_norm=odefunc_grad_clip)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_train_loss += loss.item()
        for key, value in loss_info.items():
            epoch_loss_components.setdefault(key, []).append(value)

    avg_train_loss = total_train_loss / max(len(train_dl), 1)
    avg_loss_components = {k: float(np.mean(v)) for k, v in epoch_loss_components.items()}
    return avg_train_loss, avg_loss_components


def _plot_training_curve(train_loss_history, val_loss_history, model_name, early_stop=False, run_id=None, fold_id=None):
    if not train_loss_history or not val_loss_history:
        return

    try:
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss_history, label='Train Loss', linewidth=2)
        plt.plot(val_loss_history, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curve - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        filename_parts = [f'training_curve_{model_name}']
        if run_id is not None:
            filename_parts.append(f'run{run_id}')
        if fold_id is not None:
            filename_parts.append(f'fold_{fold_id}')
        if early_stop:
            filename_parts.append('earlystop')

        plt.savefig(f"{'_'.join(filename_parts)}.png", dpi=300, bbox_inches='tight')
    except Exception as exc:
        print(f"[WARN] 训练曲线保存失败，已跳过: {exc}")
    finally:
        plt.close()


def _plot_detailed_loss_curves(loss_components_history, model_name, run_id=None, fold_id=None):
    if not loss_components_history:
        return

    fig = None
    try:
        valid_components = ['recon_loss', 'pred_loss', 'stability_loss']
        components = [comp for comp in valid_components if comp in loss_components_history[0]]

        if not components:
            return

        n_components = len(components)
        fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 4))
        if n_components == 1:
            axes = [axes]

        for i, component in enumerate(components):
            values = [epoch_data.get(component, 0) for epoch_data in loss_components_history]
            axes[i].plot(values, label=component, linewidth=2)
            axes[i].set_title(f'{component} Loss')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        title_parts = [f'detailed_loss_{model_name}']
        if run_id is not None:
            title_parts.append(f'run{run_id}')
        if fold_id is not None:
            title_parts.append(f'fold_{fold_id}')

        fig.tight_layout()
        plt.savefig(f"{'_'.join(title_parts)}.png", dpi=300, bbox_inches='tight')
    except Exception as exc:
        print(f"[WARN] 详细损失曲线保存失败，已跳过: {exc}")
    finally:
        if fig is not None:
            plt.close(fig)


def _build_dataloader(file_paths, apply_poisson_sampling, base_lambda, batch_size, shuffle):
    ds = TimeSeriesDataset(
        file_paths=file_paths,
        apply_poisson_sampling=apply_poisson_sampling,
        base_lambda=base_lambda,
        fixed_random_offset=not shuffle,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
        drop_last=False,
    )


def _infer_input_dim_from_loader(loader):
    sample_batch = next(iter(loader))
    x, _, _, _ = sample_batch
    return int(x.shape[-1]), tuple(x.shape)


def _save_checkpoint(model, model_path, model_name, model_params, input_dim, epoch, best_val_loss):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    state_dict = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in model.state_dict().items()
    }
    checkpoint = {
        'model_state_dict': state_dict,
        'model_name': model_name,
        'model_params': model_params,
        'input_dim': input_dim,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(checkpoint, model_path)
    print(f"[CHECKPOINT] 已保存最佳模型到: {model_path}")


def train_single_kfold_run(
    data_dir,
    model_path,
    model_name,
    train_paths,
    val_paths,
    test_paths,
    delta_t=None,
    num_epochs=None,
    batch_size=None,
    lr=None,
    beta_time=None,
    n_samples=None,
    apply_poisson_sampling=False,
    base_lambda=0.5,
    exp_key=None,
    run_id=1,
    fold_id=1,
    random_seed=42,
):
    print(f"[DEBUG] 训练配置: apply_poisson_sampling={apply_poisson_sampling}, base_lambda={base_lambda}")
    set_random_seeds(random_seed)
    print(f"[TRAIN] 开始第 {run_id} 次训练，第 {fold_id} 折，随机种子: {random_seed}")

    train_params = get_training_params(model_name)

    num_epochs = num_epochs if num_epochs is not None else train_params['num_epochs']
    batch_size = batch_size if batch_size is not None else train_params['batch_size']
    lr = lr if lr is not None else train_params['lr']
    beta_time = beta_time if beta_time is not None else train_params.get('beta_time', 0.1)
    n_samples = n_samples if n_samples is not None else train_params.get('n_samples', 1)

    early_stopping_patience = train_params.get('early_stopping_patience', 12)
    disable_early_stopping = os.environ.get('DISABLE_EARLY_STOPPING', '').lower() in {'1', 'true', 'yes'}
    if disable_early_stopping:
        print(f"[INFO] DISABLE_EARLY_STOPPING=1，训练将跑满 {num_epochs} 个 epoch")
    stopping_delta = train_params.get('stopping_delta', 5e-5)
    loss_explosion_threshold = train_params.get('loss_explosion_threshold', 8.0)
    grad_clip = train_params.get('grad_clip', 1.0)
    weight_decay = train_params.get('weight_decay', 1e-4)
    lr_scheduler_step_size = train_params.get('lr_scheduler_step_size', 20)
    lr_scheduler_gamma = train_params.get('lr_scheduler_gamma', 0.75)
    odefunc_weight_decay = train_params.get('odefunc_weight_decay', 5e-4)
    odefunc_grad_clip = train_params.get('odefunc_grad_clip', 0.8)

    train_dl = _build_dataloader(train_paths, apply_poisson_sampling, base_lambda, batch_size, shuffle=True)
    val_dl = _build_dataloader(val_paths, apply_poisson_sampling, base_lambda, batch_size, shuffle=False)
    test_dl = _build_dataloader(test_paths, apply_poisson_sampling, base_lambda, batch_size, shuffle=False)

    input_dim, train_shape = _infer_input_dim_from_loader(train_dl)
    _, val_shape = _infer_input_dim_from_loader(val_dl)
    _, test_shape = _infer_input_dim_from_loader(test_dl)

    print(f"[INFO] 检测到输入维度: {input_dim}")
    print(f"[DEBUG] 训练数据形状: {train_shape}")
    print(f"[DEBUG] 验证数据形状: {val_shape}")
    print(f"[DEBUG] 测试数据形状: {test_shape}")

    model_params = get_model_params(model_name, input_dim)
    clean_model_params = {k: v for k, v in model_params.items() if k != 'input_dim'}

    model = create_model(model_name, input_dim, **clean_model_params).to(DEVICE)
    print(f"[INFO] 成功创建模型: {model_name} (第 {run_id} 次训练, 第 {fold_id} 折)")

    save_data_split_config(
        exp_key=exp_key or 'default_exp',
        model_name=model_name,
        input_dim=input_dim,
        train_features_shape=train_shape,
        val_features_shape=val_shape,
        test_features_shape=test_shape,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        apply_poisson_sampling=apply_poisson_sampling,
        base_lambda=base_lambda,
        run_id=run_id,
        fold_id=fold_id,
    )

    loss_params = get_enhanced_loss_params(model_name)
    criterion = EnhancedSNNLoss(**loss_params)
    criterion.enable_debug()

    train_loss_history = []
    val_loss_history = []
    loss_components_history = []
    best_val_loss = float('inf')
    no_improve_count = 0
    is_stat_model = bool(getattr(model, 'is_stat_model', False) and hasattr(model, 'fit_statistical'))

    if is_stat_model:
        print(f"[TRAIN] {model_name} 使用统计拟合路径（无反向传播）")
        model.fit_statistical(train_dl)
        avg_train_loss, avg_loss_components = evaluate_epoch_with_components(
            model, train_dl, criterion, model_name, n_samples
        )
        avg_val_loss = validate_epoch(model, val_dl, criterion, model_name, n_samples)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        loss_components_history.append(avg_loss_components)
        best_val_loss = avg_val_loss

        _save_checkpoint(
            model=model,
            model_path=model_path,
            model_name=model_name,
            model_params=model_params,
            input_dim=input_dim,
            epoch=1,
            best_val_loss=best_val_loss
        )
        print(
            f"[STAT-EPOCH 001/001] train={avg_train_loss:.6f}, "
            f"val={avg_val_loss:.6f}"
        )
    else:
        optimizer, has_odefunc = setup_optimizer(model, model_name, lr, weight_decay, odefunc_weight_decay)
        if isinstance(optimizer, ManualOptimizer):
            scheduler = ManualStepLR(
                optimizer,
                step_size=lr_scheduler_step_size,
                gamma=lr_scheduler_gamma
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=lr_scheduler_step_size,
                gamma=lr_scheduler_gamma
            )

        for epoch in range(num_epochs):
            avg_train_loss, avg_loss_components = train_epoch(
                model, train_dl, optimizer, criterion, model_name, n_samples,
                has_odefunc, grad_clip, odefunc_grad_clip
            )
            avg_val_loss = validate_epoch(model, val_dl, criterion, model_name, n_samples)

            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            loss_components_history.append(avg_loss_components)

            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"[EPOCH {epoch + 1:03d}/{num_epochs:03d}] "
                f"train={avg_train_loss:.6f}, "
                f"val={avg_val_loss:.6f}, "
                f"lr={current_lr:.6e}"
            )

            if not np.isfinite(avg_train_loss) or not np.isfinite(avg_val_loss):
                raise ValueError(f"损失出现 NaN/Inf：train={avg_train_loss}, val={avg_val_loss}")

            if avg_train_loss > loss_explosion_threshold:
                print(f"[WARNING] 检测到训练损失过大 ({avg_train_loss:.4f})，提前停止")
                break

            if avg_val_loss < best_val_loss - stopping_delta:
                best_val_loss = avg_val_loss
                no_improve_count = 0
                _save_checkpoint(
                    model=model,
                    model_path=model_path,
                    model_name=model_name,
                    model_params=model_params,
                    input_dim=input_dim,
                    epoch=epoch + 1,
                    best_val_loss=best_val_loss
                )
            else:
                no_improve_count += 1

            if not disable_early_stopping and no_improve_count >= early_stopping_patience:
                print(f"[EARLY STOP] {early_stopping_patience} 个 epoch 无提升，停止训练")
                break

    _plot_training_curve(
        train_loss_history, val_loss_history, model_name,
        (not disable_early_stopping) and no_improve_count >= early_stopping_patience, run_id, fold_id
    )
    _plot_detailed_loss_curves(loss_components_history, model_name, run_id, fold_id)

    train_scores = compute_train_scores_for_threshold(model, model_name, train_dl, n_samples)

    os.makedirs('./train_scores', exist_ok=True)
    score_filename_parts = [model_name]
    if fold_id is not None:
        score_filename_parts.append(f'fold_{fold_id}')
    score_filename_parts.append('train_scores.npy')

    score_save_path = os.path.join('./train_scores', '_'.join(score_filename_parts))
    np.save(score_save_path, train_scores)

    print(f"[CONFIG] 第 {run_id} 次训练，第 {fold_id} 折完成")

    return {
        'training_results': {
            'model': model,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'best_val_loss': best_val_loss,
            'final_epoch': len(train_loss_history),
        },
        'model_info': {
            'model_name': model_name,
            'input_dim': input_dim,
            'model_params': model_params,
            'run_id': run_id,
            'fold_id': fold_id,
            'random_seed': random_seed
        },
        'data_info': {
            'train_samples': len(train_paths),
            'val_samples': len(val_paths),
            'test_samples': len(test_paths)
        },
        'train_scores': train_scores,
        'train_score_path': score_save_path
    }


def train_model_kfold(
    data_dir,
    model_name,
    exp_key=None,
    num_runs=1,
    n_folds=5,
    apply_poisson_sampling=None,
    base_lambda=0.5,
    **train_kwargs
):
    print(f"\n[TRAIN] 开始对模型 {model_name} 进行 {num_runs} 次 {n_folds} 折交叉验证训练")
    normal_paths = find_csv_paths(data_dir, normal=True)
    print(f"[DATA] 找到 {len(normal_paths)} 个正常样本")

    experiment_models = get_experiment_models(exp_key)
    if model_name not in experiment_models:
        raise ValueError(f"模型 {model_name} 不在实验 {exp_key} 配置中")

    base_filename = experiment_models[model_name]
    all_results = {}
    completed_folds = 0
    base_seed = 42

    for run_id in range(1, num_runs + 1):
        splits = split_data_kfold_corrected(normal_paths, n_splits=n_folds, random_state=base_seed + run_id)

        for fold_id, (train_paths, val_paths, test_paths) in enumerate(splits, start=1):
            fold_key = f'run{run_id}_fold{fold_id}'
            fold_model_filename = base_filename.replace('.pth', f'_fold{fold_id}.pth')
            fold_model_path = os.path.join(MODEL_SAVE_DIR, fold_model_filename)

            print(f"\n{'=' * 60}")
            print(f"[TRAIN] {model_name} - Run {run_id} Fold {fold_id}")
            print(f"{'=' * 60}")

            result = train_single_kfold_run(
                data_dir=data_dir,
                model_path=fold_model_path,
                model_name=model_name,
                train_paths=train_paths,
                val_paths=val_paths,
                test_paths=test_paths,
                apply_poisson_sampling=apply_poisson_sampling,
                base_lambda=base_lambda,
                exp_key=exp_key,
                run_id=run_id,
                fold_id=fold_id,
                random_seed=base_seed + run_id * 100 + fold_id,
                **train_kwargs
            )

            all_results[fold_key] = result
            completed_folds += 1

    return {
        'all_results': all_results,
        'n_folds': n_folds,
        'num_runs': num_runs,
        'completed_folds': completed_folds,
        'model_name': model_name,
    }


def train_all_models_kfold(
    data_dir,
    exp_key,
    model_names=None,
    num_runs=1,
    n_folds=5,
    apply_poisson_sampling=None,
    base_lambda=0.5
):
    if model_names is None:
        model_names = ACTIVE_MODELS

    experiment_models = get_experiment_models(exp_key)
    training_results = {}
    failed_models = {}

    for model_name in model_names:
        if model_name not in experiment_models:
            print(f"跳过 {model_name}，不在实验 {exp_key} 配置中")
            continue

        print(f"\n[TRAIN] 开始训练 {model_name} ({num_runs} 次 {n_folds} 折交叉验证)")

        try:
            result = train_model_kfold(
                data_dir=data_dir,
                model_name=model_name,
                exp_key=exp_key,
                num_runs=num_runs,
                n_folds=n_folds,
                apply_poisson_sampling=apply_poisson_sampling,
                base_lambda=base_lambda
            )
            training_results[model_name] = result
            print(f"[OK] {model_name} 所有训练完成")

        except Exception as e:
            failed_models[model_name] = str(e)
            print(f"[ERROR] 训练 {model_name} 失败: {e}")
            import traceback
            traceback.print_exc()

    return {
        'training_summary': training_results,
        'failed_models': failed_models,
        'experiment_key': exp_key,
        'total_models_trained': len(training_results),
        'total_models_failed': len(failed_models),
        'num_runs_per_model': num_runs,
        'n_folds_per_run': n_folds,
        'apply_poisson_sampling': apply_poisson_sampling,
        'base_lambda': base_lambda,
    }


def train_single_run(
    data_dir,
    model_path,
    model_name,
    delta_t=None,
    num_epochs=None,
    batch_size=None,
    lr=None,
    beta_time=None,
    n_samples=None,
    apply_poisson_sampling=False,
    base_lambda=0.5,
    exp_key=None,
):
    normal_paths = find_csv_paths(data_dir, normal=True)
    train_paths, val_paths, test_paths = split_train_test_val(normal_paths)

    return train_single_kfold_run(
        data_dir=data_dir,
        model_path=model_path,
        model_name=model_name,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        beta_time=beta_time,
        n_samples=n_samples,
        apply_poisson_sampling=apply_poisson_sampling,
        base_lambda=base_lambda,
        exp_key=exp_key,
        run_id=1,
        fold_id=1,
        random_seed=42,
    )
