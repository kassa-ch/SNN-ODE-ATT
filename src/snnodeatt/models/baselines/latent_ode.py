# models/SNN_LatentODE.py
import torch
import torch.nn as nn
from torchdiffeq import odeint  # 说明：此处未直接使用，RK4 为手写实现

# ======================================================
# Latent ODE Function f_ODE(z)
# ======================================================
class ODEFunc(nn.Module):
    """Latent ODE 神经微分算子 f_ODE(z)"""
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, t, z):
        return self.net(z)

# ======================================================
# Latent ODE Cell
# ======================================================
class LatentODE_Cell(nn.Module):
    """
    单步 Latent ODE Cell（RK4）
    """
    def __init__(
        self,
        input_dim,
        latent_dim,
        dropout_rate=0.0,
        alpha_ode_init=1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ODE 权重（可学习）
        self.alpha_ode = nn.Parameter(torch.tensor(float(alpha_ode_init)))

        # 输入映射
        self.input_to_latent = nn.Linear(input_dim, latent_dim)
        self.norm_latent = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # ODE 动力学
        self.ode_func = ODEFunc(latent_dim)

    def forward(self, x_t, z_prev, mask_t, delta_t):
        """
        x_t     : [B, D]
        z_prev  : [B, Z]
        mask_t  : [B, D] or [B, 1]
        delta_t : [B]
        """

        B = x_t.size(0)
        dt = delta_t.view(B, 1)

        # 输入注入（非 ODE 部分）
        input_latent = self.norm_latent(self.input_to_latent(x_t))

        # ===============================
        # RK4 积分
        # ===============================
        k1 = self.ode_func(0.0, z_prev)
        k2 = self.ode_func(0.0, z_prev + 0.5 * dt * k1)
        k3 = self.ode_func(0.0, z_prev + 0.5 * dt * k2)
        k4 = self.ode_func(0.0, z_prev + dt * k3)

        dz = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # ===============================
        # 状态更新
        # ===============================
        z_new = z_prev + self.alpha_ode * dz + 0.1 * input_latent
        z_new = torch.tanh(self.norm_latent(z_new))

        # ===============================
        # Mask
        # ===============================
        if mask_t.dim() == 2:
            mask_gate = mask_t.mean(dim=-1, keepdim=True)
        else:
            mask_gate = mask_t

        z_new = z_new * mask_gate
        z_new = self.dropout(z_new)

        return z_new

# ======================================================
# Predictive Latent ODE Model (Unified Interface)
# ======================================================
class PredictiveLatentODE_Model(nn.Module):
    """
    ✅ 接口与 SNN / SNN-ODE 完全一致
    """
    def __init__(self, input_dim, latent_dim, dropout_rate=0.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.event_seq_kind = "none"

        self.cell = LatentODE_Cell(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )

        self.fc_recon = nn.Linear(latent_dim, input_dim)
        self.fc_pred  = nn.Linear(latent_dim, input_dim)

    # --------------------------------------------------
    # delta_t 统一规范
    # --------------------------------------------------
    def _normalize_delta_ts(self, delta_ts, x):
        B, T, _ = x.shape
        device = x.device

        if delta_ts is None:
            raise ValueError("LatentODE requires explicit delta_t or delta_ts")

        if isinstance(delta_ts, (int, float)):
            delta_ts = torch.tensor(delta_ts, device=device)

        if delta_ts.dim() == 0:
            delta_ts = delta_ts.unsqueeze(0).expand(B, T)
        elif delta_ts.dim() == 1:
            delta_ts = delta_ts.unsqueeze(0).expand(B, T)
        elif delta_ts.dim() == 2:
            pass
        else:
            raise ValueError(f"非法 delta_ts 维度: {delta_ts.shape}")

        return delta_ts

    def forward(self, x, mask, delta_t=None, delta_ts=None):
        """
        x        : [B, T, D]
        mask     : [B, T, D]
        delta_t  : None | scalar | [T] | [B, T]
        """

        if delta_ts is None:
            delta_ts = delta_t

        delta_ts = self._normalize_delta_ts(delta_ts, x)

        B, T, D = x.shape
        device = x.device

        z = torch.zeros(B, self.latent_dim, device=device)

        recons, preds, z_seq = [], [], []

        for t in range(T):
            z = self.cell(
                x[:, t],
                z,
                mask[:, t],
                delta_ts[:, t],
            )

            recons.append(self.fc_recon(z))
            preds.append(self.fc_pred(z))
            z_seq.append(z)

        recons = torch.stack(recons, dim=1)
        preds  = torch.stack(preds, dim=1)
        z_seq  = torch.stack(z_seq, dim=1)

        z_mean = z_seq.mean(dim=1)
        mem_seq = z_seq
        mem_reset_seq = z_seq
        aux_seq = torch.zeros_like(z_seq)
        h_seq = torch.cat([mem_reset_seq, aux_seq], dim=-1)

        return recons, preds, z_mean, mem_seq, mem_reset_seq, aux_seq, h_seq

# ======================================================
# Adapter (for compatibility)
# ======================================================
LatentODEAdapter = LatentODE_Cell
