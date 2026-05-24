import torch
import torch.nn as nn

# =========================
# 工具函数
# =========================
def init_network_weights(net, std=0.01):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def split_last_dim(data):
    last_dim = data.size(-1) // 2
    return data[..., :last_dim], data[..., last_dim:]

# =========================
# GRU 单元（弱化版）
# =========================
class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=50, **kwargs):
        super().__init__()
        total_input_dim = latent_dim + input_dim

        self.update_gate = nn.Sequential(
            nn.Linear(total_input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(total_input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid()
        )
        self.new_state_net = nn.Sequential(
            nn.Linear(total_input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim)
        )

        self.epsilon = 0.1

    def forward(self, y_mean, x):
        y_concat = torch.cat([y_mean, x], -1)
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, x], -1)
        new_state = self.new_state_net(concat)
        gru_state = (1 - update_gate) * new_state + update_gate * y_mean
        return (1 - self.epsilon) * y_mean + self.epsilon * gru_state

# =========================
# 退化 ODE 动力学
# =========================
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.linear.weight)
        self.decay = 0.1

    def forward(self, t, y):
        return -self.decay * self.linear(y)

# =========================
# 简单 Euler ODE Solver
# =========================
class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, **kwargs):
        super().__init__()
        self.ode_func = ode_func

    def forward(self, y0, time_steps):
        solution = [y0.unsqueeze(0)]
        current_y = y0
        for i in range(1, len(time_steps)):
            dt = time_steps[i] - time_steps[i - 1]
            dy = self.ode_func(time_steps[i - 1], current_y)
            dy = dy.clamp(-1.0, 1.0)
            current_y = current_y + dt * dy
            current_y = current_y.clamp(-10.0, 10.0)
            solution.append(current_y.unsqueeze(0))
        return torch.cat(solution, dim=0)

# =========================
# Encoder（ODE‑RNN）
# =========================
class Encoder_z0_ODE_RNN(nn.Module):
    def __init__(self, latent_dim, input_dim, solver, device, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        self.gru = GRU_unit(latent_dim, input_dim, **kwargs).to(device)
        self.solver = solver

        self.transform_z0 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
        )

        init_network_weights(self, std=0.01)

    def forward(self, data, time_steps):
        B, T, _ = data.size()
        h = torch.randn(B, self.latent_dim, device=data.device) * 0.01

        for i in range(T):
            if i > 0:
                dt = time_steps[i] - time_steps[i - 1]
                dy = self.solver.ode_func(time_steps[i - 1], h)
                dy = dy.clamp(-1.0, 1.0)
                h = h + dt * dy
                h = h.clamp(-10.0, 10.0)

            h = self.gru(h, data[:, i, :])

        z0_params = self.transform_z0(h)
        mean_z0, std_z0 = split_last_dim(z0_params)
        std_z0 = torch.exp(std_z0.clamp(min=-5.0))

        return mean_z0, std_z0

# =========================
# Decoder
# =========================
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, output_dim)
        )
        init_network_weights(self, std=0.01)

    def forward(self, z):
        return self.net(z)

# =========================
# 主模型（统一接口）
# =========================
class PredictiveODERNN_Model(nn.Module):
    """
    ✅ ODE‑RNN（Algorithm‑1 风格）
    ✅ 接口与 SNN / LatentODE / GRU / BiLSTM 完全一致
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.3, **kwargs):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.event_seq_kind = "none"
        device = kwargs.get(
            'device',
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.ode_func = ODEFunc(hidden_dim)
        self.ode_solver = DiffeqSolver(self.ode_func)

        self.encoder = Encoder_z0_ODE_RNN(
            hidden_dim, input_dim, self.ode_solver, device
        )

        self.fc_recon = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
        self.fc_pred = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        init_network_weights(self, std=0.01)

    # --------------------------------------------------
    # delta_t 统一规范
    # --------------------------------------------------
    def _normalize_delta_ts(self, delta_ts, x):
        B, T, _ = x.shape
        device = x.device

        if delta_ts is None:
            delta_ts = torch.ones(B, T, device=device)

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

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, x, mask, delta_t=None, delta_ts=None):
        """
        x        : [B, T, D]
        mask     : [B, T, D]
        delta_t  : None | scalar | [T] | [B, T]
        """

        if delta_ts is None:
            delta_ts = delta_t

        delta_ts = self._normalize_delta_ts(delta_ts, x)

        B, T, D = x.size()
        masked_x = x * mask

        # -------- 构造连续时间轴 --------
        time_steps = torch.cumsum(delta_ts, dim=1)
        time_steps = time_steps - time_steps[:, :1]   # t0 = 0
        time_steps = time_steps[0]                     # 所有 batch 共享时间轴

        # -------- Encoder --------
        mean_z0, std_z0 = self.encoder(masked_x, time_steps)

        z0 = mean_z0 + std_z0 * torch.randn_like(mean_z0)

        # -------- ODE 解轨迹 --------
        z_seq_latent = self.ode_solver(z0, time_steps)   # [T, B, H]
        z_seq = z_seq_latent.permute(1, 0, 2)             # [B, T, H]
        z_seq = self.dropout(z_seq)

        # -------- 输出 --------
        recons = self.fc_recon(z_seq)
        preds  = self.fc_pred(z_seq)

        z_mean = z_seq.mean(dim=1)
        mem_seq = z_seq
        mem_reset_seq = z_seq
        aux_seq = torch.zeros_like(z_seq)
        h_seq = torch.cat([mem_reset_seq, aux_seq], dim=-1)

        return recons, preds, z_mean, mem_seq, mem_reset_seq, aux_seq, h_seq

    def get_model_info(self):
        return {
            "model_type": "PredictiveODERNN_degenerated",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "time_aware": True,
            "explicit_ode": True
        }
