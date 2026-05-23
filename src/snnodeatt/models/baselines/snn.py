# SNN.py
import torch
import torch.nn as nn
import sys
import os

# ======================================================
# Path setup
# ======================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from common import init_network_weights

# ======================================================
# Strict LIF SNN Cell (Time-aware)
# ======================================================
class SNN_Cell(nn.Module):
    """
    ✅ 严格 LIF：
    - 线性突触
    - 指数泄露
    - spike 驱动 recurrent
    - 无 ReLU / LN / ODE
    - ✅ 支持 delta_t（连续 / 不规则时间）
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        threshold,
        tau_mem,
        tau_syn,
        reset_mechanism="zero",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism

        # 时间常数（不学习，符合 LIF）
        self.register_buffer("tau_mem", torch.tensor(float(tau_mem)))
        self.register_buffer("tau_syn", torch.tensor(float(tau_syn)))

        # 线性突触
        self.input_to_syn = nn.Linear(input_dim, hidden_dim, bias=True)
        self.hidden_to_syn = nn.Linear(hidden_dim, hidden_dim, bias=True)

        init_network_weights(self, std=0.001)

    def forward(self, x_t, syn, mem, prev_spike, mask_t, delta_t):
        """
        x_t        : [B, D]
        syn        : [B, H]
        mem        : [B, H]
        prev_spike : [B, H]
        mask_t     : [B, D] or [B, 1]
        delta_t    : [B]
        """

        B = x_t.size(0)
        dt = delta_t.view(B, 1)

        # ===============================
        # 时间感知的 LIF 泄露
        # ===============================
        alpha_syn = torch.exp(-dt / self.tau_syn)
        alpha_mem = torch.exp(-dt / self.tau_mem)

        # ===============================
        # 突触电流
        # ===============================
        input_current = self.input_to_syn(x_t)

        recurrent_scale = 10.0
        recurrent_current = recurrent_scale * self.hidden_to_syn(prev_spike)

        syn = alpha_syn * syn + input_current + recurrent_current

        # ===============================
        # 膜电位
        # ===============================
        mem = alpha_mem * mem + syn

        # ===============================
        # Spike
        # ===============================
        spike = (mem >= self.threshold).float()

        # ===============================
        # Reset
        # ===============================
        if self.reset_mechanism == "subtract":
            mem = mem - spike * self.threshold
        elif self.reset_mechanism == "zero":
            mem = mem * (1.0 - spike)

        # ===============================
        # Mask（与数据 pipeline 对齐）
        # ===============================
        if mask_t.dim() == 2:
            mask_gate = mask_t.mean(dim=-1, keepdim=True)
        else:
            mask_gate = mask_t

        spike = spike * mask_gate

        return syn, mem, spike, spike

# ======================================================
# Predictive SNN Model (Time-aware, minimal)
# ======================================================
class PredictiveSNN_Model(nn.Module):
    """
    ✅ 最基础 LIF-SNN
    ✅ 统一 delta_t / delta_ts 接口
    ✅ 兼容多实验设置
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        threshold=30,
        tau_mem=80,
        tau_syn=10,
        reset_mechanism="zero",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.cell = SNN_Cell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            threshold=threshold,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            reset_mechanism=reset_mechanism,
        )

        # 下游仅使用 spike（严格 SNN）
        self.fc_recon = nn.Linear(hidden_dim, input_dim)
        self.fc_pred  = nn.Linear(hidden_dim, input_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # --------------------------------------------------
    # delta_t 统一（与 SNN-ODE 风格一致）
    # --------------------------------------------------
    def _normalize_delta_ts(self, delta_ts, x):
        B, T, _ = x.shape
        device = x.device

        if delta_ts is None:
            return torch.ones(B, T, device=device)

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
        x    : [B, T, D]
        mask : [B, T, D]
        """

        if delta_ts is None:
            delta_ts = delta_t

        delta_ts = self._normalize_delta_ts(delta_ts, x)

        B, T, D = x.shape
        device = x.device

        syn = torch.zeros(B, self.hidden_dim, device=device)
        mem = torch.zeros(B, self.hidden_dim, device=device)
        spike = torch.zeros(B, self.hidden_dim, device=device)

        recons, preds, spikes = [], [], []

        for t in range(T):
            syn, mem, spike, _ = self.cell(
                x[:, t],
                syn,
                mem,
                spike,
                mask[:, t],
                delta_ts[:, t],
            )

            recons.append(self.fc_recon(spike))
            preds.append(self.fc_pred(spike))
            spikes.append(spike)

        recon = torch.stack(recons, dim=1)
        pred  = torch.stack(preds, dim=1)
        z_seq = torch.stack(spikes, dim=1)
        z_mean = z_seq.mean(dim=1)

        return recon, pred, z_mean, z_seq

    def get_spike_sequence(self, z_seq):
        return z_seq