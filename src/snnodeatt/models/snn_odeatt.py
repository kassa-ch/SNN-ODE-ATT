# C:\Users\kassa\Desktop\半导体晶圆项目ATT\models\SNN_ODEATT.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.mask import normalize_sequence_mask


class ODEFunc(nn.Module):
    """
    f_ODE(M) = LN(W5 * tanh(W4 * M + b4) + b5)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, t, m):
        h = torch.tanh(self.fc1(m))
        h = self.fc2(h)
        return self.norm(h)


class ContinuousSNNODECell(nn.Module):
    """
    Same continuous local dynamics as SNN_ODE:
    - S(t), M_pre(t), R(t), M*(t)
    - Numerical evaluation on observed time grid with delta_t.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        tau_mem,
        tau_syn_base,
        threshold=1.0,
        tau_min=1.0,
        tau_max=10.0,
        alpha_ode_init=0.0,
        beta_spike=10.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.threshold = float(threshold)
        self.beta_spike = float(beta_spike)

        self.register_buffer("tau_mem", torch.tensor(float(tau_mem)))
        self.register_buffer("tau_syn_base", torch.tensor(float(tau_syn_base)))
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        self.alpha_ode_raw = nn.Parameter(torch.tensor(float(alpha_ode_init)))

        self.in_to_syn = nn.Linear(input_dim, hidden_dim)
        self.mem_to_syn = nn.Linear(hidden_dim, hidden_dim)

        self.tau_mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.ode_func = ODEFunc(hidden_dim)
        self.syn_to_mem = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ode_to_mem = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mem_bias = nn.Parameter(torch.zeros(hidden_dim))

        self.norm_syn = nn.LayerNorm(hidden_dim)
        self.norm_mem = nn.LayerNorm(hidden_dim)

    def forward(self, x_t, s_t, m_prev_reset, r_prev, delta_t):
        bsz = x_t.size(0)
        dt = delta_t.view(bsz, 1).to(dtype=x_t.dtype).clamp_min(1e-6)

        tau_input = torch.cat([x_t, m_prev_reset], dim=-1)
        rho = F.softplus(self.tau_mlp(tau_input))
        tau_s_t = torch.clamp(
            self.tau_syn_base * (1.0 + rho),
            self.tau_min,
            self.tau_max,
        )
        gamma_s = 1.0 / tau_s_t

        syn_drive = self.in_to_syn(x_t) + self.mem_to_syn(m_prev_reset)
        ds_dt = -gamma_s * s_t + syn_drive
        s_next = self.norm_syn(s_t + dt * ds_dt)

        k1 = self.ode_func(0.0, m_prev_reset)
        k2 = self.ode_func(0.0, m_prev_reset + 0.5 * dt * k1)
        k3 = self.ode_func(0.0, m_prev_reset + 0.5 * dt * k2)
        k4 = self.ode_func(0.0, m_prev_reset + dt * k3)
        f_ode_rk = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        gamma_m = 1.0 / self.tau_mem
        alpha_ode = 0.5 * torch.sigmoid(self.alpha_ode_raw)
        d_m_dt = (
            -gamma_m * m_prev_reset
            + self.syn_to_mem(s_next)
            + alpha_ode * self.ode_to_mem(f_ode_rk)
            - self.threshold * r_prev
            + self.mem_bias
        )
        m_local_pre = self.norm_mem(m_prev_reset + dt * d_m_dt)

        return s_next, m_local_pre


class ContinuousAttentionSNNODE(nn.Module):
    """
    Continuous-attention interpretation:
    - Z(t) = M_local_pre(t)
    - A(t) = Integral_{0..t} alpha(t,tau)V(tau)d tau
    - Here we use a causal Riemann-sum approximation on sampled points.

    Unified output:
    (recons, preds, z_mean, mem_seq, mem_reset_seq, rate_seq, h_seq)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        tau_mem,
        tau_syn=None,
        tau_syn_base=None,
        threshold=1.0,
        beta_spike=10.0,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__()

        if tau_syn_base is None:
            if tau_syn is None:
                raise ValueError("必须提供 tau_syn 或 tau_syn_base")
            tau_syn_base = tau_syn

        self.hidden_dim = hidden_dim
        self.threshold = float(threshold)
        self.beta_spike = float(beta_spike)
        self.event_seq_kind = "rate"

        self.cell = ContinuousSNNODECell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            tau_mem=tau_mem,
            tau_syn_base=tau_syn_base,
            threshold=threshold,
            beta_spike=beta_spike,
        )

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.lambda_att_raw = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout_rate)
        self.fc_recon = nn.Linear(hidden_dim * 2, input_dim)
        self.fc_pred = nn.Linear(hidden_dim * 2, input_dim)

    def _normalize_delta_ts(self, delta_ts, x):
        bsz, t_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        if delta_ts is None:
            return torch.ones(bsz, t_len, device=device, dtype=dtype)

        if isinstance(delta_ts, (int, float)):
            delta_ts = torch.tensor(delta_ts, device=device, dtype=dtype)

        if not isinstance(delta_ts, torch.Tensor):
            raise TypeError(f"delta_ts 必须是 Tensor，收到 {type(delta_ts)}")

        delta_ts = delta_ts.to(device=device, dtype=dtype)

        if delta_ts.dim() == 0:
            delta_ts = delta_ts.unsqueeze(0).expand(bsz, t_len)
        elif delta_ts.dim() == 1:
            if delta_ts.shape[0] == t_len:
                delta_ts = delta_ts.unsqueeze(0).expand(bsz, t_len)
            elif delta_ts.shape[0] == bsz:
                delta_ts = delta_ts.unsqueeze(1).expand(bsz, t_len)
            else:
                raise ValueError(f"非法 delta_ts 形状: {tuple(delta_ts.shape)}")
        elif delta_ts.dim() == 2:
            if delta_ts.shape != (bsz, t_len):
                raise ValueError(
                    f"delta_ts 形状应为 {(bsz, t_len)}，实际为 {tuple(delta_ts.shape)}"
                )
        else:
            raise ValueError(f"非法 delta_ts 维度: {tuple(delta_ts.shape)}")

        return delta_ts

    def _causal_attention(self, mem_hist, dt_hist):
        """
        mem_hist: [B, t, H], sampled values of continuous trajectory Z(tau)
        dt_hist: [B, t], integration steps for Riemann approximation.
        """
        q = self.q_proj(mem_hist[:, -1:, :])  # [B,1,H]
        k = self.k_proj(mem_hist)             # [B,t,H]
        v = self.v_proj(mem_hist)             # [B,t,H]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)  # [B,1,t]

        # Continuous integral approximation:
        # alpha(t,tau_i) ~ exp(score_i) * dt_i / sum_j exp(score_j) * dt_j
        dt_weights = dt_hist.unsqueeze(1).clamp_min(1e-6)  # [B,1,t]
        unnorm = torch.exp(scores - scores.max(dim=-1, keepdim=True).values) * dt_weights
        attn = unnorm / unnorm.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context = torch.matmul(attn, v).squeeze(1)  # [B,H]

        lambda_att = 0.5 * torch.sigmoid(self.lambda_att_raw)
        m_attn = self.norm_attn(mem_hist[:, -1, :] + lambda_att * self.o_proj(context))
        return m_attn, attn.squeeze(1)

    def forward(self, x, mask, delta_t=None, delta_ts=None):
        if delta_ts is None:
            delta_ts = delta_t
        delta_ts = self._normalize_delta_ts(delta_ts, x)

        bsz, t_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        mask_bt = normalize_sequence_mask(
            mask,
            batch_size=bsz,
            seq_len=t_len,
            dtype=dtype,
            device=device,
            strict=True,
        )

        s = torch.zeros(bsz, self.hidden_dim, device=device, dtype=dtype)
        m_reset = torch.zeros(bsz, self.hidden_dim, device=device, dtype=dtype)
        r_prev = torch.zeros(bsz, self.hidden_dim, device=device, dtype=dtype)

        local_mem_list = []
        local_dt_list = []

        recon_list, pred_list = [], []
        mem_seq, mem_reset_seq = [], []
        rate_seq, h_seq = [], []

        for t in range(t_len):
            mask_h = mask_bt[:, t].view(bsz, 1)
            s_prev = s
            m_reset_prev = m_reset
            r_prev_old = r_prev

            s_prop, m_local_pre_prop = self.cell(
                x[:, t],
                s,
                m_reset,
                r_prev,
                delta_ts[:, t],
            )
            s = mask_h * s_prop + (1.0 - mask_h) * s_prev
            m_local_pre = mask_h * m_local_pre_prop + (1.0 - mask_h) * m_reset_prev
            local_mem_list.append(m_local_pre)
            local_dt_list.append(delta_ts[:, t])

            mem_hist = torch.stack(local_mem_list, dim=1)  # [B,t+1,H]
            dt_hist = torch.stack(local_dt_list, dim=1)    # [B,t+1]
            m_pre_prop, _ = self._causal_attention(mem_hist, dt_hist)
            m_pre = mask_h * m_pre_prop + (1.0 - mask_h) * m_reset_prev

            rate_prop = torch.sigmoid(self.beta_spike * (m_pre - self.threshold))
            m_reset_prop = m_pre - self.threshold * rate_prop
            m_reset = mask_h * m_reset_prop + (1.0 - mask_h) * m_reset_prev
            r_prev = mask_h * rate_prop + (1.0 - mask_h) * r_prev_old
            rate_t = mask_h * rate_prop

            h_t = torch.cat([m_reset, rate_t], dim=-1)
            h_t = self.dropout(h_t)
            h_t = h_t * mask_h

            recon_list.append(self.fc_recon(h_t))
            pred_list.append(self.fc_pred(h_t))

            mem_seq.append(m_pre * mask_h)
            mem_reset_seq.append(m_reset * mask_h)
            rate_seq.append(rate_t)
            h_seq.append(h_t)

        recons = torch.stack(recon_list, dim=1)
        preds = torch.stack(pred_list, dim=1)
        mem_seq = torch.stack(mem_seq, dim=1)
        mem_reset_seq = torch.stack(mem_reset_seq, dim=1)
        rate_seq = torch.stack(rate_seq, dim=1)
        h_seq = torch.stack(h_seq, dim=1)

        valid_time = (mask_bt > 0).to(dtype=dtype)

        # Detection latent excludes padded time steps. For irregular/Poisson
        # views, delta_ts gives a time-aware weighted mean; for uniform data it
        # reduces to the standard masked mean because delta_ts is all ones.
        time_weights = valid_time * delta_ts.clamp_min(0.0)
        denom = time_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        z_mean = (mem_reset_seq * time_weights.unsqueeze(-1)).sum(dim=1) / denom

        return recons, preds, z_mean, mem_seq, mem_reset_seq, rate_seq, h_seq


PredictiveSNN_ODEATT_Model = ContinuousAttentionSNNODE
