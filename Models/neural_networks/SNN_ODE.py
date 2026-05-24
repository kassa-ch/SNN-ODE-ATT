# C:\Users\kassa\Desktop\半导体晶圆项目ATT\models\SNN_ODE.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ODEFunc(nn.Module):
    """
    f_ODE(M) = LN(W5 * tanh(W4 * M + b4) + b5)
    This term models endogenous membrane dynamics, not direct input dynamics.
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


class SNN_ODE_Cell(nn.Module):
    """
    Continuous-time state definition (theory):
    - S(t): synaptic state
    - M_pre(t): pre-reset membrane state
    - R(t): continuous firing rate (soft spike)
    - M*(t): post-reset membrane state
    - H(t) = [M*(t), R(t)]

    Numerical implementation:
    - Uses delta_t as quadrature step on observed sampling grid.
    - The equations are continuous-time; this cell performs one-step numerical evaluation.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        threshold,
        tau_mem,
        tau_syn_base,
        dropout_rate=0.0,
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

        # dS/dt = -Gamma_s * S + Ux X + Um M* + b_s
        self.in_to_syn = nn.Linear(input_dim, hidden_dim)
        self.mem_to_syn = nn.Linear(hidden_dim, hidden_dim)

        # tau_s(X, M*) = clip(tau_syn_base * (1 + rho), tau_min, tau_max)
        self.tau_mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.ode_func = ODEFunc(hidden_dim)

        # dM_pre/dt = -Gamma_m*M + U_s*S + U_o*f_ODE(M) - theta*R + b_m
        self.syn_to_mem = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ode_to_mem = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mem_bias = nn.Parameter(torch.zeros(hidden_dim))

        self.norm_syn = nn.LayerNorm(hidden_dim)
        self.norm_mem = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_t, s_t, m_prev_reset, r_prev, mask_t, delta_t):
        """
        Args:
            x_t: [B, D]
            s_t: [B, H]        numeric sample of S(t_{k-1})
            m_prev_reset: [B, H] numeric sample of M*(t_{k-1})
            r_prev: [B, H]     numeric sample of R(t_{k-1})
            mask_t: [B, D] or [B, 1]
            delta_t: [B]

        Returns:
            s_next: [B, H]        sample of S(t_k)
            m_reset: [B, H]       sample of M*(t_k)
            h_t: [B, 2H]          sample of H(t_k) = [M*(t_k), R(t_k)]
            rate_t: [B, H]        sample of R(t_k)
            m_pre: [B, H]         sample of M_pre(t_k)
        """
        bsz = x_t.size(0)
        dt = delta_t.view(bsz, 1).to(dtype=x_t.dtype).clamp_min(1e-6)

        # Continuous state-dependent synaptic memory:
        # rho(X, M*) = softplus(g_tau([X, M*]))
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

        # RK4 evaluates continuous ODE term on membrane state trajectory.
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
        m_pre = self.norm_mem(m_prev_reset + dt * d_m_dt)

        # Continuous firing rate and soft reset.
        rate_t = torch.sigmoid(self.beta_spike * (m_pre - self.threshold))
        m_reset = m_pre - self.threshold * rate_t

        h_t = torch.cat([m_reset, rate_t], dim=-1)
        h_t = self.dropout(h_t)

        if mask_t.dim() == 2:
            mask_gate = mask_t.mean(dim=-1, keepdim=True)
        else:
            mask_gate = mask_t
        h_t = h_t * mask_gate

        return s_next, m_reset, h_t, rate_t, m_pre


class PredictiveSNN_ODE_Model(nn.Module):
    """
    Unified output contract for training/demo compatibility:
    (recons, preds, z_mean, mem_seq, mem_reset_seq, rate_seq, h_seq)

    Variable semantics:
    - mem_seq: M_pre(t_k)
    - mem_reset_seq: M*(t_k)
    - rate_seq: R(t_k)
    - z_mean: TimePool(M*(t_k)), consistent with demo feature input.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        threshold,
        tau_mem,
        tau_syn=None,
        tau_syn_base=None,
        dropout_rate=0.0,
        beta_spike=10.0,
        **kwargs,
    ):
        super().__init__()

        if tau_syn_base is None:
            if tau_syn is None:
                raise ValueError("必须提供 tau_syn 或 tau_syn_base")
            tau_syn_base = tau_syn

        self.event_seq_kind = "rate"

        self.cell = SNN_ODE_Cell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            threshold=threshold,
            tau_mem=tau_mem,
            tau_syn_base=tau_syn_base,
            dropout_rate=dropout_rate,
            beta_spike=beta_spike,
        )

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

    def forward(self, x, mask, delta_t=None, delta_ts=None):
        if delta_ts is None:
            delta_ts = delta_t
        delta_ts = self._normalize_delta_ts(delta_ts, x)

        bsz, t_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        s = torch.zeros(bsz, self.cell.hidden_dim, device=device, dtype=dtype)
        m_reset = torch.zeros(bsz, self.cell.hidden_dim, device=device, dtype=dtype)
        r_prev = torch.zeros(bsz, self.cell.hidden_dim, device=device, dtype=dtype)

        recons, preds = [], []
        mem_seq, mem_reset_seq = [], []
        rate_seq, h_seq = [], []

        for t in range(t_len):
            s, m_reset, h_t, rate_t, m_pre = self.cell(
                x[:, t],
                s,
                m_reset,
                r_prev,
                mask[:, t],
                delta_ts[:, t],
            )
            r_prev = rate_t

            recons.append(self.fc_recon(h_t))
            preds.append(self.fc_pred(h_t))

            mem_seq.append(m_pre)
            mem_reset_seq.append(m_reset)
            rate_seq.append(rate_t)
            h_seq.append(h_t)

        recons = torch.stack(recons, dim=1)
        preds = torch.stack(preds, dim=1)
        mem_seq = torch.stack(mem_seq, dim=1)
        mem_reset_seq = torch.stack(mem_reset_seq, dim=1)
        rate_seq = torch.stack(rate_seq, dim=1)
        h_seq = torch.stack(h_seq, dim=1)

        z_mean = mem_reset_seq.mean(dim=1)

        return recons, preds, z_mean, mem_seq, mem_reset_seq, rate_seq, h_seq
