import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

# ============================================================
# Simple Time GP
# ============================================================
class SimpleTimeGP(nn.Module):
    """
    轻量时间 GP（仅时间轴）
    """
    def __init__(self, length_scale=1.0, noise=1e-4):
        super().__init__()
        self.length_scale = length_scale
        self.noise = noise

    def rbf_kernel(self, t1, t2):
        diff = t1.unsqueeze(1) - t2.unsqueeze(0)
        return torch.exp(-0.5 * diff.pow(2) / (self.length_scale ** 2))

    def forward(self, x, times):
        """
        x     : [B, T, D]
        times : [B, T]
        """
        B, T, D = x.shape
        device = x.device

        x_mu = torch.zeros_like(x)
        x_var = torch.zeros_like(x)

        for b in range(B):
            t = times[b]
            K = self.rbf_kernel(t, t)
            K = K + self.noise * torch.eye(T, device=device)
            K_inv = torch.inverse(K)

            for d in range(D):
                y = x[b, :, d]
                mu = K @ (K_inv @ y)
                cov = K - K @ K_inv @ K
                x_mu[b, :, d] = mu
                x_var[b, :, d] = torch.diag(cov)

        return x_mu, x_var

# ============================================================
# Predictive DF²M
# ============================================================
class PredictiveDF2M_Model(PyroModule):
    """
    ✅ DF²M + Optional Time-GP
    ✅ 接口与 SNN / ODE / RNN / LatentODE 完全一致
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        latent_dim=16,
        num_clusters=6,
        dropout_rate=0.6,
        output_dim=None,
        use_gp=True,
        gp_length_scale=1.0,
        device=None,
        **kwargs
    ):
        super().__init__()

        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.event_seq_kind = "none"

        # -------- GP --------
        self.use_gp = use_gp
        if self.use_gp:
            self.gp = SimpleTimeGP(length_scale=gp_length_scale)

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 10),
            nn.ReLU(),
        )

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

        # -------- Predictor --------
        self.predictor_lstm = nn.LSTM(
            input_size=10,
            hidden_size=10,
            num_layers=1,
            batch_first=True,
            dropout=dropout_rate
        )
        self.pred_fc = nn.Linear(10, output_dim)

        # -------- Temporal Model --------
        self.temporal_model = nn.GRU(
            input_size=10,
            hidden_size=10,
            num_layers=1,
            batch_first=True,
            dropout=dropout_rate
        )

        # -------- DF²M parameters --------
        self.z_logits = PyroSample(
            dist.Normal(0, 1)
            .expand([num_clusters, latent_dim])
            .to_event(2)
        )
        self.A = PyroSample(
            dist.Normal(0, 1)
            .expand([num_clusters, latent_dim])
            .to_event(2)
        )

        self._init_weights()

    # --------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # --------------------------------------------------
    def _normalize_delta_ts(self, delta_ts, x):
        B, T, _ = x.shape
        device = x.device

        if delta_ts is None:
            return None

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
    def encode(self, x, times=None):
        if self.use_gp and isinstance(times, torch.Tensor):
            x, _ = self.gp(x, times)

        B, T, _ = x.shape
        z_seq = []

        for t in range(T):
            z_seq.append(self.encoder(x[:, t, :]))

        return torch.stack(z_seq, dim=1)

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

        # -------- mask --------
        x = x * mask

        # -------- times --------
        times = None
        if delta_ts is not None:
            times = torch.cumsum(delta_ts, dim=1)
            times = times - times[:, :1]

        # -------- encode --------
        z = self.encode(x, times)

        # -------- temporal --------
        z_proc, _ = self.temporal_model(z)

        # -------- decode --------
        recon_seq = torch.stack(
            [self.decoder(z_proc[:, t]) for t in range(z_proc.size(1))],
            dim=1
        )

        # -------- predict --------
        lstm_out, _ = self.predictor_lstm(z_proc)
        pred_seq = torch.stack(
            [self.pred_fc(lstm_out[:, t]) for t in range(lstm_out.size(1))],
            dim=1
        )

        z_mean = z_proc.mean(dim=1)
        mem_seq = z_proc
        mem_reset_seq = z_proc
        aux_seq = torch.zeros_like(z_proc)
        h_seq = torch.cat([mem_reset_seq, aux_seq], dim=-1)

        return recon_seq, pred_seq, z_mean, mem_seq, mem_reset_seq, aux_seq, h_seq

    # --------------------------------------------------
    def get_model_info(self):
        return {
            "model_type": "PredictiveDF2M",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "time_aware": self.use_gp,
            "explicit_ode": False
        }
