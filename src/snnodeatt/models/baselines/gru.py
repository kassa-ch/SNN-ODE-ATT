# models/RNN_GRU.py
import torch
import torch.nn as nn

class PredictiveRNN_Model(nn.Module):
    """
    ✅ 公平 ODE-RNN 对照模型（Time-aware GRU）
    ✅ 接口与 SNN / SNN-ODE / LatentODE 完全一致
    ✅ 不显式解 ODE，仅通过 delta_t 调制状态演化
    """

    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # --------------------------------------------------
        # GRU（离散状态更新）
        # --------------------------------------------------
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout_rate
        )

        # --------------------------------------------------
        # 时间调制层（Δt → 演化强度）
        # --------------------------------------------------
        self.time_gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Sigmoid()
        )

        # --------------------------------------------------
        # 解码器
        # --------------------------------------------------
        self.fc_recon = nn.Linear(hidden_dim, input_dim)
        self.fc_pred = nn.Linear(hidden_dim, input_dim)

        self.dropout = nn.Dropout(dropout_rate)

        print(
            f"[INFO] Fair ODE-RNN (GRU) initialized | "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}"
        )

    # --------------------------------------------------
    # delta_t 统一规范（✅ 强制 float）
    # --------------------------------------------------
    def _normalize_delta_ts(self, delta_ts, x):
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype  # ✅ 与输入 x 对齐（float32 / float64）

        if delta_ts is None:
            raise ValueError("PredictiveRNN_Model requires delta_t or delta_ts")

        if isinstance(delta_ts, (int, float)):
            delta_ts = torch.tensor(delta_ts, device=device, dtype=dtype)

        if isinstance(delta_ts, torch.Tensor):
            delta_ts = delta_ts.to(device=device, dtype=dtype)
        else:
            raise TypeError(f"delta_ts 类型非法: {type(delta_ts)}")

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

        # --------------------------------------------------
        # GRU 编码
        # --------------------------------------------------
        z_seq_latent, _ = self.rnn(x)   # [B, T, H]

        # --------------------------------------------------
        # 时间调制（ODE-RNN 核心思想）
        # --------------------------------------------------
        z_seq_modulated = []

        for t in range(T):
            h_t = z_seq_latent[:, t, :]          # [B, H]
            dt_t = delta_ts[:, t].unsqueeze(-1) # [B, 1] ✅ float
            gamma_t = self.time_gate(dt_t)      # [B, H]
            z_seq_modulated.append(gamma_t * h_t)

        z_seq_modulated = torch.stack(z_seq_modulated, dim=1)
        z_seq_modulated = self.dropout(z_seq_modulated)

        # --------------------------------------------------
        # 输出（mask 与其他模型语义一致）
        # --------------------------------------------------
        recons = self.fc_recon(z_seq_modulated) * mask
        preds = self.fc_pred(z_seq_modulated) * mask

        z_mean = z_seq_modulated.mean(dim=1)

        return recons, preds, z_mean, z_seq_modulated

    def get_model_info(self):
        return {
            "model_type": "PredictiveRNN_Fair",
            "rnn_type": "GRU",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "time_aware": True,
            "explicit_ode": False
        }