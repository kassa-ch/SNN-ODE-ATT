# models/RNN_GRU_fixed.py
import torch
import torch.nn as nn

class PredictiveRNN_Model(nn.Module):
    """
    ✅ 传统GRU（完全忽略时间间隔）
    ✅ 接口与原来完全一致（兼容调用）
    ✅ 删除所有时间调制逻辑
    """

    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.event_seq_kind = "none"

        # --------------------------------------------------
        # 标准GRU（无时间感知）
        # --------------------------------------------------
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout_rate
        )

        # --------------------------------------------------
        # 删除时间调制层
        # --------------------------------------------------

        # --------------------------------------------------
        # 解码器
        # --------------------------------------------------
        self.fc_recon = nn.Linear(hidden_dim, input_dim)
        self.fc_pred = nn.Linear(hidden_dim, input_dim)

        self.dropout = nn.Dropout(dropout_rate)

        print(
            f"[INFO] Standard GRU (No Time) | "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}"
        )

    # --------------------------------------------------
    # 简化的时间处理（仅为了接口兼容）
    # --------------------------------------------------
    def _normalize_delta_ts(self, delta_ts, x):
        """仅为了保持参数存在，不做实际计算"""
        # 返回None或占位符，明确表示忽略
        return None

    # --------------------------------------------------
    # Forward（忽略delta_ts）
    # --------------------------------------------------
    def forward(self, x, mask, delta_t=None, delta_ts=None):
        """
        x        : [B, T, D]
        mask     : [B, T, D]
        delta_t  : 接收但不使用
        delta_ts : 接收但不使用
        """

        B, T, D = x.size()

        # --------------------------------------------------
        # GRU 编码（完全忽略时间间隔）
        # --------------------------------------------------
        z_seq_latent, _ = self.rnn(x)   # [B, T, H]
        z_seq_latent = self.dropout(z_seq_latent)

        # --------------------------------------------------
        # 输出（无时间调制）
        # --------------------------------------------------
        recons = self.fc_recon(z_seq_latent) * mask
        preds = self.fc_pred(z_seq_latent) * mask

        z_mean = z_seq_latent.mean(dim=1)

        # 返回格式完全一致
        mem_seq = z_seq_latent
        mem_reset_seq = z_seq_latent
        aux_seq = torch.zeros_like(z_seq_latent)
        h_seq = torch.cat([mem_reset_seq, aux_seq], dim=-1)
        return recons, preds, z_mean, mem_seq, mem_reset_seq, aux_seq, h_seq

    def get_model_info(self):
        return {
            "model_type": "PredictiveRNN_Standard",
            "rnn_type": "GRU",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "time_aware": False,  # 明确标注
            "explicit_ode": False
        }
