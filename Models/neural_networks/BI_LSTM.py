# models/BI_LSTM_fixed.py
import torch
import torch.nn as nn

class PredictiveBiLSTM_Model(nn.Module):
    """
    ✅ Deep Causal LSTM（单向 + 深度）
    ✅ 保持原有 BiLSTM 的所有接口与变量名
    ✅ 去除未来信息泄漏（反作弊）
    ✅ 适用于异常检测
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=16,
        num_layers=1,      # ✅ 深度 ≥ 2，用深度代替双向
        dropout=0.6,
        output_dim=None,
        **kwargs,
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.event_seq_kind = "none"

        # ✅ 输入拼接 mask（保持不变）
        self.input_size = input_dim * 2

        # ✅ 关键修改点 1：单向 LSTM（因果）
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,            # ✅ 去除双向 = 去作弊
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ✅ 关键修改点 2：输出层维度匹配单向 LSTM
        self.reconstruction_layer = nn.Linear(hidden_dim, input_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    # --------------------------------------------------
    # mask 处理（保持接口与语义完全一致）
    # --------------------------------------------------
    def _normalize_mask(self, mask, x):
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        if mask is None:
            mask = torch.ones(B, T, D, device=device, dtype=dtype)

        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        if mask.shape[-1] != D:
            if mask.shape[-1] < D:
                pad = torch.ones(B, T, D - mask.shape[-1], device=device, dtype=dtype)
                mask = torch.cat([mask, pad], dim=-1)
            else:
                mask = mask[:, :, :D]

        return mask

    # --------------------------------------------------
    # delta_ts 接口占位（保持兼容）
    # --------------------------------------------------
    def _normalize_delta_ts(self, delta_ts, x):
        """接收但不使用（因果 LSTM 不需要时间间隔）"""
        return None

    # --------------------------------------------------
    # Forward（严格因果）
    # --------------------------------------------------
    def forward(self, x, mask=None, delta_t=None, delta_ts=None):
        """
        x        : [B, T, D]
        mask     : [B, T, D] | None
        delta_t  : 接收但不使用
        delta_ts : 接收但不使用
        """

        B, T, D = x.shape

        # -------- mask --------
        mask = self._normalize_mask(mask, x)

        # -------- 输入拼接（数据 + mask）--------
        data_with_mask = torch.cat([x, mask], dim=-1)  # [B, T, 2D]

        # -------- Deep Causal LSTM --------
        # ✅ h_seq 仅依赖过去与当前
        h_seq, _ = self.lstm(data_with_mask)  # [B, T, H]

        # -------- 输出 --------
        recons = self.reconstruction_layer(h_seq)  # [B, T, D]
        preds = self.output_layer(h_seq)            # [B, T, output_dim]

        # -------- latent summary --------
        z_seq = h_seq                               # [B, T, H]
        z_mean = z_seq.mean(dim=1)                  # [B, H]
        mem_seq = z_seq
        mem_reset_seq = z_seq
        aux_seq = torch.zeros_like(z_seq)
        h_out_seq = torch.cat([mem_reset_seq, aux_seq], dim=-1)

        return recons, preds, z_mean, mem_seq, mem_reset_seq, aux_seq, h_out_seq

    def get_model_info(self):
        return {
            "model_type": "PredictiveBiLSTM_CausalDeep",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional": False,   # ✅ 明确不再双向
            "time_aware": False,
            "causal": True            # ✅ 明确标注
        }
