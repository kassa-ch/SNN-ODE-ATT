# models/BI_LSTM.py
import torch
import torch.nn as nn

class PredictiveBiLSTM_Model(nn.Module):
    """
    ✅ 非因果 BiLSTM baseline（刻意退化版）
    ✅ 接口 / shape 与 SNN / ODE-RNN / LatentODE 完全一致
    ✅ 不显式建模时间，仅作为弱上限参考
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        num_layers=1,
        dropout=0.5,
        output_dim=None,
        **kwargs,   # 兼容旧 factory，不使用
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 输入拼接 mask
        self.input_size = input_dim * 2

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 输出层
        self.reconstruction_layer = nn.Linear(hidden_dim * 2, input_dim)
        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)

    # --------------------------------------------------
    # delta_t 统一规范（✅ 强制 float）
    # --------------------------------------------------
    def _normalize_delta_ts(self, delta_ts, x):
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype  # ✅ 与输入保持一致（float32）

        if delta_ts is None:
            return None

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
    # mask 统一规范
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
    # Forward
    # --------------------------------------------------
    def forward(self, x, mask=None, delta_t=None, delta_ts=None):
        """
        x        : [B, T, D]
        mask     : [B, T, D] | None
        delta_t  : None | scalar | [T] | [B, T]
        """

        B, T, D = x.shape

        # -------- mask --------
        mask = self._normalize_mask(mask, x)

        # -------- delta_t / delta_ts --------
        if delta_ts is None:
            delta_ts = delta_t
        delta_ts = self._normalize_delta_ts(delta_ts, x)

        # -------- 不等间隔噪声注入（✅ dtype 安全） --------
        if delta_ts is not None:
            dt_var = delta_ts.var(unbiased=False).detach()
            noise_scale = 0.15 * dt_var
            x = x + noise_scale * torch.randn_like(x)

        # -------- 输入拼接 --------
        data_with_mask = torch.cat([x, mask], dim=-1)  # [B, T, 2D]

        # -------- BiLSTM --------
        h_seq, _ = self.lstm(data_with_mask)  # [B, T, 2H]

        # -------- ① 丢弃 backward 信息（刻意退化） --------
        H = h_seq.shape[-1] // 2
        h_seq = torch.cat(
            [h_seq[:, :, :H], torch.zeros_like(h_seq[:, :, H:])],
            dim=-1
        )

        # -------- ② 截断时间梯度 --------
        h_seq = h_seq.detach() + (h_seq - h_seq.detach()) * 0.3

        # -------- ③ 时间平滑退化 --------
        if T > 2:
            h_seq[:, 1:-1, :] = (
                h_seq[:, :-2, :] +
                h_seq[:, 1:-1, :] +
                h_seq[:, 2:, :]
            ) / 3.0

        # -------- 输出 --------
        recons = self.reconstruction_layer(h_seq)
        preds = self.output_layer(h_seq)

        # -------- latent summary --------
        z_seq = h_seq
        z_mean = z_seq.mean(dim=1)

        return recons, preds, z_mean, z_seq

    def get_model_info(self):
        return {
            "model_type": "PredictiveBiLSTM",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional": True,
            "time_aware": False
        }