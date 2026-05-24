import numpy as np
import torch
import torch.nn as nn


def flatten_ts(x):
    """Flatten a batch of time-series samples from [N, T, D] to [N, T*D]."""
    return np.asarray(x).reshape(np.asarray(x).shape[0], -1)


class BaseStatisticalSequenceModel(nn.Module):
    """Torch-compatible base for statistical baselines.

    The training pipeline expects the same 7-value contract as SNN_ODEATT:
    (recons, preds, z_mean, mem_seq, mem_reset_seq, aux_seq, h_seq).
    """

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.event_seq_kind = "none"
        self.is_stat_model = True

        self.register_buffer("ar_a", torch.ones(self.input_dim))
        self.register_buffer("ar_b", torch.zeros(self.input_dim))
        self.register_buffer("feat_mean", torch.zeros(self.input_dim))
        self.register_buffer("feat_std", torch.ones(self.input_dim))

        self.state_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self._init_frozen_projection()

    def _init_frozen_projection(self):
        with torch.no_grad():
            self.state_proj.weight.zero_()
            diag = min(self.input_dim, self.hidden_dim)
            self.state_proj.weight[:diag, :diag] = torch.eye(diag)
        for param in self.state_proj.parameters():
            param.requires_grad = False

    @staticmethod
    def _pad_time(tensor, target_t):
        pad_t = target_t - tensor.size(1)
        if pad_t <= 0:
            return tensor
        pad_shape = (tensor.size(0), pad_t, tensor.size(2))
        pad = tensor.new_zeros(pad_shape)
        return torch.cat([tensor, pad], dim=1)

    @torch.no_grad()
    def _collect_loader(self, train_loader):
        x_batches = []
        mask_batches = []
        max_t = 0

        for x, mask, _dt, _paths in train_loader:
            x = x.float().cpu()
            mask = mask.float().cpu()
            x_batches.append(x)
            mask_batches.append(mask)
            max_t = max(max_t, x.size(1))

        if not x_batches:
            return None, None

        x_batches = [self._pad_time(x, max_t) for x in x_batches]
        mask_batches = [self._pad_time(mask, max_t) for mask in mask_batches]
        return torch.cat(x_batches, dim=0), torch.cat(mask_batches, dim=0)

    @torch.no_grad()
    def _update_feature_stats(self, x, mask):
        denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
        mean = (x * mask).sum(dim=(0, 1)) / denom
        var = (((x - mean.view(1, 1, -1)) ** 2) * mask).sum(dim=(0, 1)) / denom
        std = var.sqrt().clamp_min(1e-6)

        self.feat_mean.copy_(mean.to(self.feat_mean.device))
        self.feat_std.copy_(std.to(self.feat_std.device))

    @torch.no_grad()
    def _fit_ar_buffers(self, x, mask, shrinkage=0.0, gain_blend=0.0, gain_clamp=None):
        self._update_feature_stats(x, mask)

        if x.size(1) < 2:
            return

        x_prev = x[:, :-1, :]
        y_next = x[:, 1:, :]
        pair_mask = (mask[:, :-1, :] * mask[:, 1:, :]).clamp_min(0.0)

        denom = pair_mask.sum(dim=(0, 1)).clamp_min(1.0)
        mx = (x_prev * pair_mask).sum(dim=(0, 1)) / denom
        my = (y_next * pair_mask).sum(dim=(0, 1)) / denom
        cov = (((x_prev - mx.view(1, 1, -1)) * (y_next - my.view(1, 1, -1))) * pair_mask).sum(dim=(0, 1)) / denom
        var = (((x_prev - mx.view(1, 1, -1)) ** 2) * pair_mask).sum(dim=(0, 1)) / denom
        a = cov / var.clamp_min(1e-6)

        if shrinkage > 0:
            a = torch.sign(a) * torch.clamp(torch.abs(a) - float(shrinkage), min=0.0)
        if gain_blend > 0:
            blend = float(gain_blend)
            a = (1.0 - blend) * a + blend * torch.tanh(a)
        if gain_clamp is not None:
            lo, hi = gain_clamp
            a = torch.clamp(a, float(lo), float(hi))

        b = my - a * mx
        self.ar_a.copy_(a.to(self.ar_a.device))
        self.ar_b.copy_(b.to(self.ar_b.device))

    def _ar_step(self, x):
        return x * self.ar_a.view(1, 1, -1) + self.ar_b.view(1, 1, -1)

    def _mask_or_ones(self, x, mask):
        if mask is None:
            return torch.ones_like(x)
        return mask.to(device=x.device, dtype=x.dtype)

    def _build_outputs(self, x, mask, recons, preds, state_source=None, aux_source=None):
        mask = self._mask_or_ones(x, mask)
        recons = recons * mask
        preds = preds * mask

        if state_source is None:
            state_source = recons

        norm_state = (state_source - self.feat_mean.view(1, 1, -1)) / self.feat_std.view(1, 1, -1)
        time_gate = mask.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
        mem_seq = torch.tanh(self.state_proj(norm_state * mask)) * time_gate
        mem_reset_seq = mem_seq

        if aux_source is None:
            aux_seq = torch.zeros_like(mem_seq)
        else:
            norm_aux = (aux_source - self.feat_mean.view(1, 1, -1)) / self.feat_std.view(1, 1, -1)
            aux_seq = torch.tanh(self.state_proj(norm_aux * mask)) * time_gate

        denom = time_gate.sum(dim=1).clamp_min(1.0)
        z_mean = (mem_reset_seq * time_gate).sum(dim=1) / denom
        h_seq = torch.cat([mem_reset_seq, aux_seq], dim=-1)
        return recons, preds, z_mean, mem_seq, mem_reset_seq, aux_seq, h_seq

    def forward(self, x, mask=None, delta_t=None, delta_ts=None):
        raise NotImplementedError
