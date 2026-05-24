import numpy as np
import torch

try:
    from ._stat_base import BaseStatisticalSequenceModel
except ImportError:
    from _stat_base import BaseStatisticalSequenceModel


class CumIntModel:
    def fit(self, x):
        cumsum = np.cumsum(x, axis=1)
        self.mean = cumsum.mean(axis=0)
        self.std = cumsum.std(axis=0) + 1e-6
        return self

    def score(self, x):
        cumsum = np.cumsum(x, axis=1)
        z = (cumsum - self.mean) / self.std
        return np.max(np.linalg.norm(z, axis=-1), axis=1)


class StatisticalCumIntModel(BaseStatisticalSequenceModel):
    def __init__(self, input_dim, hidden_dim=32, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        self.register_buffer("cum_mean", torch.zeros(1, self.input_dim))
        self.register_buffer("cum_std", torch.ones(1, self.input_dim))

    @torch.no_grad()
    def fit_statistical(self, train_loader, device=None):
        x, mask = self._collect_loader(train_loader)
        if x is None:
            return

        self._update_feature_stats(x, mask)
        self._fit_ar_buffers(x, mask, gain_clamp=(-0.5, 0.5))

        cumsum = torch.cumsum(x * mask, dim=1)
        time_mask = mask.mean(dim=-1, keepdim=True)
        denom = time_mask.sum(dim=0).clamp_min(1.0)
        mean = (cumsum * time_mask).sum(dim=0) / denom
        var = (((cumsum - mean.view(1, mean.size(0), mean.size(1))) ** 2) * time_mask).sum(dim=0) / denom
        std = var.sqrt().clamp_min(1e-6)

        target_device = self.cum_mean.device
        self.cum_mean = mean.to(target_device)
        self.cum_std = std.to(target_device)

    def _stats_for_length(self, t_len):
        if self.cum_mean.size(0) == t_len:
            return self.cum_mean, self.cum_std
        if self.cum_mean.size(0) > t_len:
            return self.cum_mean[:t_len], self.cum_std[:t_len]

        pad_t = t_len - self.cum_mean.size(0)
        mean_pad = self.cum_mean[-1:].expand(pad_t, -1)
        std_pad = self.cum_std[-1:].expand(pad_t, -1)
        return torch.cat([self.cum_mean, mean_pad], dim=0), torch.cat([self.cum_std, std_pad], dim=0)

    def forward(self, x, mask=None, delta_t=None, delta_ts=None):
        x = x.float()
        mask = self._mask_or_ones(x, mask)
        cumsum = torch.cumsum(x * mask, dim=1)
        cum_mean, cum_std = self._stats_for_length(x.size(1))
        z = (cumsum - cum_mean.view(1, x.size(1), -1)) / cum_std.view(1, x.size(1), -1)

        step = self._ar_step(x)
        recons = x
        preds = step
        return self._build_outputs(x, mask, recons, preds, state_source=z)


PredictiveStatCumInt_Model = StatisticalCumIntModel
