import numpy as np
import torch

try:
    from ._stat_base import BaseStatisticalSequenceModel, flatten_ts
except ImportError:
    from _stat_base import BaseStatisticalSequenceModel, flatten_ts


class KRRModel:
    def __init__(self, alpha=1e-2):
        self.alpha = alpha

    def fit(self, x):
        from sklearn.kernel_ridge import KernelRidge

        z = flatten_ts(x)
        self.model = KernelRidge(kernel="rbf", alpha=self.alpha)
        self.model.fit(z[:, :-1], z[:, 1:])
        return self

    def score(self, x):
        z = flatten_ts(x)
        pred = self.model.predict(z[:, :-1])
        return np.mean((z[:, 1:] - pred) ** 2, axis=1)


class StatisticalKRRModel(BaseStatisticalSequenceModel):
    def __init__(self, input_dim, hidden_dim=32, alpha=1e-2, gamma=None, max_anchors=512, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        self.alpha = float(alpha)
        self.gamma = (1.0 / self.input_dim) if gamma is None else float(gamma)
        self.max_anchors = int(max_anchors)
        self.register_buffer("anchor_x", torch.zeros(1, self.input_dim))
        self.register_buffer("dual_coef", torch.zeros(1, self.input_dim))

    @torch.no_grad()
    def fit_statistical(self, train_loader, device=None):
        x, mask = self._collect_loader(train_loader)
        if x is None:
            return

        self._update_feature_stats(x, mask)
        self._fit_ar_buffers(x, mask, gain_blend=0.2)
        if x.size(1) < 2:
            return

        pair_mask = (mask[:, :-1, :] * mask[:, 1:, :]).mean(dim=-1) > 0.5
        x_prev = x[:, :-1, :][pair_mask]
        y_next = x[:, 1:, :][pair_mask]

        if x_prev.numel() == 0:
            return

        if x_prev.size(0) > self.max_anchors:
            indices = torch.linspace(0, x_prev.size(0) - 1, steps=self.max_anchors).long()
            x_prev = x_prev[indices]
            y_next = y_next[indices]

        try:
            from sklearn.kernel_ridge import KernelRidge
        except ImportError:
            return

        model = KernelRidge(kernel="rbf", alpha=self.alpha, gamma=self.gamma)
        model.fit(x_prev.numpy(), y_next.numpy())

        target_device = self.anchor_x.device
        self.anchor_x = torch.as_tensor(model.X_fit_, dtype=torch.float32, device=target_device)
        self.dual_coef = torch.as_tensor(model.dual_coef_, dtype=torch.float32, device=target_device)

    def _kernel_step(self, x):
        flat = x.reshape(-1, self.input_dim)
        dist_sq = torch.cdist(flat, self.anchor_x) ** 2
        kernel = torch.exp(-self.gamma * dist_sq)
        pred = kernel.matmul(self.dual_coef)
        return pred.reshape_as(x)

    def forward(self, x, mask=None, delta_t=None, delta_ts=None):
        x = x.float()
        mask = self._mask_or_ones(x, mask)

        if self.anchor_x.size(0) <= 1 and torch.allclose(self.dual_coef, torch.zeros_like(self.dual_coef)):
            step = self._ar_step(x)
        else:
            step = self._kernel_step(x)

        recons = 0.5 * x + 0.5 * step
        preds = step
        state_source = x - step
        return self._build_outputs(x, mask, recons, preds, state_source=state_source)


PredictiveStatKRR_Model = StatisticalKRRModel
