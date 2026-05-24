import numpy as np
import torch

try:
    from ._stat_base import BaseStatisticalSequenceModel, flatten_ts
except ImportError:
    from _stat_base import BaseStatisticalSequenceModel, flatten_ts


class FPCAModel:
    def __init__(self, var_ratio=0.95):
        self.var_ratio = var_ratio

    def fit(self, x):
        from sklearn.decomposition import PCA

        z = flatten_ts(x)
        self.pca = PCA(n_components=self.var_ratio, svd_solver="full")
        self.pca.fit(z)
        return self

    def score(self, x):
        z = flatten_ts(x)
        z_rec = self.pca.inverse_transform(self.pca.transform(z))
        residual = z - z_rec
        return np.mean(residual ** 2, axis=1)


class StatisticalFPCAModel(BaseStatisticalSequenceModel):
    def __init__(self, input_dim, hidden_dim=32, var_ratio=0.95, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        self.var_ratio = var_ratio
        self.register_buffer("pca_mean", torch.zeros(self.input_dim))
        self.register_buffer("pca_components", torch.empty(0, self.input_dim))

    @torch.no_grad()
    def fit_statistical(self, train_loader, device=None):
        x, mask = self._collect_loader(train_loader)
        if x is None:
            return

        self._update_feature_stats(x, mask)
        self._fit_ar_buffers(x, mask, gain_clamp=(-0.95, 0.95))

        valid = mask.mean(dim=-1) > 0.5
        points = x[valid]
        if points.size(0) < 2:
            return

        max_components = min(points.size(0), self.input_dim)
        n_components = self.var_ratio
        if isinstance(n_components, float) and n_components >= 1.0:
            n_components = max_components

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            return

        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(points.numpy())

        target_device = self.pca_mean.device
        self.pca_mean = torch.as_tensor(pca.mean_, dtype=torch.float32, device=target_device)
        self.pca_components = torch.as_tensor(pca.components_, dtype=torch.float32, device=target_device)

    def _pca_reconstruct(self, x):
        if self.pca_components.numel() == 0:
            return x
        flat = x.reshape(-1, self.input_dim)
        centered = flat - self.pca_mean.view(1, -1)
        coeff = centered.matmul(self.pca_components.t())
        rec = coeff.matmul(self.pca_components) + self.pca_mean.view(1, -1)
        return rec.reshape_as(x)

    def forward(self, x, mask=None, delta_t=None, delta_ts=None):
        x = x.float()
        mask = self._mask_or_ones(x, mask)
        recons = self._pca_reconstruct(x)
        preds = self._ar_step(recons)
        state_source = x - recons
        return self._build_outputs(x, mask, recons, preds, state_source=state_source)


PredictiveStatFPCA_Model = StatisticalFPCAModel
