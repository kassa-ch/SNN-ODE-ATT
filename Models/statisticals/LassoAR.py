import numpy as np

try:
    from ._stat_base import BaseStatisticalSequenceModel, flatten_ts
except ImportError:
    from _stat_base import BaseStatisticalSequenceModel, flatten_ts


class LassoAR:
    def __init__(self, alpha=1e-3):
        self.alpha = alpha

    def fit(self, x):
        from sklearn.linear_model import Lasso

        z = flatten_ts(x)
        self.model = Lasso(alpha=self.alpha, max_iter=5000)
        self.model.fit(z[:, :-1], z[:, 1:])
        return self

    def score(self, x):
        z = flatten_ts(x)
        pred = self.model.predict(z[:, :-1])
        return np.mean((z[:, 1:] - pred) ** 2, axis=1)


class StatisticalLassoAR(BaseStatisticalSequenceModel):
    def __init__(self, input_dim, hidden_dim=32, alpha=1e-3, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        self.alpha = float(alpha)

    def fit_statistical(self, train_loader, device=None):
        x, mask = self._collect_loader(train_loader)
        if x is None:
            return
        self._fit_ar_buffers(x, mask, shrinkage=self.alpha)

    def forward(self, x, mask=None, delta_t=None, delta_ts=None):
        x = x.float()
        mask = self._mask_or_ones(x, mask)
        step = self._ar_step(x)

        recons = 0.5 * x + 0.5 * step
        preds = step
        state_source = x - step
        return self._build_outputs(x, mask, recons, preds, state_source=state_source)


PredictiveStatLassoAR_Model = StatisticalLassoAR
