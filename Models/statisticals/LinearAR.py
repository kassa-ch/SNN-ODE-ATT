import numpy as np

try:
    from ._stat_base import BaseStatisticalSequenceModel, flatten_ts
except ImportError:
    from _stat_base import BaseStatisticalSequenceModel, flatten_ts


class LinearAR:
    def fit(self, x):
        from sklearn.linear_model import LinearRegression

        z = flatten_ts(x)
        self.model = LinearRegression()
        self.model.fit(z[:, :-1], z[:, 1:])
        return self

    def score(self, x):
        z = flatten_ts(x)
        pred = self.model.predict(z[:, :-1])
        return np.mean((z[:, 1:] - pred) ** 2, axis=1)


class StatisticalLinearAR(BaseStatisticalSequenceModel):
    def fit_statistical(self, train_loader, device=None):
        x, mask = self._collect_loader(train_loader)
        if x is None:
            return
        self._fit_ar_buffers(x, mask)

    def forward(self, x, mask=None, delta_t=None, delta_ts=None):
        x = x.float()
        mask = self._mask_or_ones(x, mask)
        step = self._ar_step(x)

        recons = 0.5 * x + 0.5 * step
        preds = step
        state_source = x - step
        return self._build_outputs(x, mask, recons, preds, state_source=state_source)


PredictiveStatLinearAR_Model = StatisticalLinearAR
