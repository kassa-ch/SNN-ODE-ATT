"""
Unified distance / anomaly-score calculators for SNN-ODE-ATT demo1.

All calculators expose the same API used by demo1 Mahalanobis logic:

    calc = DistanceCalculator(**kwargs)
    calc.fit(train_features)
    scores = calc.calculate_distance(test_features)

Optional trajectory-aware usage:

    calc.fit(train_features, train_sequences=train_z_seq, mask=train_mask, time=train_time)
    scores = calc.calculate_distance(test_features, sequences=test_z_seq, mask=test_mask, time=test_time)

Input conventions:
- train_features / features: [N, H], usually z_mean or global hidden representation.
- train_sequences / sequences: [N, T, H], usually mem_reset_seq or hidden trajectory.
- mask: [N, T], 1 for valid timesteps, 0 for padding.
- time: [N, T] or delta_t [N, T]. For ODE residual and Sobolev-like scores.

No labels are used in fit(); fit() must receive train-normal data only to avoid leakage.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, Dict, Any, Tuple


Array = np.ndarray


def _as_float_array(x: Any, name: str = "array") -> Array:
    if x is None:
        raise ValueError(f"{name} is None")
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


def _as_2d(x: Any, name: str = "features") -> Array:
    x = _as_float_array(x, name)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"{name} must be [N,H], got shape {x.shape}")
    return x


def _as_3d(x: Any, name: str = "sequences") -> Array:
    x = _as_float_array(x, name)
    if x.ndim == 2:
        x = x.reshape(1, x.shape[0], x.shape[1])
    if x.ndim != 3:
        raise ValueError(f"{name} must be [N,T,H], got shape {x.shape}")
    return x


def _mask_or_ones(sequences: Array, mask: Optional[Any]) -> Array:
    n, t, _ = sequences.shape
    if mask is None:
        return np.ones((n, t), dtype=np.float64)
    m = _as_float_array(mask, "mask")
    if m.ndim == 1:
        m = m.reshape(1, -1)
    if m.shape != (n, t):
        raise ValueError(f"mask shape must be {(n, t)}, got {m.shape}")
    return (m > 0).astype(np.float64)


def _time_or_unit(sequences: Array, time: Optional[Any]) -> Array:
    n, t, _ = sequences.shape
    if time is None:
        return np.tile(np.arange(t, dtype=np.float64), (n, 1))
    tm = _as_float_array(time, "time")
    if tm.ndim == 1:
        tm = np.tile(tm.reshape(1, -1), (n, 1))
    if tm.shape != (n, t):
        raise ValueError(f"time shape must be {(n, t)}, got {tm.shape}")
    return tm


def _safe_cov(x: Array, reg: float = 1e-6) -> Array:
    x = _as_2d(x)
    if x.shape[0] <= 1:
        return np.eye(x.shape[1], dtype=np.float64) * reg
    cov = np.cov(x, rowvar=False)
    cov = np.atleast_2d(cov)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    return cov + np.eye(cov.shape[0], dtype=np.float64) * reg


def _pinv(cov: Array, reg: float = 1e-6) -> Array:
    cov = np.atleast_2d(cov)
    cov = cov + np.eye(cov.shape[0], dtype=np.float64) * reg
    return np.linalg.pinv(cov)


def _sqrtm_psd(a: Array, eps: float = 1e-9) -> Array:
    a = 0.5 * (a + a.T)
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _logm_psd(a: Array, eps: float = 1e-9) -> Array:
    a = 0.5 * (a + a.T)
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, eps, None)
    return (vecs * np.log(vals)) @ vecs.T


def _matrix_trace(a: Array) -> float:
    return float(np.real(np.trace(a)))


def _valid_points(sequences: Array, mask: Optional[Any] = None) -> Array:
    seq = _as_3d(sequences)
    m = _mask_or_ones(seq, mask)
    pts = seq[m > 0]
    if pts.size == 0:
        return seq.reshape(-1, seq.shape[-1])
    return pts.reshape(-1, seq.shape[-1])


def _sample_points(seq_i: Array, mask_i: Optional[Array] = None) -> Array:
    seq_i = _as_float_array(seq_i)
    if seq_i.ndim != 2:
        raise ValueError(f"single sequence must be [T,H], got {seq_i.shape}")
    if mask_i is None:
        return seq_i
    mi = np.asarray(mask_i).reshape(-1)
    pts = seq_i[mi > 0]
    return pts if len(pts) else seq_i


def _row_norm2(x: Array) -> Array:
    return np.sum(x * x, axis=-1)


def _pairwise_sq_dists(x: Array, y: Array) -> Array:
    x2 = _row_norm2(x)[:, None]
    y2 = _row_norm2(y)[None, :]
    d = x2 + y2 - 2.0 * (x @ y.T)
    return np.maximum(d, 0.0)


class SimpleStandardScaler:
    """Small dependency-free StandardScaler."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[Array] = None
        self.scale_: Optional[Array] = None

    def fit(self, x: Array) -> "SimpleStandardScaler":
        x = _as_2d(x)
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_ = np.where(self.scale_ < self.eps, 1.0, self.scale_)
        return self

    def transform(self, x: Array) -> Array:
        x = _as_float_array(x)
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler is not fitted")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: Array) -> Array:
        return self.fit(x).transform(x)


class BaseDistanceCalculator:
    name = "base"

    def __init__(
        self,
        standardize: bool = True,
        reg: float = 1e-6,
        eps: float = 1e-9,
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.standardize = standardize
        self.reg = reg
        self.eps = eps
        self.random_state = random_state
        self.kwargs = dict(kwargs)
        self.scaler = SimpleStandardScaler()
        self.is_fitted = False
        self.train_features_: Optional[Array] = None
        self.train_sequences_: Optional[Array] = None
        self.train_mask_: Optional[Array] = None
        self.train_time_: Optional[Array] = None

    def _scale_features_fit(self, x: Array) -> Array:
        if self.standardize:
            return self.scaler.fit_transform(x)
        self.scaler.fit(np.zeros_like(x))
        return x

    def _scale_features(self, x: Array) -> Array:
        if self.standardize:
            return self.scaler.transform(x)
        return x

    def _scale_sequences(self, seq: Array) -> Array:
        if not self.standardize:
            return seq
        shape = seq.shape
        flat = seq.reshape(-1, shape[-1])
        flat_scaled = self.scaler.transform(flat)
        return flat_scaled.reshape(shape)

    def fit(
        self,
        train_features: Any,
        train_sequences: Optional[Any] = None,
        mask: Optional[Any] = None,
        time: Optional[Any] = None,
        **kwargs: Any,
    ) -> "BaseDistanceCalculator":
        z = _as_2d(train_features, "train_features")
        z = self._scale_features_fit(z)
        self.train_features_ = z

        if train_sequences is not None:
            seq = _as_3d(train_sequences, "train_sequences")
            self.train_sequences_ = self._scale_sequences(seq)
            self.train_mask_ = _mask_or_ones(seq, mask)
            self.train_time_ = _time_or_unit(seq, time)

        self._fit_impl(z)
        self.is_fitted = True
        return self

    def _fit_impl(self, z: Array) -> None:
        raise NotImplementedError

    def calculate_distance(
        self,
        features: Any,
        sequences: Optional[Any] = None,
        mask: Optional[Any] = None,
        time: Optional[Any] = None,
        **kwargs: Any,
    ) -> Array:
        if not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} is not fitted")
        z = _as_2d(features, "features")
        z = self._scale_features(z)

        seq = None
        m = None
        tm = None
        if sequences is not None:
            seq_raw = _as_3d(sequences, "sequences")
            seq = self._scale_sequences(seq_raw)
            m = _mask_or_ones(seq_raw, mask)
            tm = _time_or_unit(seq_raw, time)

        scores = self._score_impl(z, seq, m, tm)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        return np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=0.0)

    def score(self, *args: Any, **kwargs: Any) -> Array:
        return self.calculate_distance(*args, **kwargs)

    def _score_impl(
        self,
        z: Array,
        seq: Optional[Array],
        mask: Optional[Array],
        time: Optional[Array],
    ) -> Array:
        raise NotImplementedError


class EuclideanDistanceCalculator(BaseDistanceCalculator):
    """
    S_Euc(x) = || z_x - mu_N ||_2
    Baseline point-information deviation score.
    """

    name = "euclidean"

    def _fit_impl(self, z: Array) -> None:
        self.mu_ = np.mean(z, axis=0)

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        return np.linalg.norm(z - self.mu_, axis=1)


class MahalanobisDistanceCalculator(BaseDistanceCalculator):
    """
    S_Maha(x) = sqrt((z_x - mu_N)^T Sigma_reg^{-1} (z_x - mu_N)).
    Drop-in replacement for demo1 stable Mahalanobis.
    """

    name = "mahalanobis"

    def _fit_impl(self, z: Array) -> None:
        self.mu_ = np.mean(z, axis=0)
        self.cov_ = _safe_cov(z, reg=float(self.kwargs.get("cov_reg", self.reg)))
        self.inv_cov_ = _pinv(self.cov_, reg=float(self.kwargs.get("cov_reg", self.reg)))

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        d = z - self.mu_
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.inv_cov_, d), 0.0))


class RichGlobalLocalDistanceCalculator(BaseDistanceCalculator):
    """
    M1+M2 / rich Mahalanobis-style score.
    S(x) = alpha * global_MD(z_x) + beta * local_topk_MD(m_{x,t}).
    If sequences are absent, it degrades to global Mahalanobis.
    """

    name = "rich_global_local"

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, topk: int = 5, **kwargs: Any):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.topk = topk

    def _fit_impl(self, z: Array) -> None:
        self.global_mu_ = np.mean(z, axis=0)
        self.global_inv_cov_ = _pinv(_safe_cov(z, reg=self.reg), reg=self.reg)

        if self.train_sequences_ is not None:
            pts = _valid_points(self.train_sequences_, self.train_mask_)
            self.local_mu_ = np.mean(pts, axis=0)
            self.local_inv_cov_ = _pinv(_safe_cov(pts, reg=self.reg), reg=self.reg)
        else:
            self.local_mu_ = self.global_mu_
            self.local_inv_cov_ = self.global_inv_cov_

    def _md(self, x: Array, mu: Array, inv_cov: Array) -> Array:
        d = x - mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, inv_cov, d), 0.0))

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        global_score = self._md(z, self.global_mu_, self.global_inv_cov_)

        if seq is None:
            return global_score

        local_scores = []
        for i in range(seq.shape[0]):
            pts = _sample_points(seq[i], None if mask is None else mask[i])
            md_t = self._md(pts, self.local_mu_, self.local_inv_cov_)
            k = max(1, min(self.topk, len(md_t)))
            local_scores.append(float(np.mean(np.sort(md_t)[-k:])))
        return self.alpha * global_score + self.beta * np.asarray(local_scores)


class SobolevH1DistanceCalculator(BaseDistanceCalculator):
    """
    S_H1(X)=sum_t pi_t ||m_t - mbar_t||^2 + lambda_d sum_t pi_t ||dm_t/dt - dmbar_t/dt||^2.
    If only [N,H] features are given, computes an H1-like value over feature-index differences.
    """

    name = "sobolev_h1"

    def __init__(self, lambda_deriv: float = 0.25, **kwargs: Any):
        super().__init__(**kwargs)
        self.lambda_deriv = lambda_deriv

    def _fit_impl(self, z: Array) -> None:
        self.mu_ = np.mean(z, axis=0)
        if self.train_sequences_ is not None:
            self.ref_traj_ = np.mean(self.train_sequences_, axis=0)
        else:
            self.ref_traj_ = None

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        if seq is None or self.ref_traj_ is None:
            r = z - self.mu_
            val = np.sum(r * r, axis=1)
            if r.shape[1] > 1:
                dr = np.diff(r, axis=1)
                val += self.lambda_deriv * np.sum(dr * dr, axis=1)
            return val

        ref = self.ref_traj_
        scores = []
        for i in range(seq.shape[0]):
            t_len = min(seq.shape[1], ref.shape[0])
            residual = seq[i, :t_len] - ref[:t_len]
            mi = np.ones(t_len) if mask is None else mask[i, :t_len]
            base = np.sum(mi * np.sum(residual * residual, axis=1))
            if t_len > 1:
                dres = np.diff(residual, axis=0)
                deriv = np.sum(mi[1:] * np.sum(dres * dres, axis=1))
            else:
                deriv = 0.0
            denom = np.sum(mi) + self.eps
            scores.append((base + self.lambda_deriv * deriv) / denom)
        return np.asarray(scores)


class SobolevHminus1DistanceCalculator(BaseDistanceCalculator):
    """
    Negative Sobolev H^{-1} low-frequency score:
    S_H-1 = sum_k |r_hat_k|^2 / (1 + omega_k^2).
    Emphasizes accumulated low-frequency trajectory drift.
    """

    name = "sobolev_hminus1"

    def _fit_impl(self, z: Array) -> None:
        self.mu_ = np.mean(z, axis=0)
        if self.train_sequences_ is not None:
            self.ref_traj_ = np.mean(self.train_sequences_, axis=0)
        else:
            self.ref_traj_ = None

    def _hminus1_energy(self, r: Array) -> float:
        # r: [T,H] or [H]
        if r.ndim == 1:
            r = r.reshape(-1, 1)
        fft = np.fft.rfft(r, axis=0)
        freqs = np.fft.rfftfreq(r.shape[0])
        weights = 1.0 / (1.0 + (2.0 * np.pi * freqs) ** 2)
        return float(np.sum(weights[:, None] * (np.abs(fft) ** 2)) / max(1, r.shape[0]))

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        if seq is None or self.ref_traj_ is None:
            return np.asarray([self._hminus1_energy(row - self.mu_) for row in z])
        scores = []
        ref = self.ref_traj_
        for i in range(seq.shape[0]):
            t_len = min(seq.shape[1], ref.shape[0])
            r = seq[i, :t_len] - ref[:t_len]
            if mask is not None:
                r = r[mask[i, :t_len] > 0]
            if len(r) == 0:
                r = seq[i, :t_len] - ref[:t_len]
            scores.append(self._hminus1_energy(r))
        return np.asarray(scores)


class GaussianW2DistanceCalculator(BaseDistanceCalculator):
    """
    Gaussian Wasserstein-2 score:
    W2^2 = ||mu_x-mu_N||^2 + Tr(C_x + C_N - 2(C_N^{1/2} C_x C_N^{1/2})^{1/2}).
    For [N,H] input without trajectory, C_x is eps*I.
    """

    name = "gaussian_w2"

    def _fit_impl(self, z: Array) -> None:
        if self.train_sequences_ is not None:
            pts = _valid_points(self.train_sequences_, self.train_mask_)
        else:
            pts = z
        self.mu_ = np.mean(pts, axis=0)
        self.cov_ = _safe_cov(pts, reg=self.reg)
        self.sqrt_cov_ = _sqrtm_psd(self.cov_, eps=self.eps)

    def _one_w2(self, pts: Array) -> float:
        mu = np.mean(pts, axis=0)
        cov = _safe_cov(pts, reg=self.reg) if len(pts) > 1 else np.eye(len(mu)) * self.reg
        middle = self.sqrt_cov_ @ cov @ self.sqrt_cov_
        sqrt_middle = _sqrtm_psd(middle, eps=self.eps)
        val = np.sum((mu - self.mu_) ** 2) + _matrix_trace(self.cov_ + cov - 2.0 * sqrt_middle)
        return math.sqrt(max(float(val), 0.0))

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        if seq is None:
            return np.asarray([self._one_w2(row.reshape(1, -1)) for row in z])
        return np.asarray([
            self._one_w2(_sample_points(seq[i], None if mask is None else mask[i]))
            for i in range(seq.shape[0])
        ])


class SlicedWassersteinDistanceCalculator(BaseDistanceCalculator):
    """
    Sliced Wasserstein score using random projections.
    Compares sample hidden distribution with train-normal hidden distribution.
    """

    name = "sliced_wasserstein"

    def __init__(self, n_projections: int = 64, max_ref_points: int = 512, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_projections = n_projections
        self.max_ref_points = max_ref_points

    def _fit_impl(self, z: Array) -> None:
        rng = np.random.default_rng(self.random_state)
        pts = _valid_points(self.train_sequences_, self.train_mask_) if self.train_sequences_ is not None else z
        if len(pts) > self.max_ref_points:
            pts = pts[rng.choice(len(pts), size=self.max_ref_points, replace=False)]
        self.ref_points_ = pts
        dirs = rng.normal(size=(self.n_projections, pts.shape[1]))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + self.eps
        self.directions_ = dirs
        self.ref_proj_sorted_ = np.sort(pts @ dirs.T, axis=0)

    def _one_sw(self, pts: Array) -> float:
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        proj = np.sort(pts @ self.directions_.T, axis=0)
        # quantile-match different lengths
        q = np.linspace(0.0, 1.0, num=max(len(proj), len(self.ref_proj_sorted_)))
        vals = []
        for j in range(self.n_projections):
            p1 = np.quantile(proj[:, j], q)
            p0 = np.quantile(self.ref_proj_sorted_[:, j], q)
            vals.append(np.mean((p1 - p0) ** 2))
        return math.sqrt(max(float(np.mean(vals)), 0.0))

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        if seq is None:
            return np.asarray([self._one_sw(row) for row in z])
        return np.asarray([
            self._one_sw(_sample_points(seq[i], None if mask is None else mask[i]))
            for i in range(seq.shape[0])
        ])


class SinkhornDivergenceCalculator(BaseDistanceCalculator):
    """
    Entropic Sinkhorn divergence between sample hidden points and train-normal hidden points.
    S_epsilon(Q_x, P_N) = OT_e(Q_x,P_N)-0.5 OT_e(Q_x,Q_x)-0.5 OT_e(P_N,P_N).
    """

    name = "sinkhorn_divergence"

    def __init__(self, epsilon: float = 0.1, n_iter: int = 80, max_ref_points: int = 256, **kwargs: Any):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.max_ref_points = max_ref_points

    def _fit_impl(self, z: Array) -> None:
        rng = np.random.default_rng(self.random_state)
        pts = _valid_points(self.train_sequences_, self.train_mask_) if self.train_sequences_ is not None else z
        if len(pts) > self.max_ref_points:
            pts = pts[rng.choice(len(pts), size=self.max_ref_points, replace=False)]
        self.ref_points_ = pts

    def _sinkhorn_cost(self, x: Array, y: Array) -> float:
        x = _as_2d(x)
        y = _as_2d(y)
        n, m = len(x), len(y)
        a = np.ones(n) / n
        b = np.ones(m) / m
        c = _pairwise_sq_dists(x, y)
        K = np.exp(-c / max(self.epsilon, self.eps)) + self.eps
        u = np.ones(n) / n
        v = np.ones(m) / m
        for _ in range(self.n_iter):
            u = a / (K @ v + self.eps)
            v = b / (K.T @ u + self.eps)
        pi = (u[:, None] * K) * v[None, :]
        return float(np.sum(pi * c))

    def _one_sd(self, pts: Array) -> float:
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        xy = self._sinkhorn_cost(pts, self.ref_points_)
        xx = self._sinkhorn_cost(pts, pts)
        yy = self._sinkhorn_cost(self.ref_points_, self.ref_points_)
        return math.sqrt(max(xy - 0.5 * xx - 0.5 * yy, 0.0))

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        if seq is None:
            return np.asarray([self._one_sd(row) for row in z])
        return np.asarray([
            self._one_sd(_sample_points(seq[i], None if mask is None else mask[i]))
            for i in range(seq.shape[0])
        ])


class MMDCovKernelDistanceCalculator(BaseDistanceCalculator):
    """
    MMD score with an RBF kernel:
    MMD^2 = E k(X,X') + E k(Y,Y') - 2 E k(X,Y).
    """

    name = "mmd_cov_kernel"

    def __init__(self, bandwidth: Optional[float] = None, max_ref_points: int = 512, **kwargs: Any):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.max_ref_points = max_ref_points

    def _fit_impl(self, z: Array) -> None:
        rng = np.random.default_rng(self.random_state)
        pts = _valid_points(self.train_sequences_, self.train_mask_) if self.train_sequences_ is not None else z
        if len(pts) > self.max_ref_points:
            pts = pts[rng.choice(len(pts), size=self.max_ref_points, replace=False)]
        self.ref_points_ = pts
        if self.bandwidth is None:
            d = _pairwise_sq_dists(pts, pts)
            med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
            self.bandwidth_ = math.sqrt(max(med, self.eps))
        else:
            self.bandwidth_ = float(self.bandwidth)

    def _kernel(self, x: Array, y: Array) -> Array:
        d = _pairwise_sq_dists(x, y)
        return np.exp(-d / (2.0 * self.bandwidth_ ** 2 + self.eps))

    def _one_mmd(self, pts: Array) -> float:
        pts = _as_2d(pts)
        kxx = np.mean(self._kernel(pts, pts))
        kyy = np.mean(self._kernel(self.ref_points_, self.ref_points_))
        kxy = np.mean(self._kernel(pts, self.ref_points_))
        return math.sqrt(max(float(kxx + kyy - 2.0 * kxy), 0.0))

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        if seq is None:
            return np.asarray([self._one_mmd(row.reshape(1, -1)) for row in z])
        return np.asarray([
            self._one_mmd(_sample_points(seq[i], None if mask is None else mask[i]))
            for i in range(seq.shape[0])
        ])


class HamiltonianEnergyCalculator(BaseDistanceCalculator):
    """
    Hamiltonian-like quadratic system-state energy:
    S_H(X)=sum_t pi_t m_t^T H m_t, with H defaulting to train covariance inverse.
    """

    name = "hamiltonian_energy"

    def _fit_impl(self, z: Array) -> None:
        pts = _valid_points(self.train_sequences_, self.train_mask_) if self.train_sequences_ is not None else z
        self.mu_ = np.mean(pts, axis=0)
        self.H_ = _pinv(_safe_cov(pts, reg=self.reg), reg=self.reg)

    def _energy_points(self, pts: Array) -> float:
        d = pts - self.mu_
        e = np.einsum("ij,jk,ik->i", d, self.H_, d)
        return float(np.mean(np.maximum(e, 0.0)))

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        if seq is None:
            return np.asarray([self._energy_points(row.reshape(1, -1)) for row in z])
        return np.asarray([
            self._energy_points(_sample_points(seq[i], None if mask is None else mask[i]))
            for i in range(seq.shape[0])
        ])


class ODEResidualEnergyCalculator(BaseDistanceCalculator):
    """
    ODE residual energy:
    fit d m_t / dt ~= A m_t on train-normal trajectories,
    score = mean_t ||d m_t/dt - A m_t||^2.
    If no trajectory is provided, degrades to squared Euclidean deviation.
    """

    name = "ode_residual_energy"

    def _fit_impl(self, z: Array) -> None:
        self.mu_ = np.mean(z, axis=0)
        self.A_ = None
        if self.train_sequences_ is None:
            return

        xs = []
        ys = []
        seq = self.train_sequences_
        tm = self.train_time_ if self.train_time_ is not None else _time_or_unit(seq, None)
        m = self.train_mask_ if self.train_mask_ is not None else np.ones(seq.shape[:2])
        for i in range(seq.shape[0]):
            valid = m[i] > 0
            pts = seq[i, valid]
            tt = tm[i, valid]
            if len(pts) < 2:
                continue
            dt = np.diff(tt)
            dt = np.where(np.abs(dt) < self.eps, 1.0, dt)
            x_mid = pts[:-1]
            dxdt = np.diff(pts, axis=0) / dt[:, None]
            xs.append(x_mid)
            ys.append(dxdt)
        if xs:
            X = np.vstack(xs)
            Y = np.vstack(ys)
            self.A_ = np.linalg.pinv(X) @ Y

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        if seq is None or self.A_ is None:
            r = z - self.mu_
            return np.sum(r * r, axis=1)

        scores = []
        tm = time if time is not None else _time_or_unit(seq, None)
        for i in range(seq.shape[0]):
            valid = np.ones(seq.shape[1], dtype=bool) if mask is None else mask[i] > 0
            pts = seq[i, valid]
            tt = tm[i, valid]
            if len(pts) < 2:
                scores.append(float(np.sum((z[i] - self.mu_) ** 2)))
                continue
            dt = np.diff(tt)
            dt = np.where(np.abs(dt) < self.eps, 1.0, dt)
            x_mid = pts[:-1]
            dxdt = np.diff(pts, axis=0) / dt[:, None]
            pred = x_mid @ self.A_
            scores.append(float(np.mean(np.sum((dxdt - pred) ** 2, axis=1))))
        return np.asarray(scores)


class DensityMatrixMixin:
    def _rho_from_points(self, pts: Array) -> Array:
        pts = _as_2d(pts)
        w = np.ones(len(pts), dtype=np.float64) / max(1, len(pts))
        xw = pts * np.sqrt(w[:, None])
        rho = xw.T @ xw
        rho = 0.5 * (rho + rho.T)
        rho = rho + np.eye(rho.shape[0]) * self.reg
        tr = _matrix_trace(rho)
        if tr <= self.eps:
            rho = np.eye(rho.shape[0]) / rho.shape[0]
        else:
            rho = rho / tr
        return rho

    def _sample_rhos(self, z: Array, seq: Optional[Array], mask: Optional[Array]) -> list:
        if seq is None:
            return [self._rho_from_points(row.reshape(1, -1)) for row in z]
        return [
            self._rho_from_points(_sample_points(seq[i], None if mask is None else mask[i]))
            for i in range(seq.shape[0])
        ]


class TraceDistanceCalculator(DensityMatrixMixin, BaseDistanceCalculator):
    """
    Quantum-inspired trace distance:
    S_Tr(rho_x)=1/2 ||rho_x-rho_N||_*.
    """

    name = "trace_distance"

    def _fit_impl(self, z: Array) -> None:
        pts = _valid_points(self.train_sequences_, self.train_mask_) if self.train_sequences_ is not None else z
        self.rho_ref_ = self._rho_from_points(pts)

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        scores = []
        for rho in self._sample_rhos(z, seq, mask):
            vals = np.linalg.eigvalsh(0.5 * ((rho - self.rho_ref_) + (rho - self.rho_ref_).T))
            scores.append(0.5 * float(np.sum(np.abs(vals))))
        return np.asarray(scores)


class QRECalculator(DensityMatrixMixin, BaseDistanceCalculator):
    """
    Quantum relative entropy:
    S_QRE(rho_x)=Tr{rho_x(log rho_x - log rho_N)}.
    """

    name = "qre"

    def _fit_impl(self, z: Array) -> None:
        pts = _valid_points(self.train_sequences_, self.train_mask_) if self.train_sequences_ is not None else z
        self.rho_ref_ = self._rho_from_points(pts)
        self.log_rho_ref_ = _logm_psd(self.rho_ref_, eps=self.eps)

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        scores = []
        for rho in self._sample_rhos(z, seq, mask):
            val = _matrix_trace(rho @ (_logm_psd(rho, eps=self.eps) - self.log_rho_ref_))
            scores.append(max(val, 0.0))
        return np.asarray(scores)


class BuresDistanceCalculator(DensityMatrixMixin, BaseDistanceCalculator):
    """
    Bures distance:
    S_B(rho_x)=sqrt(2(1 - F(rho_x,rho_N))),
    F = Tr sqrt(sqrt(rho_N) rho_x sqrt(rho_N)).
    """

    name = "bures"

    def _fit_impl(self, z: Array) -> None:
        pts = _valid_points(self.train_sequences_, self.train_mask_) if self.train_sequences_ is not None else z
        self.rho_ref_ = self._rho_from_points(pts)
        self.sqrt_ref_ = _sqrtm_psd(self.rho_ref_, eps=self.eps)

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        scores = []
        for rho in self._sample_rhos(z, seq, mask):
            middle = self.sqrt_ref_ @ rho @ self.sqrt_ref_
            fidelity = _matrix_trace(_sqrtm_psd(middle, eps=self.eps))
            fidelity = float(np.clip(fidelity, 0.0, 1.0))
            scores.append(math.sqrt(max(2.0 * (1.0 - fidelity), 0.0)))
        return np.asarray(scores)


class VonNeumannEntropyCalculator(DensityMatrixMixin, BaseDistanceCalculator):
    """
    Von Neumann entropy anomaly:
    entropy(rho)=-Tr(rho log rho), score=|entropy(rho_x)-mean_train_entropy|.
    """

    name = "von_neumann_entropy"

    def _entropy(self, rho: Array) -> float:
        vals = np.linalg.eigvalsh(0.5 * (rho + rho.T))
        vals = np.clip(vals, self.eps, 1.0)
        return float(-np.sum(vals * np.log(vals)))

    def _fit_impl(self, z: Array) -> None:
        if self.train_sequences_ is not None:
            rhos = self._sample_rhos(z, self.train_sequences_, self.train_mask_)
        else:
            rhos = self._sample_rhos(z, None, None)
        self.ref_entropy_ = float(np.mean([self._entropy(r) for r in rhos]))

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        return np.asarray([abs(self._entropy(r) - self.ref_entropy_) for r in self._sample_rhos(z, seq, mask)])


class PurityCalculator(DensityMatrixMixin, BaseDistanceCalculator):
    """
    Purity anomaly:
    purity(rho)=Tr(rho^2), score=|purity(rho_x)-mean_train_purity|.
    """

    name = "purity"

    def _purity(self, rho: Array) -> float:
        return _matrix_trace(rho @ rho)

    def _fit_impl(self, z: Array) -> None:
        if self.train_sequences_ is not None:
            rhos = self._sample_rhos(z, self.train_sequences_, self.train_mask_)
        else:
            rhos = self._sample_rhos(z, None, None)
        self.ref_purity_ = float(np.mean([self._purity(r) for r in rhos]))

    def _score_impl(self, z: Array, seq: Optional[Array], mask: Optional[Array], time: Optional[Array]) -> Array:
        return np.asarray([abs(self._purity(r) - self.ref_purity_) for r in self._sample_rhos(z, seq, mask)])


REGISTRY = {
    "euclidean": EuclideanDistanceCalculator,
    "Euclidean": EuclideanDistanceCalculator,

    "mahalanobis": MahalanobisDistanceCalculator,
    "stable_mahalanobis": MahalanobisDistanceCalculator,
    "robust_mahalanobis": MahalanobisDistanceCalculator,
    "MahalanobisScorer": MahalanobisDistanceCalculator,

    "rich_global_local": RichGlobalLocalDistanceCalculator,
    "Rich_global_local": RichGlobalLocalDistanceCalculator,
    "M1+M2": RichGlobalLocalDistanceCalculator,
    "m1m2_rich_mahalanobis": RichGlobalLocalDistanceCalculator,

    "sobolev_h1": SobolevH1DistanceCalculator,
    "sobolev_hminus1": SobolevHminus1DistanceCalculator,
    "sobolev_h-1": SobolevHminus1DistanceCalculator,

    "Gaussian_W2": GaussianW2DistanceCalculator,
    "gaussian_w2": GaussianW2DistanceCalculator,

    "Sliced_Wasserstein": SlicedWassersteinDistanceCalculator,
    "sliced_wasserstein": SlicedWassersteinDistanceCalculator,

    "Sinkhorn_divergence": SinkhornDivergenceCalculator,
    "sinkhorn_divergence": SinkhornDivergenceCalculator,

    "mmd_cov_kernel": MMDCovKernelDistanceCalculator,
    "MMD": MMDCovKernelDistanceCalculator,

    "hamiltonian_energy": HamiltonianEnergyCalculator,
    "ode_residual_energy": ODEResidualEnergyCalculator,

    "trace_distance": TraceDistanceCalculator,
    "qre": QRECalculator,
    "bures": BuresDistanceCalculator,
    "von_neumann_entropy": VonNeumannEntropyCalculator,
    "purity": PurityCalculator,
}


def create_distance_calculator(method: str, **kwargs: Any) -> BaseDistanceCalculator:
    if method not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown distance method: {method}. Available: {available}")
    return REGISTRY[method](**kwargs)


def available_methods() -> Tuple[str, ...]:
    return tuple(sorted(REGISTRY.keys()))
