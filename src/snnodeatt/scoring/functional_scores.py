"""Functional and Sobolev-style anomaly scores for hidden trajectories.

All public score functions accept:
    cache: dict with m_reset [N,T,D], mask [N,T], delta_t [N,T], tau [N,T].
    reference_mask: boolean mask selecting train/val normal samples.
    eval_mask: boolean mask selecting samples to score.
    config: optional dict.

They return a torch.Tensor with shape [N_eval]. Padding positions (mask=0)
are ignored. Scores are anomaly-high by convention.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch


def as_tensor(x, device=None, dtype=None):
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def choose_device(config=None):
    config = config or {}
    dev = config.get("device", "cpu")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def mask_tensor(mask, n=None, device=None):
    if torch.is_tensor(mask):
        out = mask.bool()
    else:
        out = torch.as_tensor(mask).bool()
    if n is not None and out.ndim == 0:
        out = out.repeat(n)
    return out.to(device) if device is not None else out


def split_list(cache) -> list[str]:
    split = cache["split"]
    if torch.is_tensor(split):
        return [str(x) for x in split.cpu().tolist()]
    return [str(x) for x in split]


def reference_eval_masks(cache, reference_splits=("train_normal", "val_normal"), eval_splits=("test_normal", "test_abnormal")):
    splits = split_list(cache)
    labels = as_tensor(cache["label"]).cpu().long()
    ref = torch.tensor([(s in reference_splits) and int(labels[i]) == 0 for i, s in enumerate(splits)], dtype=torch.bool)
    ev = torch.tensor([s in eval_splits for s in splits], dtype=torch.bool)
    return ref, ev


def time_weights(cache, mode="dt", device=None):
    mask = as_tensor(cache["mask"], device=device, dtype=torch.float32)
    dt = as_tensor(cache.get("delta_t", torch.ones_like(mask)), device=device, dtype=torch.float32).clamp_min(0.0)
    if mode == "trapezoid":
        w = torch.zeros_like(dt)
        if dt.shape[1] == 1:
            w[:, 0] = dt[:, 0]
        else:
            w[:, 0] = dt[:, 1] / 2
            if dt.shape[1] > 2:
                w[:, 1:-1] = (dt[:, 1:-1] + dt[:, 2:]) / 2
            w[:, -1] = dt[:, -1] / 2
    else:
        w = dt
    w = w * mask
    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)


def weighted_latent(cache, weight_mode="dt", device=None):
    m = as_tensor(cache["m_reset"], device=device, dtype=torch.float32)
    w = time_weights(cache, weight_mode, device=device)
    return (m * w.unsqueeze(-1)).sum(dim=1)


def normal_point_stats(cache, reference_mask, config=None):
    config = config or {}
    device = choose_device(config)
    m = as_tensor(cache["m_reset"], device=device, dtype=torch.float32)
    mask = as_tensor(cache["mask"], device=device, dtype=torch.bool)
    ref = mask_tensor(reference_mask, device=device)
    pts = m[ref][mask[ref]]
    mu = pts.mean(dim=0)
    var = pts.var(dim=0, unbiased=False).clamp_min(config.get("ridge", 1e-2))
    return mu, var


def normal_trajectory_mean(cache, reference_mask, device=None):
    m = as_tensor(cache["m_reset"], device=device, dtype=torch.float32)
    mask = as_tensor(cache["mask"], device=device, dtype=torch.float32)
    ref = mask_tensor(reference_mask, device=device)
    mr = m[ref]
    wr = mask[ref].unsqueeze(-1)
    return (mr * wr).sum(dim=0) / wr.sum(dim=0).clamp_min(1.0)


def finite_difference(m, delta_t, mask):
    dt = delta_t[:, 1:].clamp_min(1e-6).unsqueeze(-1)
    dm = (m[:, 1:] - m[:, :-1]) / dt
    valid = (mask[:, 1:] > 0) & (mask[:, :-1] > 0)
    return dm, valid


def euclidean_latent_score(cache, reference_mask, eval_mask, config=None):
    """Weighted latent Euclidean distance. Returns [N_eval]."""
    config = config or {}
    device = choose_device(config)
    z = weighted_latent(cache, config.get("time_weight", "dt"), device=device)
    ref = mask_tensor(reference_mask, device=device)
    ev = mask_tensor(eval_mask, device=device)
    mu = z[ref].mean(dim=0)
    return torch.linalg.norm(z[ev] - mu, dim=1).detach().cpu()


def sobolev_h1_score(cache, reference_mask, eval_mask, config=None):
    """Discrete H1-like trajectory score with value and derivative terms."""
    config = config or {}
    device = choose_device(config)
    lam = float(config.get("lambda_derivative", 0.25))
    m = as_tensor(cache["m_reset"], device=device, dtype=torch.float32)
    mask = as_tensor(cache["mask"], device=device, dtype=torch.float32)
    dt = as_tensor(cache["delta_t"], device=device, dtype=torch.float32)
    ev = mask_tensor(eval_mask, device=device)
    mu_t = normal_trajectory_mean(cache, reference_mask, device=device)
    w = time_weights(cache, config.get("time_weight", "dt"), device=device)
    value = (((m - mu_t.unsqueeze(0)) ** 2).mean(dim=-1) * w).sum(dim=1)
    dm, dvalid = finite_difference(m, dt, mask)
    dmu = (mu_t[1:] - mu_t[:-1]) / dt[:, 1:].mean(dim=0).clamp_min(1e-6).unsqueeze(-1)
    wd = (dt[:, 1:] * dvalid.float())
    wd = wd / wd.sum(dim=1, keepdim=True).clamp_min(1e-12)
    deriv = (((dm - dmu.unsqueeze(0)) ** 2).mean(dim=-1) * wd).sum(dim=1)
    return (value[ev] + lam * deriv[ev]).detach().cpu()


def negative_sobolev_hminus1_score(cache, reference_mask, eval_mask, config=None):
    """Spectral approximation of an H^{-1} norm using torch.fft.rfft."""
    config = config or {}
    device = choose_device(config)
    m = as_tensor(cache["m_reset"], device=device, dtype=torch.float32)
    mask = as_tensor(cache["mask"], device=device, dtype=torch.float32)
    ev = mask_tensor(eval_mask, device=device)
    mu_t = normal_trajectory_mean(cache, reference_mask, device=device)
    residual = (m - mu_t.unsqueeze(0)) * mask.unsqueeze(-1)
    spec = torch.fft.rfft(residual, dim=1)
    k = torch.arange(spec.shape[1], device=device, dtype=torch.float32)
    lam = (2 * math.pi * k / max(1, m.shape[1])) ** 2
    denom = (1.0 + lam).view(1, -1, 1)
    score = (spec.abs().pow(2) / denom).mean(dim=(1, 2))
    return score[ev].detach().cpu()


def covariance_aware_trajectory_energy(cache, reference_mask, eval_mask, config=None):
    """Trajectory Mahalanobis energy using normal hidden-point covariance."""
    config = config or {}
    device = choose_device(config)
    diagonal = bool(config.get("diagonal", True))
    m = as_tensor(cache["m_reset"], device=device, dtype=torch.float32)
    ev = mask_tensor(eval_mask, device=device)
    w = time_weights(cache, config.get("time_weight", "dt"), device=device)
    mu, var = normal_point_stats(cache, reference_mask, config | {"device": str(device)})
    if diagonal:
        e = (((m - mu) ** 2) / var).mean(dim=-1)
    else:
        mask = as_tensor(cache["mask"], device=device, dtype=torch.bool)
        ref = mask_tensor(reference_mask, device=device)
        pts = m[ref][mask[ref]]
        xc = pts - mu
        cov = xc.T @ xc / max(1, len(pts) - 1) + float(config.get("ridge", 1e-2)) * torch.eye(m.shape[-1], device=device)
        inv = torch.linalg.pinv(cov)
        d = m - mu
        e = torch.einsum("ntd,df,ntf->nt", d, inv, d) / m.shape[-1]
    return (e * w).sum(dim=1)[ev].detach().cpu()
