"""Kernel and energy-distance trajectory anomaly scores."""
from __future__ import annotations

import torch

from .functional_scores import as_tensor, choose_device, mask_tensor, normal_point_stats, time_weights


def _reference_points(cache, reference_mask, config, include_tau=True):
    device = choose_device(config)
    m = as_tensor(cache["m_reset"], device=device, dtype=torch.float32)
    mask = as_tensor(cache["mask"], device=device, dtype=torch.bool)
    tau = as_tensor(cache["tau"], device=device, dtype=torch.float32)
    ref = mask_tensor(reference_mask, device=device)
    pts = m[ref][mask[ref]]
    tt = tau[ref][mask[ref]]
    count = min(int(config.get("prototype_count", 64)), len(pts))
    g = torch.Generator(device="cpu").manual_seed(int(config.get("seed", 42)))
    idx = torch.randperm(len(pts), generator=g)[:count].to(device)
    return pts[idx], tt[idx]


def _sample_points(cache, i, config):
    device = choose_device(config)
    m = as_tensor(cache["m_reset"], device=device, dtype=torch.float32)[i]
    mask = as_tensor(cache["mask"], device=device, dtype=torch.bool)[i]
    tau = as_tensor(cache["tau"], device=device, dtype=torch.float32)[i]
    w_all = time_weights(cache, config.get("time_weight", "dt"), device=device)[i]
    return m[mask], tau[mask], w_all[mask] / w_all[mask].sum().clamp_min(1e-12)


def _diag_maha_cdist(x, y, var):
    return (((x[:, None, :] - y[None, :, :]) ** 2) / var).mean(dim=-1)


def mmd_covariance_kernel_score(cache, reference_mask, eval_mask, config=None):
    """Weighted MMD^2 between a sample trajectory and normal prototype points."""
    config = config or {}
    device = choose_device(config)
    ev = torch.where(mask_tensor(eval_mask))[0].tolist()
    proto, proto_tau = _reference_points(cache, reference_mask, config)
    _, var = normal_point_stats(cache, reference_mask, config | {"device": str(device)})
    b = torch.ones(len(proto), device=device) / len(proto)
    sigma_m = float(config.get("sigma_m", 1.0))
    sigma_t = float(config.get("sigma_t", 1.0))
    cyy = _diag_maha_cdist(proto, proto, var) / sigma_m**2 + (proto_tau[:, None] - proto_tau[None, :]).pow(2) / sigma_t**2
    kyy = torch.exp(-0.5 * cyy)
    ref_term = b @ kyy @ b
    scores = []
    for i in ev:
        x, tau, a = _sample_points(cache, i, config)
        cxx = _diag_maha_cdist(x, x, var) / sigma_m**2 + (tau[:, None] - tau[None, :]).pow(2) / sigma_t**2
        cxy = _diag_maha_cdist(x, proto, var) / sigma_m**2 + (tau[:, None] - proto_tau[None, :]).pow(2) / sigma_t**2
        kxx = torch.exp(-0.5 * cxx)
        kxy = torch.exp(-0.5 * cxy)
        scores.append(a @ kxx @ a + ref_term - 2 * (a @ kxy @ b))
    return torch.stack(scores).detach().cpu()


def energy_distance_score(cache, reference_mask, eval_mask, config=None):
    """Weighted empirical energy distance to normal prototype points."""
    config = config or {}
    device = choose_device(config)
    ev = torch.where(mask_tensor(eval_mask))[0].tolist()
    proto, proto_tau = _reference_points(cache, reference_mask, config)
    _, var = normal_point_stats(cache, reference_mask, config | {"device": str(device)})
    b = torch.ones(len(proto), device=device) / len(proto)
    dyy = torch.sqrt(_diag_maha_cdist(proto, proto, var).clamp_min(0))
    ref_term = b @ dyy @ b
    scores = []
    for i in ev:
        x, tau, a = _sample_points(cache, i, config)
        dxx = torch.sqrt(_diag_maha_cdist(x, x, var).clamp_min(0))
        dxy = torch.sqrt(_diag_maha_cdist(x, proto, var).clamp_min(0))
        scores.append(2 * (a @ dxy @ b) - (a @ dxx @ a) - ref_term)
    return torch.stack(scores).detach().cpu()
