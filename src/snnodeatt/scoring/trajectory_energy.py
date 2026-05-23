"""Trajectory energy scores including Hamiltonian/action-inspired variants."""
from __future__ import annotations

import torch

from .functional_scores import as_tensor, choose_device, finite_difference, mask_tensor, normal_point_stats, normal_trajectory_mean, time_weights


def hamiltonian_energy_score(cache, reference_mask, eval_mask, config=None):
    """Hamiltonian-inspired hidden energy using diagonal inverse covariance."""
    config = config or {}
    device = choose_device(config)
    m = as_tensor(cache["m_reset"], device=device, dtype=torch.float32)
    w = time_weights(cache, config.get("time_weight", "dt"), device=device)
    _, var = normal_point_stats(cache, reference_mask, config | {"device": str(device)})
    energy = ((m.pow(2) / var).mean(dim=-1) * w).sum(dim=1)
    ref = energy[mask_tensor(reference_mask, device=device)]
    ev = energy[mask_tensor(eval_mask, device=device)]
    return ((ev - ref.mean()).abs() / ref.std().clamp_min(1e-6)).detach().cpu()


def ode_residual_energy_score(cache, reference_mask, eval_mask, config=None):
    """Finite-difference residual energy. If f_ODE is supplied, replace mean dot_m with f_ODE(m)."""
    config = config or {}
    device = choose_device(config)
    m = as_tensor(cache["m_reset"], device=device, dtype=torch.float32)
    mask = as_tensor(cache["mask"], device=device, dtype=torch.float32)
    dt = as_tensor(cache["delta_t"], device=device, dtype=torch.float32)
    dm, valid = finite_difference(m, dt, mask)
    ref = mask_tensor(reference_mask, device=device)
    ref_valid = valid[ref].unsqueeze(-1)
    mean_dot = (dm[ref] * ref_valid).sum(dim=(0, 1)) / ref_valid.sum(dim=(0, 1)).clamp_min(1.0)
    wd = dt[:, 1:] * valid.float()
    wd = wd / wd.sum(dim=1, keepdim=True).clamp_min(1e-12)
    score = (((dm - mean_dot) ** 2).mean(dim=-1) * wd).sum(dim=1)
    return score[mask_tensor(eval_mask, device=device)].detach().cpu()


def action_functional_score(cache, reference_mask, eval_mask, config=None):
    """Action = derivative residual energy + potential Mahalanobis energy."""
    from .functional_scores import covariance_aware_trajectory_energy
    return ode_residual_energy_score(cache, reference_mask, eval_mask, config) + covariance_aware_trajectory_energy(cache, reference_mask, eval_mask, config)
