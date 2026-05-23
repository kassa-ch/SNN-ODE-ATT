"""Quantum-inspired density-matrix trajectory scores.

These methods use density matrix geometry as a representation tool. They do
not claim the industrial process is a physical quantum system.
"""
from __future__ import annotations

import torch

from .functional_scores import as_tensor, choose_device, mask_tensor, time_weights


def _projection_matrix(d, q, device, seed=42):
    if q is None or q >= d:
        return None
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(d, q, generator=g).to(device)
    qmat, _ = torch.linalg.qr(r, mode="reduced")
    return qmat


def build_density_matrix(m_reset, mask, weights, eps=1e-5, pca_dim=None, seed=42):
    """Build rho [q,q] from one trajectory m_reset [T,D]."""
    device = m_reset.device
    valid = mask > 0
    x = m_reset[valid]
    w = weights[valid].float()
    w = w / w.sum().clamp_min(1e-12)
    proj = _projection_matrix(x.shape[-1], pca_dim, device, seed)
    if proj is not None:
        x = x @ proj
    x = x - (x * w[:, None]).sum(dim=0, keepdim=True)
    rho = x.T @ (x * w[:, None])
    rho = rho + eps * torch.eye(rho.shape[0], device=device)
    rho = 0.5 * (rho + rho.T)
    return rho / torch.trace(rho).clamp_min(1e-12)


def _eigvalsh(rho):
    return torch.linalg.eigvalsh(0.5 * (rho + rho.T)).clamp_min(1e-8)


def _matrix_log(rho):
    vals, vecs = torch.linalg.eigh(0.5 * (rho + rho.T))
    vals = vals.clamp_min(1e-8)
    return (vecs * torch.log(vals).unsqueeze(0)) @ vecs.T


def _matrix_sqrt(rho):
    vals, vecs = torch.linalg.eigh(0.5 * (rho + rho.T))
    vals = vals.clamp_min(1e-10)
    return (vecs * torch.sqrt(vals).unsqueeze(0)) @ vecs.T


def _rhos(cache, indices, config):
    device = choose_device(config)
    m = as_tensor(cache["m_reset"], device=device, dtype=torch.float32)
    mask = as_tensor(cache["mask"], device=device, dtype=torch.bool)
    w = time_weights(cache, config.get("time_weight", "dt"), device=device)
    pca_dim = config.get("pca_dim", 32 if m.shape[-1] > 64 else None)
    return [build_density_matrix(m[i], mask[i], w[i], pca_dim=pca_dim, seed=int(config.get("seed", 42))) for i in indices]


def _reference_rho(cache, reference_mask, config):
    ref_idx = torch.where(mask_tensor(reference_mask))[0].tolist()
    rhos = _rhos(cache, ref_idx, config)
    rho = torch.stack(rhos).mean(dim=0)
    return rho / torch.trace(rho).clamp_min(1e-12), rhos


def trace_distance_score(cache, reference_mask, eval_mask, config=None):
    config = config or {}
    rho_n, _ = _reference_rho(cache, reference_mask, config)
    scores = []
    for rho in _rhos(cache, torch.where(mask_tensor(eval_mask))[0].tolist(), config):
        scores.append(0.5 * torch.linalg.matrix_norm(rho - rho_n, ord="nuc"))
    return torch.stack(scores).detach().cpu()


def quantum_relative_entropy_score(cache, reference_mask, eval_mask, config=None):
    config = config or {}
    sigma, _ = _reference_rho(cache, reference_mask, config)
    log_sigma = _matrix_log(sigma)
    scores = []
    for rho in _rhos(cache, torch.where(mask_tensor(eval_mask))[0].tolist(), config):
        scores.append(torch.trace(rho @ (_matrix_log(rho) - log_sigma)).real.clamp_min(0))
    return torch.stack(scores).detach().cpu()


def _entropy(rho):
    vals = _eigvalsh(rho)
    return -(vals * torch.log(vals)).sum()


def _purity(rho):
    return torch.trace(rho @ rho).real


def von_neumann_entropy_score(cache, reference_mask, eval_mask, config=None):
    config = config or {}
    _, ref_rhos = _reference_rho(cache, reference_mask, config)
    ref = torch.stack([_entropy(r) for r in ref_rhos])
    mu, sd = ref.mean(), ref.std().clamp_min(1e-6)
    vals = torch.stack([_entropy(r) for r in _rhos(cache, torch.where(mask_tensor(eval_mask))[0].tolist(), config)])
    return ((vals - mu).abs() / sd).detach().cpu()


def purity_score(cache, reference_mask, eval_mask, config=None):
    config = config or {}
    _, ref_rhos = _reference_rho(cache, reference_mask, config)
    ref = torch.stack([_purity(r) for r in ref_rhos])
    mu, sd = ref.mean(), ref.std().clamp_min(1e-6)
    vals = torch.stack([_purity(r) for r in _rhos(cache, torch.where(mask_tensor(eval_mask))[0].tolist(), config)])
    return ((vals - mu).abs() / sd).detach().cpu()


def bures_distance_score(cache, reference_mask, eval_mask, config=None):
    config = config or {}
    sigma, _ = _reference_rho(cache, reference_mask, config)
    sqrt_sigma = _matrix_sqrt(sigma)
    scores = []
    for rho in _rhos(cache, torch.where(mask_tensor(eval_mask))[0].tolist(), config):
        inner = sqrt_sigma @ rho @ sqrt_sigma
        fid = torch.trace(_matrix_sqrt(inner)).real.clamp_min(0).pow(2)
        scores.append(torch.sqrt((2 * (1 - torch.sqrt(fid.clamp_min(0)).clamp_max(1))).clamp_min(0)))
    return torch.stack(scores).detach().cpu()


def density_matrix_combined_score(cache, reference_mask, eval_mask, config=None):
    config = config or {}
    a = float(config.get("alpha", 1.0)); b = float(config.get("beta", 0.25)); c = float(config.get("gamma", 0.25))
    return a * trace_distance_score(cache, reference_mask, eval_mask, config) + b * quantum_relative_entropy_score(cache, reference_mask, eval_mask, config) + c * von_neumann_entropy_score(cache, reference_mask, eval_mask, config)
