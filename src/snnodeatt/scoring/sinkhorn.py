import torch

def _normalize(w):
    return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

def pairwise_cost(x, y):
    return torch.cdist(x, y, p=2).pow(2)

def sinkhorn_ot(a, x, b, y, epsilon=0.05, n_iters=300):
    """Entropic OT cost with log-domain updates.

    Args:
        a: [N] weights, x: [N,D], b: [M] weights, y: [M,D].
    """
    a = _normalize(a.reshape(-1))
    b = _normalize(b.reshape(-1))
    c = pairwise_cost(x, y)
    log_a = torch.log(a.clamp_min(1e-12))
    log_b = torch.log(b.clamp_min(1e-12))
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(n_iters):
        f = epsilon * (log_a - torch.logsumexp((g[None, :] - c) / epsilon, dim=1))
        g = epsilon * (log_b - torch.logsumexp((f[:, None] - c) / epsilon, dim=0))
    log_pi = (f[:, None] + g[None, :] - c) / epsilon
    pi = torch.exp(log_pi)
    return torch.sum(pi * c)

def sinkhorn_divergence(a, x, b, y, epsilon=0.05, n_iters=300):
    ot_xy = sinkhorn_ot(a, x, b, y, epsilon, n_iters)
    ot_xx = sinkhorn_ot(a, x, a, x, epsilon, n_iters)
    ot_yy = sinkhorn_ot(b, y, b, y, epsilon, n_iters)
    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy
