import torch
from snnodeatt.scoring.sinkhorn import sinkhorn_divergence

def sinkhorn_normal_regularization(hidden, mask, prototypes, epsilon=0.05):
    """Normal-only trajectory-to-reference Sinkhorn regularization."""
    weights = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    b = torch.ones(len(prototypes), device=hidden.device) / len(prototypes)
    losses = []
    for h, w, m in zip(hidden, weights, mask):
        valid = m > 0
        losses.append(sinkhorn_divergence(w[valid], h[valid], b, prototypes, epsilon=epsilon))
    return torch.stack(losses).mean()
