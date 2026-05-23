import numpy as np

def view_signature(mask, delta_t, tau=None):
    valid = np.asarray(mask) > 0
    dt = np.asarray(delta_t)[valid]
    if len(dt) == 0:
        return np.zeros(5)
    return np.array([valid.sum(), dt.mean(), dt.std(), dt.max(), np.quantile(dt, 0.9)], dtype=float)

def topk_local_indices(signature, normal_signatures, k=15):
    normal_signatures = np.asarray(normal_signatures, dtype=float)
    sig = np.asarray(signature, dtype=float)
    dist = np.linalg.norm(normal_signatures - sig[None, :], axis=1)
    return np.argsort(dist)[:k]
