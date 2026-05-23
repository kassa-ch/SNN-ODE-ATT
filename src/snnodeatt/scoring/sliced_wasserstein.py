import numpy as np

def sliced_wasserstein_1d(x, ref, num_projections=128, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); ref = np.asarray(ref)
    dirs = rng.normal(size=(num_projections, x.shape[-1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    out = []
    for v in dirs:
        sx = np.sort(x @ v)
        sr = np.sort(ref @ v)
        n = min(len(sx), len(sr))
        out.append(np.mean((sx[:n] - sr[:n]) ** 2))
    return float(np.mean(out))
