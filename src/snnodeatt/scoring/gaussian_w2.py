import numpy as np

def gaussian_w2_mean_only(x, mu):
    x = np.asarray(x); mu = np.asarray(mu)
    return np.sum((x - mu) ** 2, axis=-1)
