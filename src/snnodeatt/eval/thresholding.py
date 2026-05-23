import numpy as np

def quantile_threshold(normal_scores, q=0.99):
    return float(np.quantile(np.asarray(normal_scores, dtype=float), q))
