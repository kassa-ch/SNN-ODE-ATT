import numpy as np

class MahalanobisScorer:
    def __init__(self, regularization=1e-2):
        self.regularization = regularization

    def fit(self, z_normal):
        z = np.asarray(z_normal, dtype=float)
        self.mu_ = z.mean(axis=0)
        xc = z - self.mu_
        cov = xc.T @ xc / max(1, len(z) - 1)
        cov = cov + self.regularization * np.eye(cov.shape[0])
        self.inv_ = np.linalg.pinv(cov)
        return self

    def score(self, z):
        z = np.asarray(z, dtype=float)
        d = z - self.mu_
        return np.sqrt(np.maximum(0.0, np.einsum("bi,ij,bj->b", d, self.inv_, d)))
