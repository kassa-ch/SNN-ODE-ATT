import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA


def flatten_ts(X):
    return X.reshape(X.shape[0], -1)


# =========================
# 1. Linear Auto-Regressive
# =========================
class LinearAR:
    def fit(self, X):
        Z = flatten_ts(X)
        self.model = LinearRegression()
        self.model.fit(Z[:, :-1], Z[:, 1:])

    def score(self, X):
        Z = flatten_ts(X)
        pred = self.model.predict(Z[:, :-1])
        return np.mean((Z[:, 1:] - pred) ** 2, axis=1)


# =========================
# 2. LASSO Auto-Regressive
# =========================
class LassoAR:
    def __init__(self, alpha=1e-3):
        self.alpha = alpha

    def fit(self, X):
        Z = flatten_ts(X)
        self.model = Lasso(alpha=self.alpha, max_iter=5000)
        self.model.fit(Z[:, :-1], Z[:, 1:])

    def score(self, X):
        Z = flatten_ts(X)
        pred = self.model.predict(Z[:, :-1])
        return np.mean((Z[:, 1:] - pred) ** 2, axis=1)


# =========================
# 3. Kernel Ridge Regression
# =========================
class KRRModel:
    def __init__(self):
        self.model = KernelRidge(kernel="rbf", alpha=1e-2)

    def fit(self, X):
        Z = flatten_ts(X)
        self.model.fit(Z[:, :-1], Z[:, 1:])

    def score(self, X):
        Z = flatten_ts(X)
        pred = self.model.predict(Z[:, :-1])
        return np.mean((Z[:, 1:] - pred) ** 2, axis=1)


# =========================
# 4. FPCA (low-variance energy)
# =========================
class FPCAModel:
    def __init__(self, var_ratio=0.95):
        self.var_ratio = var_ratio

    def fit(self, X):
        Z = flatten_ts(X)
        self.pca = PCA(n_components=self.var_ratio, svd_solver="full")
        self.pca.fit(Z)

    def score(self, X):
        Z = flatten_ts(X)
        Z_rec = self.pca.inverse_transform(self.pca.transform(Z))
        residual = Z - Z_rec
        return np.mean(residual ** 2, axis=1)


# =========================
# 5. Cumulative Integral
# =========================
class CumIntModel:
    def fit(self, X):
        C = np.cumsum(X, axis=1)
        self.mean = C.mean(axis=0)
        self.std = C.std(axis=0) + 1e-6

    def score(self, X):
        C = np.cumsum(X, axis=1)
        z = (C - self.mean) / self.std
        return np.max(np.linalg.norm(z, axis=-1), axis=1)
