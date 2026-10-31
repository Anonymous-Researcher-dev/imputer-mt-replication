"""Local SoftImpute solver (the study's own implementation, kept as a *secondary* subject).

Algorithm (Mazumder, Hastie & Tibshirani, 2010): iterative singular-value soft-thresholding with
threshold  lambda = alpha * ||P_Omega(X)||_2  and observed entries restored after each sweep.
"""
import numpy as np


class SoftImpute:
    def __init__(self, alpha=0.05, max_iter=60, tol=1e-5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def _shrink(self, Z, threshold):
        U, s, VT = np.linalg.svd(Z, full_matrices=False)
        s = np.maximum(s - threshold, 0.0)
        return (U * s) @ VT

    def fit_transform(self, X):
        X = np.array(X, dtype=float)
        if np.isinf(X).any():
            raise ValueError("Input X contains infinity.")
        missing = np.isnan(X)
        if not missing.any():
            return X
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.nanmean(X, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        Z = np.where(missing, means[None, :], X)
        threshold = self.alpha * float(np.linalg.norm(np.nan_to_num(X, nan=0.0), ord=2))
        for _ in range(self.max_iter):
            previous = Z.copy()
            reconstructed = self._shrink(Z, threshold)
            Z = np.where(missing, reconstructed, X)
            change = np.linalg.norm(Z - previous) / (np.linalg.norm(previous) + 1e-12)
            if change < self.tol:
                break
        return Z
