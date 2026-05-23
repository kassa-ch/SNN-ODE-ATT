"""Covariance-aware MMD score wrapper."""

from snnodeatt.scoring.kernel_scores import mmd_covariance_kernel_score

SCORE_NAME = "mmd_cov_kernel"
score = mmd_covariance_kernel_score
