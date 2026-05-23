"""Density-matrix purity score wrapper."""

from snnodeatt.scoring.quantum_scores import purity_score

SCORE_NAME = "purity"
score = purity_score
