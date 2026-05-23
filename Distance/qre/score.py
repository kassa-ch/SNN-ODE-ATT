"""Quantum relative entropy score wrapper."""

from snnodeatt.scoring.quantum_scores import quantum_relative_entropy_score

SCORE_NAME = "qre"
score = quantum_relative_entropy_score
