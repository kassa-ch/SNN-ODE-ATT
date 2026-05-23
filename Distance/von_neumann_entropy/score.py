"""Von Neumann entropy score wrapper."""

from snnodeatt.scoring.quantum_scores import von_neumann_entropy_score

SCORE_NAME = "von_neumann_entropy"
score = von_neumann_entropy_score
