"""Trace distance density-matrix score wrapper."""

from snnodeatt.scoring.quantum_scores import trace_distance_score

SCORE_NAME = "trace_distance"
score = trace_distance_score
