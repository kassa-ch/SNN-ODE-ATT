"""Negative Sobolev H-minus-1 score wrapper."""

from snnodeatt.scoring.functional_scores import negative_sobolev_hminus1_score

SCORE_NAME = "sobolev_hminus1"
score = negative_sobolev_hminus1_score
