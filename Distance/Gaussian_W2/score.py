"""
Gaussian_W2 score module.

Drop-in demo1 usage:
    from Distance.Gaussian_W2.score import DistanceCalculator
    calc = DistanceCalculator()
    calc.fit(train_features)
    scores = calc.calculate_distance(test_features)

Optional trajectory usage:
    calc.fit(train_features, train_sequences=train_seq, mask=train_mask, time=train_time)
    scores = calc.calculate_distance(test_features, sequences=test_seq, mask=test_mask, time=test_time)
"""

try:
    from Distance.common import GaussianW2DistanceCalculator
except Exception:
    from ..common import GaussianW2DistanceCalculator


DistanceCalculator = GaussianW2DistanceCalculator


def build_scorer(**kwargs):
    return DistanceCalculator(**kwargs)


__all__ = ["DistanceCalculator", "build_scorer"]
