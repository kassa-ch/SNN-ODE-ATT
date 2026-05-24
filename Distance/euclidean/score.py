"""
euclidean score module.

Drop-in demo1 usage:
    from Distance.euclidean.score import DistanceCalculator
    calc = DistanceCalculator()
    calc.fit(train_features)
    scores = calc.calculate_distance(test_features)

Optional trajectory usage:
    calc.fit(train_features, train_sequences=train_seq, mask=train_mask, time=train_time)
    scores = calc.calculate_distance(test_features, sequences=test_seq, mask=test_mask, time=test_time)
"""

try:
    from Distance.common import EuclideanDistanceCalculator
except Exception:
    from ..common import EuclideanDistanceCalculator


DistanceCalculator = EuclideanDistanceCalculator


def build_scorer(**kwargs):
    return DistanceCalculator(**kwargs)


__all__ = ["DistanceCalculator", "build_scorer"]
