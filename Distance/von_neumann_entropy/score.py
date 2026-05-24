"""
von_neumann_entropy score module.

Drop-in demo1 usage:
    from Distance.von_neumann_entropy.score import DistanceCalculator
    calc = DistanceCalculator()
    calc.fit(train_features)
    scores = calc.calculate_distance(test_features)

Optional trajectory usage:
    calc.fit(train_features, train_sequences=train_seq, mask=train_mask, time=train_time)
    scores = calc.calculate_distance(test_features, sequences=test_seq, mask=test_mask, time=test_time)
"""

try:
    from Distance.common import VonNeumannEntropyCalculator
except Exception:
    from ..common import VonNeumannEntropyCalculator


DistanceCalculator = VonNeumannEntropyCalculator


def build_scorer(**kwargs):
    return DistanceCalculator(**kwargs)


__all__ = ["DistanceCalculator", "build_scorer"]
