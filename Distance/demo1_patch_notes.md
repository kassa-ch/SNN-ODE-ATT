# Minimal demo1 replacement patch

Replace the distance calculator selection block in demo1:

```python
# old:
if config['distance_method'] == 'stable_mahalanobis':
    distance_calculator = StableDistanceCalculator(...)
else:
    distance_calculator = RobustDistanceCalculator(...)
```

with:

```python
from Distance.distance_factory import create_distance_calculator

distance_calculator = create_distance_calculator(
    config.get('distance_method', 'stable_mahalanobis'),
    confidence_level=config.get('confidence_level', 0.95),
    robust_covariance=config.get('robust_covariance', False),
)
```

Then use exactly the same calls already in demo1:

```python
distance_calculator.fit(train_features_global)
train_distances = distance_calculator.calculate_distance(train_features_global)
test_distances = distance_calculator.calculate_distance(z_test)
```

For trajectory-aware scoring later, collect `z_seq` for train/test and call:

```python
distance_calculator.fit(
    train_features_global,
    train_sequences=train_z_seq,
    mask=train_mask,
    time=train_time,
)
test_distances = distance_calculator.calculate_distance(
    z_test,
    sequences=test_z_seq,
    mask=test_mask,
    time=test_time,
)
```
