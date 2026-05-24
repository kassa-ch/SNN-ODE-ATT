# Distance methods for demo1 replacement

Copy this `Distance/` folder into:

`C:\Users\kassa\Desktop\SNNODEATT\Distance`

Every method has the same demo1-compatible API:

```python
calc.fit(train_features)
scores = calc.calculate_distance(test_features)
```

For trajectory-aware scores:

```python
calc.fit(train_features, train_sequences=train_z_seq, mask=train_mask, time=train_time)
scores = calc.calculate_distance(test_features, sequences=test_z_seq, mask=test_mask, time=test_time)
```

Use factory:

```python
from Distance.distance_factory import create_distance_calculator

distance_calculator = create_distance_calculator("sinkhorn_divergence")
distance_calculator.fit(train_features_global)
train_scores = distance_calculator.calculate_distance(train_features_global)
test_scores = distance_calculator.calculate_distance(z_test)
```

Available canonical method names:

- euclidean
- stable_mahalanobis / mahalanobis / MahalanobisScorer
- rich_global_local / M1+M2 / m1m2_rich_mahalanobis
- sobolev_h1
- sobolev_hminus1
- Gaussian_W2 / gaussian_w2
- Sliced_Wasserstein / sliced_wasserstein
- Sinkhorn_divergence / sinkhorn_divergence
- mmd_cov_kernel
- hamiltonian_energy
- ode_residual_energy
- trace_distance
- qre
- bures
- von_neumann_entropy
- purity
