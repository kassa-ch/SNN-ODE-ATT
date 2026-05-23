# Anomaly Score Benchmark Library

Sinkhorn divergence is only one possible geometry for hidden trajectories. This benchmark adds functional, kernel, energy, and quantum-inspired scores so SNN-ODEATT hidden caches can be evaluated without retraining.

## Hidden Cache Format

```python
{
  "m_reset": Tensor[N, T, D_h],
  "mask": Tensor[N, T],
  "delta_t": Tensor[N, T],
  "tau": Tensor[N, T],
  "label": Tensor[N],
  "split": List[str],
  "sample_id": List[str],
}
```

## Unsupervised Evaluation

Train/validation normal samples estimate references. Thresholds are normal-score quantiles. Test labels are used only for final metrics.

## Change-Point Detection

Any score can be adapted to left-window vs right-window comparison by treating the left window as reference and the right window as the evaluated empirical trajectory.

## Recommended Priority

1. trajectory Mahalanobis
2. ODE residual / action energy
3. MMD
4. Energy distance
5. quantum-inspired density matrix scores
