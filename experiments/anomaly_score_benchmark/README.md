# Anomaly Score Benchmark

This experiment compares anomaly scores on a unified hidden cache. It is designed for CPU toy tests and can later be pointed at a real SNN-ODEATT hidden cache.

```bash
python experiments/anomaly_score_benchmark/run_benchmark.py --toy
```

Method families:

- Functional norms: Euclidean latent, Sobolev H1, negative Sobolev H-1.
- Trajectory energies: covariance-aware Mahalanobis, Hamiltonian energy, ODE residual/action energy.
- Kernel distances: MMD and energy distance.
- Quantum-inspired density geometry: trace distance, QRE, entropy, purity, Bures distance.
