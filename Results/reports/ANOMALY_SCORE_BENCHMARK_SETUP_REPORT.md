# ANOMALY_SCORE_BENCHMARK_SETUP_REPORT

## 1. ??????
- experiments/anomaly_score_benchmark/README.md
- experiments/anomaly_score_benchmark/run_benchmark.py
- experiments/anomaly_score_benchmark/benchmark_config.yaml
- experiments/anomaly_score_benchmark/score_method_summary.md
- src/snnodeatt/scoring/functional_scores.py
- src/snnodeatt/scoring/kernel_scores.py
- src/snnodeatt/scoring/energy_scores.py
- src/snnodeatt/scoring/quantum_scores.py
- src/snnodeatt/scoring/trajectory_energy.py
- src/snnodeatt/scoring/score_registry.py
- src/snnodeatt/scoring/score_benchmark.py
- docs/anomaly_score_benchmark.md
- docs/quantum_inspired_scores.md
- tests/test_functional_scores.py
- tests/test_kernel_scores.py
- tests/test_quantum_scores.py
- tests/test_score_benchmark.py
- configs/scoring/anomaly_score_benchmark.yaml

## 2. Scoring ??????
- euclidean: ?? toy/reference ???
- sobolev_h1: ??????????? value + derivative?
- sobolev_hminus1: FFT spectral approximation?
- trajectory_mahalanobis: diagonal covariance trajectory energy???????
- mmd_cov_kernel: diagonal covariance RBF MMD?prototype reference?
- energy_distance: weighted empirical energy distance?prototype reference?
- trace_distance/qre/entropy/purity/bures: quantum-inspired density matrix geometry?
- hamiltonian_energy: diagonal inverse covariance Hamiltonian-like energy?
- ode_residual_energy/action: finite-difference residual approximation???? trained ODE ???? dot_m - f_ODE(m)?

## 3. ??????
- Hminus1 ?? torch.fft.rfft ?????
- MMD/Energy ???? fixed random prototype reference????? normal ????
- Quantum scores ? density-matrix inspired??????????????
- ODE residual ??? finite-difference generic residual???????? ODEFunc?

## 4. Toy benchmark ??
?????sobolev_hminus1 auc=1.0000 f1=1.0000 TP/FP/TN/FN=4/0/4/0

?? toy ????????
- experiments/anomaly_score_benchmark/outputs/metrics_table.csv
- experiments/anomaly_score_benchmark/outputs/scores.csv
- experiments/anomaly_score_benchmark/outputs/report.md

?? toy ????? .gitignore ??????? outputs/.gitkeep?

## 5. ????
- compileall exit: 0
- test_functional_scores exit: 0
- test_kernel_scores exit: 0
- test_quantum_scores exit: 0
- test_score_benchmark exit: 0
- toy benchmark exit: 0

## 6. ????
???????
```text
No sensitive strings found.
```

??????
```text
No files over 50M outside .git.
```

## 7. ???????? exp4 aligned cache ???
???
```bash
cd /root/autodl-tmp/SNNODEATT
python experiments/anomaly_score_benchmark/run_benchmark.py \
  --cache_path /root/autodl-tmp/wafer_att_ctsr_vca/ctsr_vca/cache_sinkhorn_fix/exp4_hidden_cache_aligned.pt \
  --output_dir experiments/anomaly_score_benchmark/outputs_exp4_local
```

????? exp4 ???????? .gitignore????? CSV?

## 8. GitHub push
??? push???? GitHub ?????????
