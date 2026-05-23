# Exp4 Score Diagnostic Interpretation

## 1. Executive Summary

This interpretation is based on the real exp4 aligned hidden cache and the integrated diagnostic outputs committed from the server.

- Source cache: `/root/autodl-tmp/wafer_att_ctsr_vca/ctsr_vca/cache_sinkhorn_fix/exp4_hidden_cache_aligned.pt`
- Hidden cache shape: `[374, 78, 256]`
- Split counts: `train_normal=294`, `val_normal=37`, `test_normal=37`, `test_abnormal=6`
- Candidate scores include Sobolev H1, negative Sobolev H-1, trajectory Mahalanobis, MMD, Energy distance, Bures, QRE, Trace distance, Von Neumann entropy, Purity, Hamiltonian energy, and ODE residual energy.
- The default high-score + q0.95 threshold setting fails: it is too conservative and misses abnormal samples.
- Multiple methods have reversed score direction: abnormal samples tend to have lower scores rather than higher scores.
- Threshold tuning gives local improvements, but does not reach the current exp4 best result.
- Local/rich reference variants do not improve the best F1 enough to change the conclusion.
- Final recommendation: stop further tuning of these new scores on wafer exp4. Keep M1+M2 / rich Mahalanobis as the main wafer exp4 method, and move the anomaly score benchmark library to external long-sequence datasets such as HAI, TEP, SWaT, and WADI.

## 2. Baselines

- Post-hoc Sinkhorn: F1=`0.2222`
- Aligned Mahalanobis: TP/FP/TN/FN=`4/4/33/2`, F1=`0.5714`, AUC=`0.9099`
- Current exp4 best M1+M2: TP/FP/TN/FN=`6/3/34/0`, Precision=`0.6667`, Recall=`1.0000`, F1=`0.8000`, F2=`0.9091`, AUC=`0.9955`, PR-AUC=`0.8075`

## 3. Direction Diagnostic

High-direction methods with better or primary high-score ordering include:

- `bures`
- `qre`
- `sobolev_hminus1`
- `trace_distance`

Low-direction methods where AUC(-score) is better than AUC(score) include:

- `energy_distance`
- `euclidean`
- `hamiltonian_energy`
- `mmd_cov_kernel`
- `ode_residual_energy`
- `purity`
- `sobolev_h1`
- `trajectory_mahalanobis`
- `von_neumann_entropy`

This means the score library cannot assume that "larger score = more abnormal". On wafer exp4, several function-space, energy, and entropy-style statistics rank abnormal samples as lower-energy or lower-entropy trajectories. That behavior is itself informative, but it makes a default high-score q0.95 control limit invalid.

## 4. Threshold Diagnostic

The q0.95 threshold is too conservative for this one-fold exp4 setting. The best global grid row is:

- `von_neumann_entropy`, direction=`low`, q=`0.8`, TP/FP/TN/FN=`2/0/37/4`, Precision=`1.0000`, Recall=`0.3333`, F1=`0.5000`, F2=`0.3846`, AUC=`0.6577`, PR-AUC=`0.4761`

This result has zero false positives, but it only detects 2 of 6 abnormal samples. Its F1=`0.5000` remains below the aligned Mahalanobis baseline F1=`0.5714`, and far below the current M1+M2 best F1=`0.8000`.

## 5. High-recall but high-FP cases

The strongest high-recall signal is:

- `hamiltonian_energy`, direction=`low`, q=`0.5`, TP/FP/TN/FN=`5/16/21/1`, Recall=`0.8333`, F1=`0.3704`, F2=`0.5556`

This shows that Hamiltonian energy contains some abnormal-recall signal, but the false-positive cost is too high. It cannot serve as a usable exp4 control limit under the current wafer split.

## 6. Quantum-inspired scores

The quantum-inspired family has some ranking signal, but not enough for an exp4 decision boundary.

- Bures has high-direction AUC around `0.6126` and PR-AUC around `0.2375`.
- Trace distance has high-direction AUC around `0.5991`.
- QRE has high-direction AUC around `0.5676`.
- Von Neumann entropy is strongest in the low direction, with AUC around `0.6577` and PR-AUC around `0.4761`.

Even the strongest quantum-inspired result, Von Neumann entropy in the low direction, reaches only TP/FP/TN/FN=`2/0/37/4` at its best F1 operating point. These scores are useful as negative/ablation evidence, but they do not replace M1+M2 / rich Mahalanobis on wafer exp4.

## 7. Why These Scores Still Fail on Wafer Exp4

1. The current SNN-ODEATT hidden representation is better aligned with aggregated Mahalanobis/rich scoring than with whole-trajectory point-cloud, energy, or density-matrix statistics.
2. The new scores compare full hidden trajectories, function-space residuals, or density matrix geometry, but exp4 abnormalities do not form a strong separable cluster under those statistics.
3. Normal and abnormal score distributions overlap heavily.
4. Direction reversal shows that abnormal samples are not consistently "farther", "higher energy", or "higher entropy" than normal samples.
5. Local/rich reference variants do not materially improve the best F1, so the issue is not only that the global reference is too coarse.
6. Exp4 has only 6 test abnormal samples, making threshold selection extremely sensitive.
7. Continuing to tune these scores on wafer exp4 risks overfitting and is not suitable as the main line.

## 8. Decision

Decision C: All candidate scores remain weak on wafer exp4. Stop further score tuning on wafer exp4. Keep current M1+M2 / rich Mahalanobis as the main wafer exp4 scoring method. Use the anomaly score benchmark library as an extension for external long-sequence industrial datasets.

## 9. Next Recommended Work

1. Do not expand the wafer exp4 score grid further.
2. Keep these diagnostic results as negative/ablation evidence.
3. Start external dataset integration for HAI, TEP, SWaT, and WADI.
4. Suggested priority: TEP -> HAI -> SWaT -> WADI.
5. On external datasets, focus on state accumulation anomalies, change-point detection, long-window dynamic drift, multivariate coupling changes, and whether quantum-inspired density matrix or Hamiltonian energy scores outperform Mahalanobis.

## 10. Files

- Source report: `docs/exp4_score_diagnostic_report.md`
- Source CSV: `experiments/anomaly_score_benchmark/score_diagnostic_exp4.csv`
- Diagnostic script: `experiments/anomaly_score_benchmark/run_score_diagnostic.py`
- Top summary CSV: `experiments/anomaly_score_benchmark/score_diagnostic_exp4_top_summary.csv`

## Appendix: Parsed Diagnostic Checks

- Best F1: `von_neumann_entropy`, direction=`low`, q=`0.8`, TP/FP/TN/FN=`2/0/37/4`, Precision=`1.0000`, Recall=`0.3333`, F1=`0.5000`, F2=`0.3846`, AUC=`0.6577`, PR-AUC=`0.4761`
- Best F2: `hamiltonian_energy`, direction=`low`, q=`0.5`, TP/FP/TN/FN=`5/16/21/1`, Precision=`0.2381`, Recall=`0.8333`, F1=`0.3704`, F2=`0.5556`, AUC=`0.6216`, PR-AUC=`0.2040`
- Best AUC: `von_neumann_entropy`, direction=`low`, q=`0.8`, TP/FP/TN/FN=`2/0/37/4`, Precision=`1.0000`, Recall=`0.3333`, F1=`0.5000`, F2=`0.3846`, AUC=`0.6577`, PR-AUC=`0.4761`
- Best PR-AUC: `von_neumann_entropy`, direction=`low`, q=`0.8`, TP/FP/TN/FN=`2/0/37/4`, Precision=`1.0000`, Recall=`0.3333`, F1=`0.5000`, F2=`0.3846`, AUC=`0.6577`, PR-AUC=`0.4761`
- Methods with Recall >= 0.5: `21` rows
- Methods with FP <= 3: `9` rows
- Methods exceeding aligned Mahalanobis F1=0.5714: `0`
- Methods approaching exp4 best F1=0.8000: `0`
- Methods satisfying TP>=5 and FP<=3: `0`
- Methods satisfying Recall>=0.67 and FP<=3: `0`
