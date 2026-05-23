# Phase R0: Repository Reorganization and Mapping Report

## 1. Executive Summary

Phase R0 created a research-facing repository organization layer:

- `Data/`
- `Models/`
- `Distance/`
- `Scrips/`
- `Detect/`
- `Results/`
- `Archive/`

The active package-safe pipeline remains in `src/snnodeatt/` and `experiments/multidata_snnodeatt_distance_benchmark/`. No training was launched, no model math was changed, and no new distance method was added.

Server code backfill status: `SERVER_SOURCE_NOT_FOUND`. The requested paths were not accessible in the current local Codex environment:

- `/autodl-tmp/半导体晶圆项目ATT/`
- `/root/autodl-tmp/半导体晶圆项目ATT/`

## 2. Current Problem

The previous repository layout mixed several concerns:

- public dataset scaffolds under lowercase `data/`;
- active package implementation under `src/snnodeatt/`;
- old scripts under `scripts/`;
- benchmark experiments under `experiments/`;
- result placeholders under `results/`;
- paper-facing method names only partially visible from source paths.

Phase R0 does not move active source code. It promotes lowercase `data/` to `Data/` and lowercase `results/` to `Results/` via `git mv` because Windows cannot maintain both case variants in one checkout.

## 3. New Directory Structure

```text
Data/
  Wafer/exp1..exp4/{raw,processed,manifests}
  HAI/{raw,processed,manifests}
  SWaT/{raw,processed,manifests}
  WADI/{raw,processed,manifests}
  TEP/{raw,processed,manifests}
  ST-AWFD/{raw,processed,manifests}
  Bosch_Production_Line/{raw,processed,manifests}
Models/
  statistical/{LinearAR,LassoAR,KRR,FPCA,CumInt,_legacy}
  neural_networks/{GRU,LSTM,Latent_ODE,ODE_RNN,SNN,SNNODE,SNNODE_ATT}
Distance/
  MahalanobisScorer, Sinkhorn_divergence, Gaussian_W2, Sliced_Wasserstein
  Rich_global_local, euclidean, sobolev_h1, sobolev_hminus1
  mmd_cov_kernel, trace_distance, qre, von_neumann_entropy, purity, bures
  hamiltonian_energy, ode_residual_energy
Scrips/
  config, data_loader, data_lader, model_factory, utils, enhanced_loss, train, main
Detect/
  demo1, demo2, demo3, demo4
Results/
  figures, tables, models, logs, reports, hidden_cache, benchmarks, diagnostics, manifests
Archive/
  legacy_repo_structure
```

Compatibility note: previous `data/ST_AWFD` and `data/BoschProductionLine` scaffold names were moved to `Archive/legacy_repo_structure/data_old_names/`. The canonical new names are `Data/ST-AWFD` and `Data/Bosch_Production_Line`.

## 4. Data Directory

`Data/` is the new research-facing dataset landing area. It does not contain real data. Each dataset README states the expected BTD payload:

```text
x       [N,T,D]
mask    [N,T]
delta_t [N,T]
time    [N,T]
label   [N]
split   [N] or list[str]
```

Wafer experiment meanings:

| Experiment | Meaning |
|---|---|
| exp1 | augmented anomalies, no Poisson |
| exp2 | original anomalies, no Poisson |
| exp3 | augmented anomalies, Poisson/nonuniform |
| exp4 | original anomalies, Poisson/nonuniform |

All dataset README files currently state `STATUS: NO_REAL_DATA_IN_REPO`.

## 5. Models Directory

`Models/` is a paper-facing method taxonomy. It uses lightweight wrappers pointing to existing source files.

### Statistical Model Mapping

| Method | Source implementation | New wrapper | Status |
|---|---|---|---|
| LinearAR | `src/snnodeatt/models/baselines/linear_ar.py` | `Models/statistical/LinearAR/model.py` | WRAPPER |
| LassoAR | `src/snnodeatt/models/baselines/lasso_ar.py` | `Models/statistical/LassoAR/model.py` | WRAPPER |
| KRR | `src/snnodeatt/models/baselines/krr.py` | `Models/statistical/KRR/model.py` | WRAPPER |
| FPCA | `src/snnodeatt/models/baselines/fpca.py` | `Models/statistical/FPCA/model.py` | WRAPPER |
| CumInt | `src/snnodeatt/models/baselines/cumulative_integral.py` | `Models/statistical/CumInt/model.py` | WRAPPER |

Requested `baseline.py` split status: `SERVER_SOURCE_NOT_FOUND`. No server `baseline.py` was available, so full source-level splitting is pending.

### Neural Network Model Mapping

| Method | Source implementation | New wrapper | Status |
|---|---|---|---|
| GRU | `src/snnodeatt/models/baselines/gru.py` | `Models/neural_networks/GRU/model.py` | WRAPPER |
| LSTM | not found as standalone | `Models/neural_networks/LSTM/README.md` | NOT_IMPLEMENTED_OR_NOT_FOUND |
| Latent ODE | `src/snnodeatt/models/baselines/latent_ode.py` | `Models/neural_networks/Latent_ODE/model.py` | WRAPPER |
| ODE-RNN | `src/snnodeatt/models/baselines/ode_rnn.py` | `Models/neural_networks/ODE_RNN/model.py` | WRAPPER |
| SNN | `src/snnodeatt/models/baselines/snn.py` | `Models/neural_networks/SNN/model.py` | WRAPPER |
| SNNODE | `src/snnodeatt/models/baselines/snn_ode_baseline.py` | `Models/neural_networks/SNNODE/model.py` | WRAPPER |
| SNNODE_ATT | `src/snnodeatt/models/snn_odeatt.py` | `Models/neural_networks/SNNODE_ATT/model.py` | WRAPPER |

## 6. Distance Directory

`Distance/` maps canonical distance names to active scoring implementations.

| Canonical name | Source implementation | Wrapper path | Status |
|---|---|---|---|
| MahalanobisScorer | `src/snnodeatt/scoring/mahalanobis.py` | `Distance/MahalanobisScorer/score.py` | WRAPPER |
| Sinkhorn divergence | `src/snnodeatt/scoring/sinkhorn.py` | `Distance/Sinkhorn_divergence/score.py` | WRAPPER |
| Gaussian W2 | `src/snnodeatt/scoring/gaussian_w2.py` | `Distance/Gaussian_W2/score.py` | WRAPPER |
| Sliced Wasserstein | `src/snnodeatt/scoring/sliced_wasserstein.py` | `Distance/Sliced_Wasserstein/score.py` | WRAPPER |
| Rich global-local | `src/snnodeatt/scoring/rich_global_local.py` | `Distance/Rich_global_local/score.py` | WRAPPER |
| euclidean | `src/snnodeatt/scoring/functional_scores.py` | `Distance/euclidean/score.py` | WRAPPER |
| sobolev_h1 | `src/snnodeatt/scoring/functional_scores.py` | `Distance/sobolev_h1/score.py` | WRAPPER |
| sobolev_hminus1 | `src/snnodeatt/scoring/functional_scores.py` | `Distance/sobolev_hminus1/score.py` | WRAPPER |
| mmd_cov_kernel | `src/snnodeatt/scoring/kernel_scores.py` | `Distance/mmd_cov_kernel/score.py` | WRAPPER |
| trace_distance | `src/snnodeatt/scoring/quantum_scores.py` | `Distance/trace_distance/score.py` | WRAPPER |
| qre | `src/snnodeatt/scoring/quantum_scores.py` | `Distance/qre/score.py` | WRAPPER |
| von_neumann_entropy | `src/snnodeatt/scoring/quantum_scores.py` | `Distance/von_neumann_entropy/score.py` | WRAPPER |
| purity | `src/snnodeatt/scoring/quantum_scores.py` | `Distance/purity/score.py` | WRAPPER |
| bures | `src/snnodeatt/scoring/quantum_scores.py` | `Distance/bures/score.py` | WRAPPER |
| hamiltonian_energy | `src/snnodeatt/scoring/trajectory_energy.py` | `Distance/hamiltonian_energy/score.py` | WRAPPER |
| ode_residual_energy | `src/snnodeatt/scoring/trajectory_energy.py` | `Distance/ode_residual_energy/score.py` | WRAPPER |

Phase 1 no-leak wrapper remains unchanged at `experiments/multidata_snnodeatt_distance_benchmark/distance_wrapper.py`.

## 7. Scrips Directory

`Scrips/` is preserved because it matches the user's requested research organization. `Scrips/data_loader/` is the standard code spelling. `Scrips/data_lader/` is intentionally a README-only compatibility note.

Server source files were not accessible, so `Scrips/*` directories contain README placeholders and status notes rather than copied server code.

## 8. Detect Directory

`Detect/demo1` through `Detect/demo4` are placeholders for:

- offline anomaly scoring demo;
- prefix / online scoring demo;
- external dataset detection demo;
- visualization demo.

No UI or demo training logic was added in Phase R0.

## 9. Results Directory

`Results/` centralizes future experiment outputs. `.gitignore` excludes heavy result subdirectories while preserving README and `.gitkeep` files.

Do not commit:

- checkpoints;
- best model weights;
- hidden caches;
- raw benchmark dumps;
- logs containing secrets.

## 10. Server Code Backfill Status

| Requested source | Status |
|---|---|
| `/autodl-tmp/半导体晶圆项目ATT/` | MISSING |
| `/root/autodl-tmp/半导体晶圆项目ATT/` | MISSING |
| server `models/` | PENDING |
| server `baseline.py` | PENDING |
| server `config.py` | PENDING |
| server `data_loader.py` | PENDING |
| server `model_factory.py` | PENDING |
| server `enhanced_loss.py` | PENDING |
| server `train.py` | PENDING |
| server `main.py` | PENDING |
| server `utils/` | PENDING |

No files were fabricated. Existing recovered source under `src/snnodeatt/` was mapped through wrappers.

## 11. Migration Mapping

| Old / active path | New path | Reason |
|---|---|---|
| `src/snnodeatt/models/baselines/linear_ar.py` | `Models/statistical/LinearAR/model.py` | Paper-facing statistical taxonomy |
| `src/snnodeatt/models/baselines/lasso_ar.py` | `Models/statistical/LassoAR/model.py` | Paper-facing statistical taxonomy |
| `src/snnodeatt/models/baselines/krr.py` | `Models/statistical/KRR/model.py` | Paper-facing statistical taxonomy |
| `src/snnodeatt/models/baselines/fpca.py` | `Models/statistical/FPCA/model.py` | Paper-facing statistical taxonomy |
| `src/snnodeatt/models/baselines/cumulative_integral.py` | `Models/statistical/CumInt/model.py` | Paper-facing statistical taxonomy |
| `src/snnodeatt/models/baselines/gru.py` | `Models/neural_networks/GRU/model.py` | Paper-facing neural taxonomy |
| `src/snnodeatt/models/baselines/latent_ode.py` | `Models/neural_networks/Latent_ODE/model.py` | Paper-facing neural taxonomy |
| `src/snnodeatt/models/baselines/ode_rnn.py` | `Models/neural_networks/ODE_RNN/model.py` | Paper-facing neural taxonomy |
| `src/snnodeatt/models/baselines/snn.py` | `Models/neural_networks/SNN/model.py` | Paper-facing neural taxonomy |
| `src/snnodeatt/models/baselines/snn_ode_baseline.py` | `Models/neural_networks/SNNODE/model.py` | Paper-facing neural taxonomy |
| `src/snnodeatt/models/snn_odeatt.py` | `Models/neural_networks/SNNODE_ATT/model.py` | Main method mapping |
| `src/snnodeatt/scoring/*.py` | `Distance/*/score.py` | Paper-facing distance taxonomy |
| `data/` | `Data/` | User-requested top-level dataset structure |
| `results/` | `Results/` | User-requested top-level result structure |
| `data/ST_AWFD` | `Archive/legacy_repo_structure/data_old_names/ST_AWFD` | Preserve old scaffold name without duplicate active dataset directory |
| `data/BoschProductionLine` | `Archive/legacy_repo_structure/data_old_names/BoschProductionLine` | Preserve old scaffold name without duplicate active dataset directory |

## 12. Active Pipeline Preservation

These active paths were not moved:

- `src/snnodeatt/`
- `experiments/multidata_snnodeatt_distance_benchmark/`
- `experiments/anomaly_score_benchmark/`
- `tests/`

## 13. Legacy / Archive

`Archive/legacy_repo_structure/` was created but no active files were moved into it. Future phases may use `git mv` only after compatibility wrappers and tests are ready.

## 14. Missing Files

The full server source requested in Phase R0 was not accessible. The next phase should either run this task on the server or provide a reachable copy of:

- `models/`
- `baseline.py`
- `config.py`
- `data_loader.py`
- `model_factory.py`
- `enhanced_loss.py`
- `train.py`
- `main.py`
- `utils/`

## 15. Large File / Sensitive Information Handling

No large data/checkpoint/cache files were intentionally added. `.gitignore` now covers `Data/**/raw`, `Data/**/processed`, `Data/**/manifests`, and heavy `Results/` subdirectories while preserving README and `.gitkeep` files.

## 16. Follow-up Recommendation

1. Phase R1: place real wafer BTD payloads under `Data/Wafer/exp*/processed/`.
2. Phase 2R2: rerun real BTD smoke using the Phase 2R converter and smoke guard.
3. Phase 3: run full single-dataset training only after real smoke passes.

## 17. Decision

```text
GO_TO_PHASE_R1_DATA_PLACEMENT = YES
```

Reason: repository structure is ready and active pipeline remains intact. Server backfill is pending because the source path was unavailable.
