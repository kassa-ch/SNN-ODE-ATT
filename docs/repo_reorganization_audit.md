# Repository Reorganization Audit

## Executive Summary

Phase R0 audit found a working package-safe pipeline under `src/snnodeatt/` and `experiments/multidata_snnodeatt_distance_benchmark/`, plus earlier public-facing scaffolds under `data/`, `configs/`, `scripts/`, `demos/`, `results/`, and `docs/`.

On Windows, `Data/` and `data/` cannot coexist because the filesystem is case-insensitive. Phase R0 therefore uses `git mv` to promote the dataset scaffold to `Data/`. The same applies to `Results/` and `results/`.

The repository was not reorganized by moving active code. Instead, Phase R0 adds a research-facing organization layer while preserving the current active pipeline.

## Current Tree Summary

Important existing areas before Phase R0:

| Area | Current role | Notes |
|---|---|---|
| `src/snnodeatt/models/` | Active package model code | Includes active SNN-ODE-ATT and baseline sources. |
| `src/snnodeatt/scoring/` | Active distance / score code | Includes registry and anomaly score implementations. |
| `experiments/multidata_snnodeatt_distance_benchmark/` | Active Phase 1/2/2R pipeline | Must not be moved yet. |
| `experiments/anomaly_score_benchmark/` | Existing anomaly score benchmark library | Contains exp4 diagnostic summaries and toy benchmark. |
| `Data/` | Dataset scaffold | README and `.gitkeep`; no real trainable payload found. |
| `configs/experiments/` | Dataset config scaffold | Includes wafer exp1-exp4 and external dataset configs. |
| `scripts/` | Existing CLI-style scripts | Separate from user-requested `Scrips/`. |
| `Results/`, `checkpoints/` | Result/checkpoint placeholders | `results/` promoted to `Results/`; `checkpoints/` kept for compatibility. |

## Duplicated / Confusing Directories

| Duplicate or ambiguity | Risk | Phase R0 handling |
|---|---|---|
| `data/` vs requested `Data/` | Naming mismatch and dataset placement ambiguity | Promoted to `Data/` using `git mv`; legacy old-name dataset variants archived. |
| `scripts/` vs requested `Scrips/` | User research logic uses `Scrips`, Python convention uses `scripts` | New `Scrips/` added with README; old `scripts/` kept. |
| `results/` vs requested `Results/` | Result management ambiguity | Promoted to `Results/` using `git mv`. |
| scoring in `src/snnodeatt/scoring/` vs requested `Distance/` | Paper-facing distance taxonomy not obvious | New `Distance/` wrappers added. |
| baselines under `src/snnodeatt/models/baselines/` vs requested `Models/` | Paper-facing model taxonomy not obvious | New `Models/` wrappers added. |

## Active Pipeline Files

These should not be moved without compatibility wrappers:

- `src/snnodeatt/utils/mask.py`
- `src/snnodeatt/models/snn_odeatt.py`
- `src/snnodeatt/losses/enhanced_loss.py`
- `experiments/multidata_snnodeatt_distance_benchmark/btd_processed_loader.py`
- `experiments/multidata_snnodeatt_distance_benchmark/distance_wrapper.py`
- `experiments/multidata_snnodeatt_distance_benchmark/hidden_cache_extractor.py`
- `experiments/multidata_snnodeatt_distance_benchmark/run_phase2_smoke_training.py`
- `experiments/multidata_snnodeatt_distance_benchmark/run_phase2r_realdata.py`

## Legacy / Recovered Files

Recovered server-side baseline files already present:

- `src/snnodeatt/models/baselines/linear_ar.py`
- `src/snnodeatt/models/baselines/lasso_ar.py`
- `src/snnodeatt/models/baselines/krr.py`
- `src/snnodeatt/models/baselines/fpca.py`
- `src/snnodeatt/models/baselines/cumulative_integral.py`
- `src/snnodeatt/models/baselines/gru.py`
- `src/snnodeatt/models/baselines/latent_ode.py`
- `src/snnodeatt/models/baselines/ode_rnn.py`
- `src/snnodeatt/models/baselines/snn.py`
- `src/snnodeatt/models/baselines/snn_ode_baseline.py`
- `src/snnodeatt/models/baselines/df2m.py`

## Files That Must Not Be Moved Yet

The active Phase 0-2R pipeline is intentionally left in place. Moving it would require import rewrites and new regression tests.

## Files Safe To Reorganize

README, wrapper, and taxonomy files can safely be added under:

- `Data/`
- `Models/`
- `Distance/`
- `Scrips/`
- `Detect/`
- `Results/`
- `Archive/`

## Files Requiring Compatibility Wrappers

| Target taxonomy | Active source | Wrapper needed |
|---|---|---|
| `Models/statistical/*` | `src/snnodeatt/models/baselines/*.py` | Yes |
| `Models/neural_networks/*` | `src/snnodeatt/models/*.py` and baselines | Yes |
| `Distance/*` | `src/snnodeatt/scoring/*.py` | Yes |
| `Scrips/*` | server source or active pipeline | README first; code wrapper later |
