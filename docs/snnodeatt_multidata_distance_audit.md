# SNNODEATT Multidata Distance Scoring Phase 0 Audit

## Executive Summary

Phase 0 only audited repository structure, data flow, SNNODEATT forward behavior, `delta_t` usage, mask handling, and existing scoring methods. No training was launched, no model math was changed, the ATT block was not split, and no new distance method was added.

Current conclusion:

```text
GO_TO_TRAINING = NO
```

The repository already contains valuable pieces for the intended mainline:

- `src/snnodeatt/models/snn_odeatt.py` defines the current full SNNODEATT implementation.
- SNNODEATT returns per-time tensors: `mem_seq [B,T,H]`, `mem_reset_seq [B,T,H]`, `rate_seq [B,T,H]`, and `h_seq [B,T,2H]`.
- `delta_t` is used inside the SNN-ODE local cell and causal time-weighted attention.
- The anomaly-score benchmark library already supports hidden-cache scoring methods over `m_reset [N,T,H]`, `mask [N,T]`, `delta_t [N,T]`, and `tau [N,T]`.

However, several blockers prevent safe training or formal multidataset benchmarking now:

- The packaged wafer dataloader import fails because `src/snnodeatt/data/wafer_loader.py` imports missing top-level `utils`.
- `src/snnodeatt/train/trainer.py` imports old top-level modules (`data_loader`, `config`, `enhanced_loss`, `model_factory`) and is not package-ready.
- The current wafer dataloader emits `mask [B,T,D]`, not the target sample-level `mask [B,T]`.
- SNNODEATT forward currently fails for target `mask [B,T]`; it only works with legacy feature-level `mask [B,T,D]`.
- `time/tau [B,T]` is not returned by the current dataloader; only `delta_t` is returned and cumulative time is used internally for positional encoding.
- Most multidataset prepare scripts and dataset loaders are scaffolds, not executable converters.
- Existing score benchmark uses train+val normal together as the reference/threshold pool; the requested stricter protocol needs train normal for reference and val normal for threshold.

The safe next step is a minimal adaptation layer, not training.

## Commands Run

Read-only / lightweight checks:

```text
git status --short
git log --oneline -8
rg --files
rg -n "poisson_sampling_series_segmented|custom_collate|mask|delta_t|delta_ts|time|tau|BatchNorm|LayerNorm|key_padding|causal|mean\\(|softmax|forward\\(" src scripts experiments tests configs README.md docs/data_format.md
rg -n "class .*Dataset|DataLoader|collate|processed|manifest|label_column|train_split|val_split|test_split|normal|anomaly|Response|target" src configs data scripts experiments tests
rg -n "def .*score|class .*Scorer|SCORE_REGISTRY|m_reset|delta_t|tau|time_weight|reference_mask|eval_mask|weighted_latent|Mahalanobis|sinkhorn|bures|mmd|energy|sobolev" src/snnodeatt/scoring tests experiments/anomaly_score_benchmark
python import smoke tests for model, scoring registry, wafer_loader, loss, trainer
synthetic SNNODEATT forward smoke with mask [B,T] and mask [B,T,D]
toy score benchmark smoke for first registry methods
```

Smoke-test summary:

```text
snnodeatt.models.snn_odeatt import: OK
snnodeatt.scoring.score_registry import: OK
snnodeatt.losses.enhanced_loss import: OK
snnodeatt.data.wafer_loader import: FAIL, ModuleNotFoundError: No module named 'utils'
snnodeatt.train.trainer import: FAIL, ModuleNotFoundError: No module named 'data_loader'

target mask [B,T] forward: FAIL
legacy mask [B,T,D] forward: OK

legacy synthetic shapes:
x.shape = (2, 5, 4)
mask.shape = (2, 5, 4)
delta_t.shape = (2, 5)
time.shape = (2, 5), computed as cumsum(delta_t)
label.shape = (2,)
valid length min/max = 3 / 5
padding_tail_ok = True
delta_t_nonnegative = True

SNNODEATT outputs with mask [B,T,D]:
recons = (2, 5, 4)
preds = (2, 5, 4)
z_mean = (2, 8)
mem_seq = (2, 5, 8)
mem_reset_seq = (2, 5, 8)
rate_seq = (2, 5, 8)
h_seq = (2, 5, 16)
```

## Dataset Paths and Configs

| dataset | raw path | processed path | manifest path | config path | dataloader / class | label source | split source | normal/anomaly rule | BTD readiness |
|---|---|---|---|---|---|---|---|---|---|
| wafer exp1 | `data/wafer/raw` | `data/wafer/processed` | `data/wafer/manifests` | `configs/experiments/wafer_exp1.yaml` | `src/snnodeatt/data/wafer_loader.py`, `TimeSeriesDataset` | filename convention in legacy loader, not config field | legacy K-fold functions in `src/snnodeatt/train/trainer.py` | `_normal.csv` / `_abnormal.csv` suffix in `find_csv_paths` | PARTIAL; config exists but `data/wafer` is absent and loader import fails |
| wafer exp2 | `data/wafer/raw` | `data/wafer/processed` | `data/wafer/manifests` | `configs/experiments/wafer_exp2.yaml` | same as wafer exp1 | same as wafer exp1 | same as wafer exp1 | same as wafer exp1 | PARTIAL |
| wafer exp3 | `data/wafer/raw` | `data/wafer/processed` | `data/wafer/manifests` | `configs/experiments/wafer_exp3.yaml` | same as wafer exp1 | same as wafer exp1 | same as wafer exp1 | same as wafer exp1 | PARTIAL; Poisson mentioned but packaged sampler import missing |
| wafer exp4 | `data/wafer/raw` | `data/wafer/processed` | `data/wafer/manifests` | `configs/experiments/wafer_exp4.yaml` | same as wafer exp1 | same as wafer exp1 | same as wafer exp1 | same as wafer exp1 | PARTIAL; exact replay artifacts are not a general dataloader |
| HAI | `data/HAI/raw` | `data/HAI/processed` | physical dir exists: `data/HAI/manifests`; no `manifest_dir` in config | `configs/experiments/hai.yaml` | `src/snnodeatt/data/hai_loader.py` is a stub | `label_column: label` | `train_split`, `val_split`, `test_split` config fields | UNKNOWN; converter not implemented | NO; scaffold only |
| SWaT | `data/SWAT/raw` in config, but repo directory is `data/SWaT` | `data/SWAT/processed` in config, but repo directory is `data/SWaT` | physical dir exists under `data/SWaT/manifests`; no `manifest_dir` in config | `configs/experiments/swat.yaml` | `src/snnodeatt/data/swat_loader.py` is a stub | `label_column: label` | config fields | UNKNOWN | NO; path-case mismatch plus stub |
| WADI | `data/WADI/raw` | `data/WADI/processed` | physical dir exists: `data/WADI/manifests`; no `manifest_dir` in config | `configs/experiments/wadi.yaml` | `src/snnodeatt/data/wadi_loader.py` is a stub | `label_column: label` | config fields | UNKNOWN | NO; scaffold only |
| TEP | `data/TEP/raw` | `data/TEP/processed` | physical dir exists: `data/TEP/manifests`; no `manifest_dir` in config | `configs/experiments/tep.yaml` | `src/snnodeatt/data/tep_loader.py` is a stub | `label_column: label` | config fields | UNKNOWN | NO; scaffold only |
| ST-AWFD | `data/ST_AWFD/raw` | `data/ST_AWFD/processed` | `data/ST_AWFD/manifests` | `configs/experiments/st_awfd.yaml` | no dataset class; `scripts/prepare_st_awfd.py` is a scaffold | `target` | config fields | UNKNOWN until converter maps labels | NO; scaffold only |
| Bosch Production Line | `data/BoschProductionLine/raw` | `data/BoschProductionLine/processed` | `data/BoschProductionLine/manifests` | `configs/experiments/bosch_production_line.yaml` | no dataset class; `scripts/prepare_bosch.py` is a scaffold | `Response` | config fields | UNKNOWN until converter maps labels | NO; staged BTD scaffold only |

Dataset audit conclusion:

- Only the legacy wafer loader contains actual dataset logic.
- HAI, SWaT, WADI, TEP, ST-AWFD, and Bosch are documented entrypoints but not executable BTD loaders yet.
- The desired common processed payload is documented in `data/README.md` and `docs/data_format.md`, but there is no unified loader that consumes it.
- No real processed data or manifest files are committed, as expected for GitHub hygiene.

## Dataloader / Padding / Mask / Delta_t Audit

Current concrete implementation:

- File: `src/snnodeatt/data/wafer_loader.py`
- Class: `TimeSeriesDataset`
- Collate: `custom_collate`

Observed behavior:

- `TimeSeriesDataset.__getitem__` reads a CSV into `x [T,D_raw]`.
- It performs per-sample standardization with `x.mean(dim=0)` and `x.std(dim=0)` over the whole sample.
- If `apply_poisson_sampling=True`, it calls `poisson_sampling_series_segmented`, but this function is imported from missing top-level `utils`, so packaged import fails.
- If `apply_poisson_sampling=False`, `delta_t = ones[T]` and `mask = ones_like(x)`, i.e. `mask [T,D]`.
- It computes `times = cumsum(delta_t)` and appends sine/cosine positional encoding to `x`.
- It pads the mask with ones for the added positional-encoding dimensions.
- `custom_collate` pads `x`, `mask`, and `delta_t` at the sequence tail.

Requirement-by-requirement audit:

| requirement | current status | evidence / note |
|---|---|---|
| compact observed sequence after Poisson | UNKNOWN | sampler is missing from package; cannot verify whether sampled output is compact `[x(t1),x(t3),x(t6)]` or sparse-with-gaps |
| padding only at sequence tail | PASS for `custom_collate` | collate appends zero rows after each shorter sample |
| sample-level `mask [B,T]` | FAIL | current wafer loader emits feature-level `mask [B,T,D_aug]` |
| `delta_t [B,T]` | PASS in collate shape | current collate stacks `[T_i]` into `[B,T]` |
| `delta_t[b,k] = time[b,k]-time[b,k-1]` | UNKNOWN for Poisson; PARTIAL for uniform | uniform uses ones, not physical time; Poisson depends on missing sampler |
| padding `delta_t = 0` | PASS in collate | padding delta rows are zeros |
| `time/tau [B,T]` returned | FAIL | `time` is only used internally for positional encoding; not returned |
| scaler fit train only | FAIL / RISK | current loader standardizes each sample using its own full sequence, not a train-fitted scaler |
| dense approximation leakage | UNKNOWN | dense/PCHIP logic is not present in current GitHub package |
| Poisson view label leakage | UNKNOWN | sampler unavailable; no label input seen in current loader |
| test leakage to train | UNKNOWN | split functions exist for wafer normal paths; no unified manifest loader to verify for all datasets |

Dataflow blockers:

- Missing packaged sampler import: `from utils import poisson_sampling_series_segmented`.
- Missing unified processed-cache loader for the documented payload.
- No `time/tau` output from dataloader.
- Current `mask [B,T,D]` is incompatible with the requested mainline `mask [B,T]`.
- Per-sample standardization may erase amplitude anomalies and is not a train-only scaler. For online prefix work it also sees future points inside the same sample.

Minimal fix before training:

- Add a unified BTD processed dataset loader that reads `.pt` payloads with `x`, `mask`, `delta_t`, `tau/time`, `label`, `split`, and `sample_id`.
- Add a compatibility adapter for legacy wafer CSVs, but do not use it as the multidataset canonical loader.
- Convert the old feature-level mask to sample-level mask by `valid_time = mask.mean(-1) > 0` only inside a compatibility wrapper, not as the long-term data contract.
- Use train normal only for scaler fitting. Store scaler parameters in the manifest or processed artifact metadata.

## SNNODEATT Forward Audit

Current model file:

- `src/snnodeatt/models/snn_odeatt.py`

Classes:

- `ODEFunc`
- `ContinuousSNNODECell`
- `ContinuousAttentionSNNODE`
- alias: `PredictiveSNN_ODEATT_Model = ContinuousAttentionSNNODE`

Forward signature:

```python
def forward(self, x, mask, delta_t=None, delta_ts=None):
```

Inputs:

- `x`: expected `[B,T,D]`.
- `mask`: accepted by implementation, but current code works reliably only with `[B,T,D]`; `[B,T]` fails because `mask[:, t]` is `[B]` and is multiplied with `h_t [B,2H]` without unsqueeze.
- `delta_t` / `delta_ts`: normalized to `[B,T]`.
- `time`: not accepted.

Current outputs:

```python
return recons, preds, z_mean, mem_seq, mem_reset_seq, rate_seq, h_seq
```

Output semantics from smoke test with hidden_dim=8:

- `recons [B,T,D]`
- `preds [B,T,D]`
- `z_mean [B,H]`
- `mem_seq [B,T,H]`
- `mem_reset_seq [B,T,H]`
- `rate_seq [B,T,H]`
- `h_seq [B,T,2H]`, concatenation of `m_reset` and `rate_t`

Mask concatenation into `x`:

- No direct evidence that `mask` is concatenated into `x`.
- However, positional encoding is appended to `x` in the dataloader, and mask is padded with ones for those positional dimensions. This makes `mask` a feature-level mask and should be replaced by a clean sample-level mask in the multidataset path.

Time positional encoding:

- Current wafer dataloader appends sine/cosine positional encoding to `x`.
- SNNODEATT itself does not accept a separate `time` tensor.
- This can remain a config behavior, but the canonical multidataset interface should preserve raw `time/tau` separately for distance scoring and prefix evaluation.

Forward audit conclusion:

- Per-time hidden trajectory is available.
- Optional event-like information is available as `rate_seq` and `h_seq`.
- The target sample-level `mask [B,T]` is a forward blocker.
- The model does not return a dict, but the tuple is sufficient for a wrapper to build:

```python
{
    "hidden": mem_reset_seq,
    "membrane": mem_seq,
    "post_reset": mem_reset_seq,
    "spike": rate_seq,
    "membrane_spike": h_seq,
    "pooled_latent": z_mean,
}
```

No model-math change is needed; only a readout/extraction adapter is needed after mask support is fixed.

## Delta_t Into ODE / RK4 / Decay Audit

Status:

```text
PARTIAL PASS
```

Evidence in `src/snnodeatt/models/snn_odeatt.py`:

- `ContinuousAttentionSNNODE.forward` normalizes `delta_t/delta_ts` to `[B,T]`.
- At each time step it calls:

```python
self.cell(x[:, t], s, m_reset, r_prev, delta_ts[:, t])
```

- `ContinuousSNNODECell.forward` computes:

```python
dt = delta_t.view(bsz, 1).clamp_min(1e-6)
```

- Synaptic dynamics use `dt`:

```python
ds_dt = -gamma_s * s_t + syn_drive
s_next = norm_syn(s_t + dt * ds_dt)
```

- RK4-like ODE evaluation uses `dt` inside intermediate states:

```python
k1 = ode_func(0.0, m_prev_reset)
k2 = ode_func(0.0, m_prev_reset + 0.5 * dt * k1)
k3 = ode_func(0.0, m_prev_reset + 0.5 * dt * k2)
k4 = ode_func(0.0, m_prev_reset + dt * k3)
```

- Membrane update uses `dt`:

```python
d_m_dt = -gamma_m * m_prev_reset + syn_to_mem(s_next) + alpha_ode * ode_to_mem(f_ode_rk) - threshold * r_prev + mem_bias
m_local_pre = norm_mem(m_prev_reset + dt * d_m_dt)
```

- Causal attention uses `dt_hist` as Riemann weights:

```python
unnorm = exp(scores) * dt_weights
attn = unnorm / sum(unnorm)
```

What passes:

- `delta_t` truly enters local state update.
- `delta_t` enters synaptic update.
- `delta_t` enters membrane update.
- `delta_t` enters the RK4-like intermediate states.
- Attention is time-weighted by `delta_t`.
- No fixed `dt=1` appears in SNNODEATT when a valid `delta_t` is provided.

Risks / limitations:

- `ODEFunc.forward(t, m)` ignores `t`; the ODE is autonomous. This is acceptable if claimed as autonomous continuous-time dynamics, but not as explicitly time-dependent dynamics.
- There is no external `odeint`; the implementation is a custom RK4-like numerical step.
- `delta_t=0` is clamped to `1e-6`, so padding steps are not mathematically frozen by `dt=0`.
- Membrane and synaptic decay use Euler-style `dt * d_state`, not exact exponential decay.
- If `delta_t=None`, `_normalize_delta_ts` defaults to ones, which becomes a fixed-step fallback.

Delta_t audit conclusion:

- This is not a blocker for continuous-time SNN-ODE claims if the experiment always supplies true `delta_t`.
- It is a blocker for training only if the intended data pipeline cannot guarantee `delta_t [B,T]` and cannot prevent padding updates from being interpreted as valid hidden dynamics.

## Mask Controls Padding Audit

Status:

```text
PARTIAL / NEEDS MINIMAL FIX
```

State update:

- Current code updates `s`, `m_local_pre`, attention, `m_reset`, and `r_prev` at every `t` before masking.
- It does not implement:

```python
state = mask_t * proposed_state + (1 - mask_t) * previous_state
```

- For tail padding this is less harmful because no future valid time step follows padding, but it does not satisfy the strict requirement.

Hidden output:

- `h_t` is multiplied by `mask_gate`.
- `mem_seq`, `mem_reset_seq`, and `rate_seq` are not explicitly zeroed at padding.
- `z_mean` uses `valid_time * delta_ts` and therefore excludes padded time steps.
- Distance methods can ignore padding if cache mask is correct.

Attention:

- Attention is causal by construction: at time `t`, it attends only over `mem_hist[:, :t+1]`.
- It does not use `key_padding_mask`.
- For tail-only padding, padded steps occur after all valid points for that sample, so valid hidden states do not look into future padding.
- For online evaluation, prefix forward is still required as the main result to avoid any ambiguity.

Pooling:

- `z_mean` is mask-aware and `delta_t`-weighted.
- Some baseline copied models use unmasked `mean(dim=1)`, but the main SNNODEATT `z_mean` is mask-aware.

Loss:

- `EnhancedSNNLoss` reconstruction and prediction losses use masks.
- Stability loss uses sample-time mask derived from `mask.mean(dim=-1)` if 3D, or `mask.unsqueeze(-1)` if 2D.
- `compute_anomaly_scores` in `trainer.py` multiplies by mask but then takes `mean(dim=(1,2))`, which can be length-biased and should not be the new benchmark path.

Normalization:

- SNNODEATT uses `LayerNorm`, not `BatchNorm`.
- LayerNorm is per token and lower risk, but padded tokens should still be masked downstream.

Mask audit conclusion:

- Current implementation is usable for legacy tail-padded offline runs.
- It is not yet clean enough for the proposed sample-level `mask [B,T]` multidataset mainline.
- Minimal fix should be a compatibility patch in SNNODEATT forward to support `[B,T]` masks and freeze states on padding, plus hidden-cache extraction that zeroes or ignores padding consistently.

## Hidden Trajectory Output Audit

Hidden outputs are already present:

| output | shape | mainline use |
|---|---|---|
| `mem_seq` | `[B,T,H]` | membrane / pre-reset trajectory |
| `mem_reset_seq` | `[B,T,H]` | recommended default hidden trajectory |
| `rate_seq` | `[B,T,H]` | soft spike/rate event representation |
| `h_seq` | `[B,T,2H]` | concatenated `[mem_reset, rate]` candidate |
| `z_mean` | `[B,H]` | pooled latent baseline only |

Current forward does not return `att_hidden` separately. The attention-adjusted membrane `m_pre` is stored in `mem_seq`; this can be documented as the attention-aware membrane trajectory.

Minimal extraction mapping:

| feature_mode | source tensor |
|---|---|
| `membrane` | `mem_seq` |
| `post_reset` | `mem_reset_seq` |
| `spike` | `rate_seq` |
| `membrane_spike` | `h_seq` |
| `att_hidden` | `mem_seq` unless a future non-invasive hook exposes post-attention context |
| `pooled_latent` | `z_mean`, used only for comparison |

## Existing Distance / Score Method Audit

Primary registry:

- `src/snnodeatt/scoring/score_registry.py`
- `SCORE_REGISTRY`

The current registered methods are designed around hidden caches, not raw model outputs. They expect a cache with at least:

```python
{
    "m_reset": Tensor[N,T,H],
    "mask": Tensor[N,T],
    "delta_t": Tensor[N,T],
    "tau": Tensor[N,T],
    "label": Tensor[N],
    "split": List[str],
    "sample_id": List[str],
}
```

| method_name | source_file | function / class | current_input | mask_support | delta_t_support | prefix_support | wrapper_needed | risk_level |
|---|---|---|---|---|---|---|---|---|
| `euclidean` | `src/snnodeatt/scoring/functional_scores.py` | `euclidean_latent_score` | cache `[N,T,H]`, internally pooled to `[N,H]` | YES | YES via `time_weights` | NO direct | YES, for feature_mode/prefix cache | LOW |
| `sobolev_h1` | `src/snnodeatt/scoring/functional_scores.py` | `sobolev_h1_score` | cache `[N,T,H]` | YES | YES in finite difference / weights | NO direct | YES | MEDIUM; assumes comparable time grid length |
| `sobolev_hminus1` | `src/snnodeatt/scoring/functional_scores.py` | `negative_sobolev_hminus1_score` | cache `[N,T,H]` | PARTIAL | NO direct in FFT denominator | NO direct | YES | MEDIUM; spectral approximation on padded grid |
| `trajectory_mahalanobis` | `src/snnodeatt/scoring/functional_scores.py` | `covariance_aware_trajectory_energy` | cache `[N,T,H]` | YES | YES via weights | NO direct | YES | LOW |
| `mmd_cov_kernel` | `src/snnodeatt/scoring/kernel_scores.py` | `mmd_covariance_kernel_score` | empirical hidden/time points | YES | YES via weights and `tau` | NO direct | YES | MEDIUM; prototype sampling / bandwidth sensitive |
| `energy_distance` | `src/snnodeatt/scoring/kernel_scores.py` | `energy_distance_score` | empirical hidden points | YES | YES via weights | NO direct | YES | MEDIUM |
| `trace_distance` | `src/snnodeatt/scoring/quantum_scores.py` | `trace_distance_score` | density matrix from `[T,H]` | YES | YES via weights | NO direct | YES | MEDIUM; projection/eigendecomp cost |
| `qre` | `src/snnodeatt/scoring/quantum_scores.py` | `quantum_relative_entropy_score` | density matrix | YES | YES via weights | NO direct | YES | MEDIUM |
| `von_neumann_entropy` | `src/snnodeatt/scoring/quantum_scores.py` | `von_neumann_entropy_score` | density matrix entropy z-score | YES | YES via weights | NO direct | YES | MEDIUM |
| `purity` | `src/snnodeatt/scoring/quantum_scores.py` | `purity_score` | density matrix purity z-score | YES | YES via weights | NO direct | YES | MEDIUM |
| `bures` | `src/snnodeatt/scoring/quantum_scores.py` | `bures_distance_score` | density matrix | YES | YES via weights | NO direct | YES | MEDIUM/HIGH for numerical cost |
| `hamiltonian_energy` | `src/snnodeatt/scoring/trajectory_energy.py` | `hamiltonian_energy_score` | cache `[N,T,H]` | YES | YES via weights | NO direct | YES | LOW |
| `ode_residual_energy` | `src/snnodeatt/scoring/trajectory_energy.py` | `ode_residual_energy_score` | finite difference over cache `[N,T,H]` | YES | YES | NO direct | YES | MEDIUM; currently residual to mean derivative, not trained ODEFunc |
| `mahalanobis` | `src/snnodeatt/scoring/mahalanobis.py` | `MahalanobisScorer` | latent `[N,H]` only | NO | NO | NO direct | YES | LOW as pooled baseline |
| `sinkhorn` | `src/snnodeatt/scoring/sinkhorn.py` | `sinkhorn_divergence` | point weights and point clouds | caller must mask | caller must encode time | NO direct | YES | MEDIUM |
| `rich_global_local` | `src/snnodeatt/scoring/rich_global_local.py` | `view_signature`, `topk_local_indices` | signatures from mask/delta_t | YES | YES | NO direct | YES | LOW as local-reference helper |
| `gaussian_w2` | `src/snnodeatt/scoring/gaussian_w2.py` | `gaussian_w2_mean_only` | arrays `[N,H]` or `[H]` | NO | NO | NO direct | YES | LOW but not registry-integrated |
| `sliced_wasserstein` | `src/snnodeatt/scoring/sliced_wasserstein.py` | `sliced_wasserstein_1d` | point clouds | caller must mask | NO | NO direct | YES | MEDIUM |

Important no-leak issue:

- `score_benchmark.run_score_benchmark` currently uses `reference_eval_masks` with `reference_splits=("train_normal","val_normal")`.
- It computes both reference scores and threshold from that combined reference mask.
- The requested protocol is stricter: fit reference on train normal, choose threshold on val normal, evaluate test only.

Minimal wrapper requirement:

- Add a wrapper that converts model outputs into the existing cache schema.
- Add split-specific masks:

```python
fit_reference_mask = split == "train_normal"
threshold_mask = split == "val_normal"
eval_mask = split in {"test_normal", "test_abnormal"}
```

- Existing distance math can remain unchanged.

## Online Prefix Leakage Risk Audit

Current SNNODEATT attention is causal in implementation because `_causal_attention` only attends over `mem_hist` up to the current time. That reduces but does not eliminate the need for prefix forward in the online benchmark.

Main reasons prefix forward is still required:

- It makes the online protocol independent of subtle attention or pooling behavior.
- It prevents any future code change from creating hidden leakage.
- It aligns inference with the planned training/prefix augmentation distribution.
- It allows each prefix ratio to have its own train/val normal score distribution.

Correct online benchmark design:

```text
for r in {0.60, 0.61, ..., 1.00}:
    for each sample:
        K_i = sum(mask_i)
        L_i = ceil(r * K_i)
        x_prefix = first L_i valid observations
        mask_prefix = first L_i valid mask values
        delta_t_prefix = first L_i valid delta_t values
        time_prefix = first L_i valid time values
        forward SNNODEATT on prefix
        score prefix hidden trajectory with existing distance wrapper

    fit reference using train_normal prefix scores/features only
    tau_r = quantile(val_normal prefix scores, 0.95)
    evaluate test_normal and test_abnormal prefix scores against tau_r
```

Reporting:

- Full offline metrics: Accuracy, Precision, Recall, F1, F2, AUC, PR-AUC, TP, FP, TN, FN.
- Prefix metrics per ratio.
- First alarm ratio when no anomaly onset is available.
- Do not claim detection delay unless true onset labels exist.

## Minimal Adaptation Plan

Recommended new isolated directory:

```text
experiments/multidata_snnodeatt_distance_benchmark/
```

Recommended files:

| file | purpose | behavior change | affects existing training | rollback |
|---|---|---|---|---|
| `experiments/multidata_snnodeatt_distance_benchmark/audit_repo_dataflow.py` | automated audit probe for configs, imports, shapes | read-only diagnostics | NO | delete file |
| `experiments/multidata_snnodeatt_distance_benchmark/extract_hidden_cache.py` | load processed BTD data, run model forward, write hidden cache | new experiment-only extraction | NO | delete file/output |
| `experiments/multidata_snnodeatt_distance_benchmark/run_offline_distance_benchmark.py` | call existing score registry on full hidden cache | wrapper only, no new math | NO | delete file/output |
| `experiments/multidata_snnodeatt_distance_benchmark/run_prefix_distance_benchmark.py` | prefix forward and per-ratio scoring | wrapper only, no new math | NO | delete file/output |
| `experiments/multidata_snnodeatt_distance_benchmark/summarize_results.py` | summarize offline/prefix metrics | reporting only | NO | delete file/output |
| `src/snnodeatt/data/btd_cache_loader.py` | canonical loader for processed `.pt` payload | enables documented payload | NO if only used by new experiments | delete file |
| `src/snnodeatt/models/forward_adapter.py` or experiment-local adapter | convert tuple outputs to dict without changing model math | new wrapper only | NO | delete file |
| optional small patch to `src/snnodeatt/models/snn_odeatt.py` | support `mask [B,T]` by unsqueezing `mask_gate`; optionally freeze state on padding | minimal forward compatibility | YES, but mathematically same for valid points if done carefully | revert patch |

Do not modify:

- ATT internals.
- ODEFunc math.
- SNN reset/spike logic.
- Existing score definitions.

## Blockers Before Training

Blocking items:

1. `src/snnodeatt/data/wafer_loader.py` cannot import because `utils.poisson_sampling_series_segmented` is missing from the package.
2. `src/snnodeatt/train/trainer.py` cannot import because it uses old top-level imports and missing `model_factory`.
3. SNNODEATT forward fails with the target sample-level `mask [B,T]`.
4. Canonical multidataset loaders for HAI, SWaT, WADI, TEP, ST-AWFD, and Bosch do not exist yet; prepare scripts are scaffolds.
5. `time/tau [B,T]` is not produced by the current dataloader.
6. Current `score_benchmark` does not separate train normal reference fitting from val normal threshold selection.
7. Current wafer loader standardizes per sample using the full sample rather than train-fitted scaler statistics.
8. Padding states are not strictly frozen inside SNNODEATT; only downstream output/pooling/loss masking reduces impact.
9. The Poisson sampler cannot be audited inside this repo because the implementation is not present.

Non-blocking but important risks:

- SWaT config uses `data/SWAT/...` while the repo directory is `data/SWaT/...`.
- Bosch is staged BTD, not native continuous time; later reports must preserve this wording.
- Several copied baseline files are provenance archives and not yet clean train/eval entrypoints.

## Go / No-Go Decision

```text
GO_TO_TRAINING = NO
```

Reason:

- Dataset paths are documented, but executable loaders are incomplete.
- SNNODEATT forward is clear and returns `[B,T,H]` trajectories, but it fails for target `mask [B,T]`.
- `delta_t` enters SNN-ODE dynamics, but padding `delta_t=0` is clamped and state update is not frozen.
- Masking is partially correct for legacy `[B,T,D]` paths but not the target sample-level path.
- Existing distance methods are usable through wrappers, but strict train/val/test no-leak thresholding needs an adapter.
- Offline and prefix benchmark design is feasible, but implementation prerequisites are missing.

Training should start only after a minimal Phase 1 adaptation verifies:

- importable package-relative dataloader/trainer or experiment-local runner;
- processed BTD loader emitting `x [B,T,D]`, `mask [B,T]`, `delta_t [B,T]`, `time/tau [B,T]`, `label [B]`, `split`, and `sample_id`;
- SNNODEATT forward smoke passes with `mask [B,T]`;
- hidden cache extraction writes `hidden [N,T,H]` plus mask/time metadata;
- distance wrappers use train normal for reference, val normal for threshold, and test only for evaluation;
- prefix forward smoke passes for at least a toy batch.

## Modified Files in Phase 0

Modified/new file:

- `docs/snnodeatt_multidata_distance_audit.md`

Reason:

- Required Phase 0 audit report.

Behavior change:

- None. Documentation only.

Tests/checks run:

- Repository file and grep audits.
- Python import smoke tests.
- Synthetic SNNODEATT forward smoke with `[B,T]` and `[B,T,D]` masks.
- Toy score benchmark smoke for existing registry methods.

Rollback note:

- Delete `docs/snnodeatt_multidata_distance_audit.md` to remove this Phase 0 report.
