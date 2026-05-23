# SNNODEATT Multidataset Distance Scoring Phase 1

## 1. Executive Summary

Phase 1 completed the minimal adaptation required after the Phase 0 audit. It
did not launch formal training, did not introduce a new model structure, did
not split the existing ATT block, and did not add a new distance method.

Main result:

```text
GO_TO_TRAINING = YES
```

Meaning of this YES:

- The repository is ready for **single-dataset smoke training / smoke hidden-cache extraction**.
- It is not yet a green light for full multidataset training, because real HAI,
  SWaT, WADI, TEP, ST-AWFD, and Bosch converters still need dataset-specific
  implementation.

What now works:

- `mask [B,T]`, `mask [B,T,1]`, and legacy `mask [B,T,D]` are normalized into
  canonical sample-level `mask [B,T]`.
- Legacy `mask [B,T,D]` is accepted only when all channels share the same
  validity at each time step.
- Channel-wise asynchronous/value-level masks are rejected, matching the current
  assumption that all sensors share the same sampled time grid.
- SNNODEATT forward now supports target `x [B,T,D]`, `mask [B,T]`,
  `delta_t [B,T]`.
- Padding hidden outputs are zeroed and padding time steps freeze the recurrent
  state instead of updating it.
- Hidden cache extraction saves per-time trajectories, not only final hidden.
- Existing score methods are called through a no-leak wrapper:
  `train_normal` builds the reference, `val_normal` sets the threshold,
  and `test` is only evaluated.
- Prefix forward smoke works for `r=0.60` and `r=1.00` by re-forwarding prefix
  batches, not slicing full-sequence hidden states.

## 2. Phase 0 Blockers Recap

Phase 0 found these blockers:

1. SNNODEATT worked with legacy `mask [B,T,D]` but failed with target
   `mask [B,T]`.
2. The wafer dataloader imported missing top-level
   `utils.poisson_sampling_series_segmented`.
3. `trainer.py` relied on old top-level modules:
   `data_loader`, `config`, `enhanced_loss`, `model_factory`.
4. External dataset entries were mostly scaffolds.
5. Existing distance methods lacked a strict no-leak train/val/test wrapper.

Phase 1 addressed blockers 1 and 5 directly, and added an independent synthetic
BTD loader plus hidden-cache extractor so we do not need to repair the old
trainer before smoke validation.

## 3. Modified Files

| file | purpose | behavior change | affects old training? | rollback |
|---|---|---|---|---|
| `src/snnodeatt/utils/mask.py` | canonical mask normalization | new helper only | no direct effect | delete file and imports |
| `src/snnodeatt/models/snn_odeatt.py` | support `mask [B,T]`, freeze padding state | forward accepts `[B,T]`, `[B,T,1]`, legacy `[B,T,D]`; padding hidden is zero | yes, but only mask compatibility and padding handling | revert file |
| `src/snnodeatt/losses/enhanced_loss.py` | support sample-level mask in reconstruction/prediction loss | `[B,T]` masks now broadcast safely to `[B,T,D]` loss tensors | yes, but only mask shape compatibility | revert file |
| `experiments/multidata_snnodeatt_distance_benchmark/mask_utils.py` | experiment-local mask utility exports | no math change | no | delete experiment dir |
| `experiments/multidata_snnodeatt_distance_benchmark/btd_processed_loader.py` | unified BTD processed/synthetic loader | new loader for smoke and future BTD payloads | no | delete file |
| `experiments/multidata_snnodeatt_distance_benchmark/hidden_cache_extractor.py` | extract per-time SNNODEATT trajectories | new cache extractor | no | delete file |
| `experiments/multidata_snnodeatt_distance_benchmark/distance_wrapper.py` | no-leak wrapper for existing score registry | no new distance math; split-safe scoring | no | delete file |
| `experiments/multidata_snnodeatt_distance_benchmark/smoke_phase1.py` | synthetic Phase 1 validation | smoke only, no training | no | delete file |
| `experiments/multidata_snnodeatt_distance_benchmark/README.md` | documents experiment scope | documentation only | no | delete file |
| `experiments/multidata_snnodeatt_distance_benchmark/phase1_smoke_summary.json` | smoke output artifact | small reproducibility record | no | delete file |
| `docs/snnodeatt_multidata_distance_phase1.md` | Phase 1 report | documentation only | no | delete file |

## 4. Mask Normalizer Design and Test Results

Implementation:

- `src/snnodeatt/utils/mask.py`
- `normalize_sequence_mask(mask, batch_size=None, seq_len=None, strict=True)`

Rules:

- `mask [B,T]` returns unchanged as canonical sample-level mask.
- `mask [B,T,1]` squeezes to `[B,T]`.
- `mask [B,T,D]` is accepted only when all feature channels are equal at each
  time step, then compressed to `mask[:,:,0]`.
- Non-uniform value-level masks raise `ValueError`.

Smoke result:

```text
mask[B,T] -> PASS
mask[B,T,1] -> PASS
legacy mask[B,T,D] -> PASS
invalid channel-wise mask[B,T,D] -> PASS_RAISED
```

This preserves the current project assumption that all sensors in one sample
share the same non-homogeneous Poisson sampling times.

## 5. SNNODEATT `mask [B,T]` Forward Fix

Changed file:

- `src/snnodeatt/models/snn_odeatt.py`

Key behavior:

- `forward(x, mask, delta_t)` converts mask to `mask_bt [B,T]` once at the
  beginning.
- At each time step:

```python
mask_t = mask_bt[:, t]
mask_h = mask_t.view(B, 1)
s = mask_h * s_prop + (1 - mask_h) * s_prev
m_reset = mask_h * m_reset_prop + (1 - mask_h) * m_reset_prev
r_prev = mask_h * rate_prop + (1 - mask_h) * r_prev_old
```

- Output hidden tensors are masked so padding hidden is zero.
- ATT remains the original causal time-weighted attention. It was not split or
  mathematically changed.

Synthetic forward smoke:

| mask form | status | mem_seq | mem_reset_seq | rate_seq | h_seq | padding hidden zero |
|---|---|---:|---:|---:|---:|---|
| `mask [B,T]` | PASS | `[8,14,12]` | `[8,14,12]` | `[8,14,12]` | `[8,14,24]` | true |
| `mask [B,T,1]` | PASS | `[8,14,12]` | `[8,14,12]` | `[8,14,12]` | `[8,14,24]` | true |
| legacy `mask [B,T,D]` | PASS | `[8,14,12]` | `[8,14,12]` | `[8,14,12]` | `[8,14,24]` | true |

Loss smoke:

```text
EnhancedSNNLoss with mask[B,T]: PASS
loss finite: true
loss keys: pred_loss, recon_loss, stability_loss, total_loss, weighted_stability
```

## 6. BTD Processed Loader Status

New file:

- `experiments/multidata_snnodeatt_distance_benchmark/btd_processed_loader.py`

Supported:

- `.pt` processed payloads.
- `.npz` processed payloads.
- synthetic fallback dataset.
- compact observed sequences padded only at batch tail.

Canonical batch:

```text
x        [B,T,D]
mask     [B,T]
delta_t  [B,T]
time     [B,T]
label    [B]
sample_id: list
split: list
```

Real dataset status:

- No committed real processed data exists, which is expected.
- HAI/SWaT/WADI/TEP/ST-AWFD/Bosch still require dataset-specific converters.
- Synthetic fallback is now executable and package-safe.

## 7. Mini-batch Shape Check

Synthetic BTD mini-batch:

```text
dataset_name = synthetic_btd
split = mixed
x.shape = [8, 14, 5]
mask.shape = [8, 14]
delta_t.shape = [8, 14]
time.shape = [8, 14]
label.shape = [8]
valid_len_min = 7
valid_len_max = 14
padding_tail_check = PASS
delta_t_nonnegative = PASS
mask_unique_values = [0.0, 1.0]
```

## 8. Hidden Cache Extractor Smoke

New file:

- `experiments/multidata_snnodeatt_distance_benchmark/hidden_cache_extractor.py`

Cache keys:

```text
delta_t
h_seq
label
mask
mem_reset_seq
mem_seq
pooled_latent
rate_seq
sample_id
split
time
```

Tensor shapes:

```text
mem_seq        [24, 14, 12]
mem_reset_seq  [24, 14, 12]
rate_seq       [24, 14, 12]
h_seq          [24, 14, 24]
mask           [24, 14]
delta_t        [24, 14]
time           [24, 14]
pooled_latent  [24, 12]
label          [24]
```

Smoke checks:

```text
hidden_dtype = torch.float32
valid_len_min = 7
valid_len_max = 14
hidden_contains_nan = false
delta_t_contains_negative = false
```

## 9. Distance Wrapper No-leak Logic and Toy Smoke

New file:

- `experiments/multidata_snnodeatt_distance_benchmark/distance_wrapper.py`

No-leak policy:

- Reference fitting uses only `train_normal`.
- Threshold calibration uses only `val_normal`, default `q=0.95`.
- Evaluation uses only test samples.
- Existing score functions are called through `SCORE_REGISTRY`.
- No new distance mathematics was added.

Toy metrics table:

| method | feature | threshold source | reference source | TP | FP | TN | FN | Accuracy | Precision | Recall | F1 | F2 | AUC | PR-AUC |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| euclidean | mem_reset_seq | val_normal | train_normal | 0 | 0 | 6 | 6 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5556 | 0.5545 |
| trajectory_mahalanobis | mem_reset_seq | val_normal | train_normal | 5 | 4 | 2 | 1 | 0.5833 | 0.5556 | 0.8333 | 0.6667 | 0.7576 | 0.6389 | 0.6403 |
| sobolev_h1 | mem_reset_seq | val_normal | train_normal | 2 | 2 | 4 | 4 | 0.5000 | 0.5000 | 0.3333 | 0.4000 | 0.3571 | 0.6111 | 0.6724 |

These are synthetic smoke metrics only and should not be interpreted as real
model performance.

## 10. Prefix Forward Smoke

Prefix rule:

```text
prefix_len_i = ceil(r * valid_len_i)
```

The smoke creates prefix batches and re-runs SNNODEATT. It does not slice hidden
states from a full-sequence forward pass.

Results:

| ratio | status | re-forwarded | prefix hidden shape | prefix mask shape |
|---:|---|---|---:|---:|
| 0.60 | PASS | true | `[8,9,12]` | `[8,9]` |
| 1.00 | PASS | true | `[8,14,12]` | `[8,14]` |

Formal prefix benchmark should later run `r = 0.60, 0.61, ..., 1.00`, and each
ratio should have its own train-normal reference and val-normal threshold.

## 11. Still-existing Blockers

Remaining blockers before **full multidataset formal training**:

1. Real dataset converters for HAI, SWaT, WADI, TEP, ST-AWFD, and Bosch are
   still scaffolds.
2. The legacy wafer dataloader and trainer still have package-import issues.
   Phase 1 intentionally avoids fixing them by using independent experiment
   adapters.
3. No real processed BTD payload has been validated in this repository yet.
4. The next training runner should be implemented in the isolated experiment
   directory first, not by reviving the old trainer.
5. Bosch remains staged BTD, not a natural continuous-time curve; future reports
   must preserve that wording.

No blockers remain for synthetic single-dataset smoke training or for building
the next isolated single-dataset smoke runner.

## 12. Checks Run

```text
python experiments/multidata_snnodeatt_distance_benchmark/smoke_phase1.py
python -m compileall src scripts experiments tests
python import smoke for:
  snnodeatt.models.snn_odeatt
  experiments.multidata_snnodeatt_distance_benchmark.mask_utils
  experiments.multidata_snnodeatt_distance_benchmark.btd_processed_loader
  experiments.multidata_snnodeatt_distance_benchmark.hidden_cache_extractor
  experiments.multidata_snnodeatt_distance_benchmark.distance_wrapper
sensitive information scan
>50MB large-file scan
```

All required Phase 1 smoke checks passed.

## 13. GO / NO-GO

```text
GO_TO_TRAINING = YES
```

Scope:

- YES for next-step **single-dataset smoke training** using the isolated Phase 1
  BTD loader and SNNODEATT forward path.
- NO for immediate full multidataset formal training until real dataset
  converters and processed payloads are implemented and validated.
