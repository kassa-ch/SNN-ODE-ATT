# Public Dataset Training Next Steps

## Current State

HAI and ST-AWFD now have public `preprocess_data` directories that match the
current loader contract:

- `Data/HAI/preprocess_data`
- `Data/ST-AWFD/preprocess_data`

Each CSV is one `[T,D]` sample. `TimeSeriesDataset` appends two positional
encoding dimensions and `custom_collate` batches samples into `[B,T,D+2]`.

TEP, SWaT, WADI, and Bosch are not yet in the training chain:

- TEP has no usable files under `Data/TEP/Origin_Datasets` for this phase.
- SWaT and WADI currently have no `Origin_Datasets` directory.
- Bosch is not present locally.

## Immediate Plan

1. Use HAI and ST-AWFD first.
2. Run SNNODEATT check-only validation on uniform and nonuniform views.
3. Run only tiny smoke training when needed: 1 epoch, at most 2 batches.
4. Use smoke checkpoints only to validate Distance scoring plumbing.
5. Search for the best distance score using public benchmark splits.
6. After fixing the best distance score, compare SNNODEATT vs SNNODE vs SNN.

## Leakage Control

Public benchmark detection must disable test-normal filtering:

`remove_test_extreme_k = 0`

Train and validation splits must contain normal data only. Abnormal data is used
only in the test split. This avoids using test labels or abnormal membership to
clean the normal test set.

## GitHub Policy

`Data/**`, checkpoints, plots, score outputs, and large benchmark outputs stay
out of GitHub. Only lightweight manifests under `Results/manifests/` and source
configuration/scripts should be mirrored.
