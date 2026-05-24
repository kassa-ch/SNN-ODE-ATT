# SNN_ODEATT Training Audit

## Shape Contract

The current data contract is:

1. `preprocess_data/*.csv` stores one sample as a 2D matrix `[T,D]`.
2. `TimeSeriesDataset.__getitem__` reads one CSV, standardizes it per sample, optionally applies Poisson sampling, adds 2 positional-encoding dimensions, and returns `(x, mask, delta_t, path)`.
3. `custom_collate` pads variable-length samples and stacks the batch into `[B,T,D+2]`.
4. `SNN_ODEATT.forward(x, mask, delta_t)` consumes that `[B,T,D+2]` tensor.

## Training Expansion

The `Scrips/main.py` CLI accepts:

- `--exp`
- `--models`
- `--data_dir`
- `--force_retrain`
- `--no_cv`
- `--n_splits`

For `PredictiveSNN_ODEATT_Model`, the flow is:

1. `main.py` calls `run_single_experiment`.
2. `run_single_experiment` resolves experiment settings from `config.EXPERIMENTS`.
3. `train_all_models_kfold` loops selected models.
4. `train_model_kfold` finds normal samples via `find_csv_paths(data_dir, normal=True)`.
5. `split_data_kfold_corrected` builds K-fold train/validation/test splits from normal samples.
6. `train_single_kfold_run` builds loaders, infers `input_dim` from a batch, creates `PredictiveSNN_ODEATT_Model`, and trains/evaluates each fold.
7. `model_forward` normalizes the model output to `(recon, pred, z_mean, mem_seq, mem_reset_seq, spike_seq, rate_seq, h_seq)`.
8. `EnhancedSNNLoss` computes reconstruction loss, prediction loss, and stability loss from `mem_seq`.
9. Checkpoints are written under `MODEL_SAVE_DIR`, currently `./saved_models`.

## SNN_ODEATT Forward Path

`ContinuousAttentionSNNODE` expands the sequence over time:

1. Normalize `delta_t` to `[B,T]`.
2. For each timestep:
   - update continuous SNN/ODE cell state;
   - append local membrane state to history;
   - compute causal continuous attention over the observed history;
   - derive rate/reset state;
   - emit reconstruction and one-step prediction.
3. Stack timestep outputs back into `[B,T,*]`.
4. Return the 7-output contract:
   `(recons, preds, z_mean, mem_seq, mem_reset_seq, rate_seq, h_seq)`.

## Current Path Notes

- Default `DATA_DIR` is still `./origin_samples/preprocess_data`.
- For the local data generated from `Data/<dataset>/Origin_Datasets`, pass `--data_dir Data/<dataset>/preprocess_data`.
- `find_csv_paths(..., normal=True)` only uses files ending in `_normal.csv` for training.
- Abnormal files are useful for later detection/evaluation, but the current training loop trains from normal samples only.
