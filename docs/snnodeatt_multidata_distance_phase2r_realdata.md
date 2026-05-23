# SNNODEATT Multidataset Distance Scoring Phase 2R Real Data

## 1. Executive Summary

- selected_real_dataset: `NO_REAL_DATA_FOUND`
- dataset_type: `none`
- GO_TO_FULL_SINGLE_DATASET: `NO`
- GO_TO_MULTI_DATASET: `NO`

Phase 2R first audits real data candidates. It only converts and trains when a real file has features, labels, and a safe split strategy. Synthetic fallback is not accepted as the final Phase 2R dataset.

## 2. Phase 2 Synthetic Fallback Recap

Phase 2 validated the pipeline on `synthetic_btd`, but kept `GO_TO_FULL_SINGLE_DATASET=NO` because no real processed BTD payload was present.

## 3. Real Data Discovery Audit

```json
[
  {
    "candidate_name": "wafer_exp1",
    "file_or_dir_path": "configs/experiments/wafer_exp1.yaml",
    "file_type": "yaml",
    "size": 316,
    "inferred_dataset": "wafer",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "wafer_exp2",
    "file_or_dir_path": "configs/experiments/wafer_exp2.yaml",
    "file_type": "yaml",
    "size": 315,
    "inferred_dataset": "wafer",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "wafer_exp3",
    "file_or_dir_path": "configs/experiments/wafer_exp3.yaml",
    "file_type": "yaml",
    "size": 314,
    "inferred_dataset": "wafer",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "wafer_exp4",
    "file_or_dir_path": "configs/experiments/wafer_exp4.yaml",
    "file_type": "yaml",
    "size": 334,
    "inferred_dataset": "wafer",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "tep",
    "file_or_dir_path": "configs/experiments/tep.yaml",
    "file_type": "yaml",
    "size": 322,
    "inferred_dataset": "TEP",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "hai",
    "file_or_dir_path": "configs/experiments/hai.yaml",
    "file_type": "yaml",
    "size": 322,
    "inferred_dataset": "HAI",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "swat",
    "file_or_dir_path": "configs/experiments/swat.yaml",
    "file_type": "yaml",
    "size": 335,
    "inferred_dataset": "SWaT",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "wadi",
    "file_or_dir_path": "configs/experiments/wadi.yaml",
    "file_type": "yaml",
    "size": 335,
    "inferred_dataset": "WADI",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "st_awfd",
    "file_or_dir_path": "configs/experiments/st_awfd.yaml",
    "file_type": "yaml",
    "size": 697,
    "inferred_dataset": "ST_AWFD",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "base",
    "file_or_dir_path": "configs/base.yaml",
    "file_type": "yaml",
    "size": 142,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "bosch_production_line",
    "file_or_dir_path": "configs/experiments/bosch_production_line.yaml",
    "file_type": "yaml",
    "size": 867,
    "inferred_dataset": "BoschProductionLine",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "snn_odeatt",
    "file_or_dir_path": "configs/model/snn_odeatt.yaml",
    "file_type": "yaml",
    "size": 192,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "anomaly_score_benchmark",
    "file_or_dir_path": "configs/scoring/anomaly_score_benchmark.yaml",
    "file_type": "yaml",
    "size": 335,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "gaussian_w2",
    "file_or_dir_path": "configs/scoring/gaussian_w2.yaml",
    "file_type": "yaml",
    "size": 50,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "mahalanobis",
    "file_or_dir_path": "configs/scoring/mahalanobis.yaml",
    "file_type": "yaml",
    "size": 68,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "sinkhorn",
    "file_or_dir_path": "configs/scoring/sinkhorn.yaml",
    "file_type": "yaml",
    "size": 139,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "sliced_wasserstein",
    "file_or_dir_path": "configs/scoring/sliced_wasserstein.yaml",
    "file_type": "yaml",
    "size": 58,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "server_baseline_model_file_inventory",
    "file_or_dir_path": "docs/model_inventory/server_baseline_model_file_inventory.json",
    "file_type": "json",
    "size": 12275,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "list_len=11",
    "usable_for_phase2r": "NO",
    "reason": "no feature-like x/data/features evidence"
  },
  {
    "candidate_name": "environment",
    "file_or_dir_path": "environment.yml",
    "file_type": "yml",
    "size": 221,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "no feature-like x/data/features evidence"
  },
  {
    "candidate_name": "benchmark_config",
    "file_or_dir_path": "experiments/anomaly_score_benchmark/benchmark_config.yaml",
    "file_type": "yaml",
    "size": 511,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "config_text",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "score_diagnostic_exp4",
    "file_or_dir_path": "experiments/anomaly_score_benchmark/score_diagnostic_exp4.csv",
    "file_type": "csv",
    "size": 16346,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "header_cols=27, preview_rows=40",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "score_diagnostic_exp4_top_summary",
    "file_or_dir_path": "experiments/anomaly_score_benchmark/score_diagnostic_exp4_top_summary.csv",
    "file_type": "csv",
    "size": 1419,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "header_cols=15, preview_rows=7",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "score_method_summary_exp4",
    "file_or_dir_path": "experiments/anomaly_score_benchmark/score_method_summary_exp4.csv",
    "file_type": "csv",
    "size": 1554,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "header_cols=13, preview_rows=13",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  },
  {
    "candidate_name": "phase1_smoke_summary",
    "file_or_dir_path": "experiments/multidata_snnodeatt_distance_benchmark/phase1_smoke_summary.json",
    "file_type": "json",
    "size": 5188,
    "inferred_dataset": "unknown",
    "has_x_or_features": "NO",
    "has_label": "NO",
    "has_split": "NO",
    "has_time_or_delta_t": "NO",
    "shape_if_readable": "dict_keys=7",
    "usable_for_phase2r": "NO",
    "reason": "result/config/diagnostic file, not a trainable data payload"
  }
]
```

## 4. Selected Real Dataset

```json
{
  "selected_dataset": "NO_REAL_DATA_FOUND",
  "dataset_type": "none",
  "reason_for_selection": "No repository file contains enough feature + label evidence for real Phase 2R BTD conversion."
}
```

## 5. Original Data Shape and BTD Conversion Strategy

```json
{}
```

## 6. BTD Payload Shape Check

```json
{}
```

## 7. Standardization / Split / Label No-leak Notes

- Scaler fitting is allowed only on train split when conversion runs.
- Train-normal is used for reference, val-normal for q0.95 threshold, and test only for evaluation.
- If no real payload is found, no scaler is fit and no training is launched.

## 8. Real Single-dataset Smoke Training Config

```json
{
  "seed": 42,
  "epochs": 3,
  "batch_size": 4,
  "learning_rate": 0.001,
  "hidden_dim": 16,
  "threshold": 0.3,
  "tau_mem": 10.0,
  "tau_syn_base": 8.0,
  "max_batches_per_epoch": 5,
  "num_workers": 0,
  "distance_methods": [
    "euclidean",
    "trajectory_mahalanobis",
    "sobolev_h1",
    "energy_distance"
  ],
  "feature_modes": [
    "mem_seq",
    "mem_reset_seq",
    "rate_seq",
    "h_seq"
  ],
  "prefix_ratios": [
    0.6,
    0.8,
    1.0
  ],
  "threshold_quantile": 0.95
}
```

## 9. Training Loss Summary

```json
{}
```

## 10. Checkpoint Save/Reload Result

```json
{}
```

## 11. Hidden Cache Shape

```json
{}
```

## 12. Offline Benchmark Summary

```csv
```

## 13. Prefix Benchmark Summary

```csv
```

## 14. NaN / Inf / OOM / Traceback Check

```json
{
  "loss_nan_or_inf": false,
  "grad_nan_or_inf": false,
  "hidden_nan": false,
  "delta_t_negative": false,
  "oom": false,
  "traceback": false
}
```

## 15. Blockers

- NO_REAL_DATA_FOUND: repository contains data scaffolds and small result/config files, but no trainable real x+label payload.

## 16. Decision

```text
GO_TO_FULL_SINGLE_DATASET = NO
GO_TO_MULTI_DATASET = NO
```
