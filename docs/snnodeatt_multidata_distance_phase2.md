# SNNODEATT Multidataset Distance Scoring Phase 2

## 1. Executive Summary

Phase 2 ran a small single-dataset smoke training pipeline. No full multidataset training was launched, no new model math was added, and no new distance method was introduced.

- selected_dataset: `synthetic_btd`
- dataset_type: `synthetic_fallback`
- GO_TO_FULL_SINGLE_DATASET: `NO`
- GO_TO_MULTI_DATASET: `NO`

Because no real processed BTD payload was found in the repository, this run used a clearly marked synthetic fallback. Therefore the pipeline is validated, but real-data full single-dataset training is not yet approved.

## 2. Selected Dataset and Reason

```json
{
  "selected_dataset": "synthetic_btd",
  "dataset_type": "synthetic_fallback",
  "raw_path": "N/A",
  "processed_path": "N/A",
  "manifest_path": "N/A",
  "split_source": "SyntheticBTDDataset built-in split labels",
  "label_source": "SyntheticBTDDataset synthetic labels",
  "reason_for_selection": "No real processed BTD payload was found in the repository.",
  "availability": [
    {
      "dataset_name": "wafer",
      "raw_path": "data/wafer/raw",
      "processed_path": "data/wafer/processed",
      "manifest_path": "data/wafer/manifests",
      "processed_payload_count": 0,
      "status": "unavailable",
      "reason": "no .pt/.npz processed BTD payload found"
    },
    {
      "dataset_name": "ST_AWFD",
      "raw_path": "data/ST_AWFD/raw",
      "processed_path": "data/ST_AWFD/processed",
      "manifest_path": "data/ST_AWFD/manifests",
      "processed_payload_count": 0,
      "status": "unavailable",
      "reason": "no .pt/.npz processed BTD payload found"
    },
    {
      "dataset_name": "BoschProductionLine",
      "raw_path": "data/BoschProductionLine/raw",
      "processed_path": "data/BoschProductionLine/processed",
      "manifest_path": "data/BoschProductionLine/manifests",
      "processed_payload_count": 0,
      "status": "unavailable",
      "reason": "no .pt/.npz processed BTD payload found"
    },
    {
      "dataset_name": "HAI",
      "raw_path": "data/HAI/raw",
      "processed_path": "data/HAI/processed",
      "manifest_path": "data/HAI/manifests",
      "processed_payload_count": 0,
      "status": "unavailable",
      "reason": "no .pt/.npz processed BTD payload found"
    },
    {
      "dataset_name": "TEP",
      "raw_path": "data/TEP/raw",
      "processed_path": "data/TEP/processed",
      "manifest_path": "data/TEP/manifests",
      "processed_payload_count": 0,
      "status": "unavailable",
      "reason": "no .pt/.npz processed BTD payload found"
    },
    {
      "dataset_name": "SWaT",
      "raw_path": "data/SWaT/raw",
      "processed_path": "data/SWaT/processed",
      "manifest_path": "data/SWaT/manifests",
      "processed_payload_count": 0,
      "status": "unavailable",
      "reason": "no .pt/.npz processed BTD payload found"
    },
    {
      "dataset_name": "WADI",
      "raw_path": "data/WADI/raw",
      "processed_path": "data/WADI/processed",
      "manifest_path": "data/WADI/manifests",
      "processed_payload_count": 0,
      "status": "unavailable",
      "reason": "no .pt/.npz processed BTD payload found"
    }
  ]
}
```

## 3. Training Config

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

## 4. Mini-batch Shape Check

```json
{
  "dataset_name": "synthetic_btd",
  "split": "mixed",
  "x_shape": [
    4,
    13,
    5
  ],
  "mask_shape": [
    4,
    13
  ],
  "delta_t_shape": [
    4,
    13
  ],
  "time_shape": [
    4,
    13
  ],
  "label_shape": [
    4
  ],
  "valid_len_min": 7,
  "valid_len_max": 13,
  "padding_tail_check": "PASS",
  "delta_t_nonnegative": "PASS",
  "mask_unique_values": [
    0.0,
    1.0
  ]
}
```

## 5. Smoke Training Loss Summary

```json
{
  "epochs": 3,
  "batch_size": 4,
  "max_batches_per_epoch": 5,
  "training_log_path": "experiments/multidata_snnodeatt_distance_benchmark/runs/phase2_single_dataset_smoke/training_log.csv",
  "rows": [
    {
      "epoch": 1,
      "train_loss": 0.4780774414539337,
      "val_loss": 0.33234167098999023,
      "train": {
        "loss": 0.4780774414539337,
        "batches": 2,
        "grad_nan_or_inf": false,
        "hidden_nan": false,
        "delta_t_negative": false,
        "valid_len_min": 7.0,
        "valid_len_max": 14.0,
        "gpu_memory": {
          "allocated": 17167872,
          "reserved": 23068672
        },
        "elapsed_sec": 1.209
      },
      "val": {
        "loss": 0.33234167098999023,
        "batches": 1,
        "grad_nan_or_inf": false,
        "hidden_nan": false,
        "delta_t_negative": false,
        "valid_len_min": 7.0,
        "valid_len_max": 13.0,
        "gpu_memory": {
          "allocated": 17166848,
          "reserved": 23068672
        },
        "elapsed_sec": 0.108
      }
    },
    {
      "epoch": 2,
      "train_loss": 0.30723462998867035,
      "val_loss": 0.21702148020267487,
      "train": {
        "loss": 0.30723462998867035,
        "batches": 2,
        "grad_nan_or_inf": false,
        "hidden_nan": false,
        "delta_t_negative": false,
        "valid_len_min": 7.0,
        "valid_len_max": 14.0,
        "gpu_memory": {
          "allocated": 17164288,
          "reserved": 23068672
        },
        "elapsed_sec": 0.577
      },
      "val": {
        "loss": 0.21702148020267487,
        "batches": 1,
        "grad_nan_or_inf": false,
        "hidden_nan": false,
        "delta_t_negative": false,
        "valid_len_min": 7.0,
        "valid_len_max": 13.0,
        "gpu_memory": {
          "allocated": 17166848,
          "reserved": 23068672
        },
        "elapsed_sec": 0.089
      }
    },
    {
      "epoch": 3,
      "train_loss": 0.21025153249502182,
      "val_loss": 0.15429280698299408,
      "train": {
        "loss": 0.21025153249502182,
        "batches": 2,
        "grad_nan_or_inf": false,
        "hidden_nan": false,
        "delta_t_negative": false,
        "valid_len_min": 7.0,
        "valid_len_max": 14.0,
        "gpu_memory": {
          "allocated": 17168384,
          "reserved": 23068672
        },
        "elapsed_sec": 0.604
      },
      "val": {
        "loss": 0.15429280698299408,
        "batches": 1,
        "grad_nan_or_inf": false,
        "hidden_nan": false,
        "delta_t_negative": false,
        "valid_len_min": 7.0,
        "valid_len_max": 13.0,
        "gpu_memory": {
          "allocated": 17166848,
          "reserved": 23068672
        },
        "elapsed_sec": 0.099
      }
    }
  ]
}
```

## 6. Checkpoint Save/Reload Result

```json
{
  "checkpoint_reload": "PASS",
  "checkpoint_path": "experiments/multidata_snnodeatt_distance_benchmark/runs/phase2_single_dataset_smoke/checkpoints/best.pt",
  "mem_seq_shape": [
    4,
    13,
    16
  ],
  "mem_reset_seq_shape": [
    4,
    13,
    16
  ],
  "rate_seq_shape": [
    4,
    13,
    16
  ],
  "h_seq_shape": [
    4,
    13,
    32
  ],
  "hidden_has_nan": false
}
```

## 7. Hidden Cache Extraction Result

```json
{
  "cache_keys": [
    "delta_t",
    "h_seq",
    "label",
    "mask",
    "mem_reset_seq",
    "mem_seq",
    "pooled_latent",
    "rate_seq",
    "sample_id",
    "split",
    "time"
  ],
  "tensor_shapes": {
    "mem_seq": [
      24,
      14,
      16
    ],
    "mem_reset_seq": [
      24,
      14,
      16
    ],
    "rate_seq": [
      24,
      14,
      16
    ],
    "h_seq": [
      24,
      14,
      32
    ],
    "mask": [
      24,
      14
    ],
    "delta_t": [
      24,
      14
    ],
    "time": [
      24,
      14
    ],
    "pooled_latent": [
      24,
      16
    ],
    "label": [
      24
    ]
  },
  "hidden_dtype": "torch.float32",
  "valid_len_min": 7.0,
  "valid_len_max": 14.0,
  "hidden_contains_nan": false,
  "delta_t_contains_negative": false,
  "cache_path": "experiments/multidata_snnodeatt_distance_benchmark/runs/phase2_single_dataset_smoke/cache/hidden_cache.pt",
  "split_counts": {
    "test_abnormal": 6,
    "test_normal": 6,
    "train_normal": 8,
    "val_normal": 4
  },
  "label_distribution": {
    "tensor(0)": 18,
    "tensor(1)": 6
  }
}
```

## 8. Offline Distance Benchmark Smoke Result

```csv
dataset,feature_mode,status,failure_reason,method,feature_key,threshold_source,reference_source,threshold_quantile,threshold,Accuracy,Precision,Recall,F1,F2,AUC,PR_AUC,TP,FP,TN,FN
selected,mem_seq,PASS,,euclidean,mem_seq,val_normal,train_normal,0.95,0.9910704374313353,0.5,0.0,0.0,0.0,0.0,0.4166666666666667,0.4956709956709957,0,0,6,6
selected,mem_seq,PASS,,trajectory_mahalanobis,mem_seq,val_normal,train_normal,0.95,0.874733880162239,0.4166666666666667,0.0,0.0,0.0,0.0,0.75,0.6746031746031745,0,1,5,6
selected,mem_seq,PASS,,sobolev_h1,mem_seq,val_normal,train_normal,0.95,0.11013222448527812,0.5833333333333334,0.6666666666666666,0.3333333333333333,0.4444444444444444,0.3703703703703703,0.6944444444444444,0.6569444444444444,2,1,5,4
selected,mem_seq,PASS,,energy_distance,mem_seq,val_normal,train_normal,0.95,0.4053776741027832,0.5833333333333334,1.0,0.16666666666666666,0.2857142857142857,0.19999999999999996,0.6111111111111112,0.6438492063492063,1,0,6,5
selected,mem_reset_seq,PASS,,euclidean,mem_reset_seq,val_normal,train_normal,0.95,0.8817520469427107,0.5,0.0,0.0,0.0,0.0,0.3888888888888889,0.5608164983164984,0,0,6,6
selected,mem_reset_seq,PASS,,trajectory_mahalanobis,mem_reset_seq,val_normal,train_normal,0.95,0.8462918698787688,0.4166666666666667,0.0,0.0,0.0,0.0,0.75,0.6746031746031745,0,1,5,6
selected,mem_reset_seq,PASS,,sobolev_h1,mem_reset_seq,val_normal,train_normal,0.95,0.09624630510807036,0.5833333333333334,0.6666666666666666,0.3333333333333333,0.4444444444444444,0.3703703703703703,0.6944444444444444,0.6496031746031746,2,1,5,4
selected,mem_reset_seq,PASS,,energy_distance,mem_reset_seq,val_normal,train_normal,0.95,0.386822497844696,0.5833333333333334,1.0,0.16666666666666666,0.2857142857142857,0.19999999999999996,0.6111111111111112,0.6438492063492063,1,0,6,5
selected,rate_seq,PASS,,euclidean,rate_seq,val_normal,train_normal,0.95,0.6165551662445068,0.5,0.0,0.0,0.0,0.0,0.6388888888888888,0.5801587301587301,0,0,6,6
selected,rate_seq,PASS,,trajectory_mahalanobis,rate_seq,val_normal,train_normal,0.95,0.5083509936928748,0.5,0.0,0.0,0.0,0.0,0.5555555555555556,0.5797979797979798,0,0,6,6
selected,rate_seq,PASS,,sobolev_h1,rate_seq,val_normal,train_normal,0.95,0.09028362967073918,0.5,0.5,0.5,0.5,0.5,0.5555555555555556,0.5862794612794613,3,3,3,3
selected,rate_seq,PASS,,energy_distance,rate_seq,val_normal,train_normal,0.95,0.3979275465011596,0.5,0.0,0.0,0.0,0.0,0.5277777777777778,0.5403138528138528,0,0,6,6
selected,h_seq,PASS,,euclidean,h_seq,val_normal,train_normal,0.95,1.0760106056928633,0.5,0.0,0.0,0.0,0.0,0.4444444444444444,0.4917027417027417,0,0,6,6
selected,h_seq,PASS,,trajectory_mahalanobis,h_seq,val_normal,train_normal,0.95,0.6695285081863402,0.4166666666666667,0.0,0.0,0.0,0.0,0.7222222222222222,0.6680555555555556,0,1,5,6
selected,h_seq,PASS,,sobolev_h1,h_seq,val_normal,train_normal,0.95,0.08370489254593849,0.6666666666666666,0.75,0.5,0.6,0.5357142857142857,0.6666666666666666,0.6453703703703704,3,1,5,3
selected,h_seq,PASS,,energy_distance,h_seq,val_normal,train_normal,0.95,0.3952772140502929,0.5,0.0,0.0,0.0,0.0,0.5833333333333334,0.6378306878306879,0,0,6,6
```

## 9. Prefix Benchmark Smoke Result

```csv
prefix_ratio,train_normal_count,val_normal_count,test_count,hidden_shape,threshold_value,Accuracy,Precision,Recall,F1,F2,AUC,PR_AUC,TP,FP,TN,FN,first_alarm_ratio,status
0.6,8,4,12,"[24, 9, 16]",0.8906889736652374,0.6666666666666666,1.0,0.3333333333333333,0.5,0.3846153846153846,0.8055555555555556,0.8218253968253969,2,0,6,4,not_applicable_without_onset_labels,PASS
0.8,8,4,12,"[24, 12, 16]",0.8453081965446471,0.5,0.0,0.0,0.0,0.0,0.7777777777777778,0.6968253968253967,0,0,6,6,not_applicable_without_onset_labels,PASS
1.0,8,4,12,"[24, 14, 16]",0.8462918698787688,0.4166666666666667,0.0,0.0,0.0,0.0,0.75,0.6746031746031745,0,1,5,6,not_applicable_without_onset_labels,PASS
```

## 10. No-leak Verification

- Reference fitting uses only `train_normal`.
- Threshold calibration uses only `val_normal` with q=0.95.
- Test samples are only evaluated.
- No test labels are used to select thresholds or references.

## 11. NaN / OOM / Shape Error Scan

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

## 12. Remaining Blockers

- selected dataset is synthetic_fallback; real data training chain has not been validated

## 13. Decision

```text
GO_TO_FULL_SINGLE_DATASET = NO
GO_TO_MULTI_DATASET = NO
```
