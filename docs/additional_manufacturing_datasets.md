# Additional Manufacturing Datasets

This repository includes two supplementary production-quality datasets for
future SNN-ODE-ATT evaluation beyond the original wafer experiments.

## ST-AWFD

Source: https://github.com/STMicroelectronics/ST-AWFD

ST-AWFD is the stronger fit for the current wafer-oriented SNN-ODE-ATT story.
It targets semiconductor wafer fault detection and can be represented as a BTD
trajectory:

```text
B = wafer/material instance
T = ordered process samples or process steps
D = multivariate process features
```

Use case:

- wafer-level fault detection;
- semiconductor process monitoring;
- multivariate coupling drift across process steps;
- functional profile anomaly detection.

## Bosch Production Line Performance

Source: https://www.kaggle.com/c/bosch-production-line-performance

Bosch Production Line Performance is useful as a high-end manufacturing quality
anomaly supplement, but it is not a native continuous-time curve dataset. The
safe formulation is **staged BTD**:

```text
B = manufactured part
T = ordered station, stage, or grouped measurement block
D = measurements observed at that stage
```

Use case:

- production-line failure or quality anomaly detection;
- stage-wise multivariate coupling monitoring;
- discrete process-profile anomaly detection.

Paper-writing caveat:

Bosch should be described as a staged/discrete production sequence. It may be
used as a discrete approximation to a functional production profile, but should
not be claimed as a naturally continuous-time sensor trajectory.

## Unified payload

Both datasets should be converted to:

```python
{
    "x": Tensor[N, T, D],
    "mask": Tensor[N, T],
    "delta_t": Tensor[N, T],
    "tau": Tensor[N, T],
    "label": Tensor[N],
    "sample_id": List[str],
    "feature_names": List[str],
}
```

For Bosch, `tau` and `delta_t` may be stage-index based when physical time is
not available. For ST-AWFD, prefer physical duration/process time fields when
available.
