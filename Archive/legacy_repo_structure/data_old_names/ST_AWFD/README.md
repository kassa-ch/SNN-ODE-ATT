# ST-AWFD

ST-AWFD is a public wafer fault detection dataset released by STMicroelectronics.
It is a strong fit for SNN-ODE-ATT because each wafer/material can be converted
to a multivariate trajectory:

```text
x:       [B, T, D]
mask:    [B, T]
delta_t: [B, T]
tau:     [B, T]
label:   [B]
```

Recommended interpretation:

- `B`: wafer/material instances.
- `T`: ordered process samples or process-step observations.
- `D`: multivariate process features.
- `label`: wafer-level or material-level fault/quality target.

This dataset is suitable for high-end semiconductor manufacturing process
monitoring, wafer quality anomaly detection, and multivariate coupling drift
analysis.

Raw files are not committed to this repository. Place downloaded files under
`data/ST_AWFD/raw/`, then adapt and run:

```bash
python scripts/prepare_st_awfd.py --raw-dir data/ST_AWFD/raw --out-dir data/ST_AWFD/processed
```

Reference repository:

- https://github.com/STMicroelectronics/ST-AWFD
