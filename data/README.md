# Data Layout

Raw and processed datasets are intentionally not committed.

Recommended processed `.pt` payload:

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

Use Git LFS or external storage for large processed artifacts. SWaT/WADI users must follow the original access agreement.

## Included dataset entrypoints

- `HAI`: industrial control / HIL process data.
- `SWaT`: water treatment cyber-physical process data; access-controlled.
- `WADI`: water distribution cyber-physical process data; access-controlled.
- `TEP`: Tennessee Eastman Process simulation benchmark.
- `ST_AWFD`: semiconductor wafer fault detection. This is the closest public
  supplement to the wafer manufacturing setting and can be represented as
  `B=wafer/material`, `T=ordered process observations`, `D=process features`.
- `BoschProductionLine`: production-line quality/failure data. This should be
  treated as **staged BTD**: `T` is the ordered station/stage/measurement-block
  axis. It is not a native continuous-time curve, but it can be used as a
  discrete approximation to a functional production profile.
