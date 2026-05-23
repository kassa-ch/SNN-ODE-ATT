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
