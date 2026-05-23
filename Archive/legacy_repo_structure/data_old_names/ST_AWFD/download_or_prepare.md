# Download or Prepare ST-AWFD

1. Visit the public ST-AWFD repository:
   https://github.com/STMicroelectronics/ST-AWFD
2. Follow the dataset instructions and license terms from the provider.
3. Put raw files under `data/ST_AWFD/raw/`.
4. Convert them to the unified BTD payload with `scripts/prepare_st_awfd.py`.

The processed payload should follow:

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
