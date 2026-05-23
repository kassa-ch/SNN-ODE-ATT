# Download or Prepare Bosch Production Line Performance

1. Obtain the dataset from Kaggle:
   https://www.kaggle.com/c/bosch-production-line-performance
2. Follow Kaggle and Bosch dataset terms.
3. Put raw files under `data/BoschProductionLine/raw/`.
4. Convert station/stage columns into a staged BTD payload with
   `scripts/prepare_bosch.py`.

The recommended staged BTD payload is:

```python
{
    "x": Tensor[N, T, D],
    "mask": Tensor[N, T],
    "delta_t": Tensor[N, T],
    "tau": Tensor[N, T],
    "label": Tensor[N],
    "sample_id": List[str],
    "feature_names": List[str],
    "stage_names": List[str],
}
```

For Bosch, `delta_t` and `tau` may be stage-index based if physical timestamps
are unavailable. This should be documented as staged/discrete time rather than
native continuous time.
