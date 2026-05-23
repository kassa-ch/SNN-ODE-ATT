# Data

Canonical dataset landing area for the reorganized SNN-ODE-ATT workflow.

Large raw files, processed tensors, checkpoints, caches, and private dataset artifacts must not be committed to GitHub. Each dataset directory contains:

- `raw/`: original downloaded or authorized data files.
- `processed/`: converted BTD payloads.
- `manifests/`: split manifests and metadata.

Canonical BTD payload fields:

```text
x:       [N,T,D]
mask:    [N,T]
delta_t: [N,T]
time:    [N,T]
label:   [N]
split:   list[str] or [N]
```

STATUS: NO_REAL_DATA_IN_REPO
