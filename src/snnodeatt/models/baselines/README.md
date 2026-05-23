# Baseline Model Implementations

This directory stores baseline model `.py` files recovered from the original server project for reproducible comparison with SNN-ODE-ATT.

Expected method families:

- LinearAR
- LassoAR
- KRR
- FPCA
- CumInt / cumulative integral trend
- SNN
- GRU
- Latent ODE
- ODE-RNN
- DF2M
- SNN-ODE

The server-side collection script is:

```bash
bash scripts/ops/collect_server_baseline_model_files.sh
```

It searches the original server project read-only, copies matching Python files here, writes an inventory report, and commits/pushes only small source files and reports.
