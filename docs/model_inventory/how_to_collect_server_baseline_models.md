# How to Collect Server Baseline Model Files

This document describes the safe GitHub-centered workflow for collecting baseline model `.py` files from the server into the organized SNN-ODE-ATT repository.

## Target Methods

The collection script looks for the following methods:

| Method category | Representative method |
|---|---|
| Linear prediction | LinearAR |
| Sparse linear model | LassoAR |
| Kernel method | KRR |
| Functional statistics | FPCA |
| Integral trend | CumInt / Cumlnt |
| Spiking neural network | SNN |
| Gated recurrent network | GRU |
| Continuous-time latent model | Latent ODE |
| Continuous-time recurrent model | ODERNN |
| Probabilistic latent factor model | DF2M |
| Paper method | SNN-ODE |

## Server Command

Run on the server:

```bash
cd /root/autodl-tmp/SNNODEATT
git pull origin main
bash scripts/ops/collect_server_baseline_model_files.sh
```

The script will:

- search `/root/autodl-tmp/wafer_att_ctsr_vca` and `/root/autodl-tmp/SNNODEATT` for small Python model files;
- copy the best matching files into `src/snnodeatt/models/baselines/`;
- generate `docs/model_inventory/server_baseline_model_file_inventory.md`;
- generate `docs/model_inventory/server_baseline_model_file_inventory.json`;
- run a sensitive-string scan and a large-file check before committing;
- commit and push only small source files and inventory reports.

## Safety Rules

- It does not copy raw data.
- It does not copy checkpoints.
- It does not copy cache files.
- It does not run training.
- It does not run evaluation.
- It does not overwrite the main SNN-ODE-ATT model implementation.

If a method is marked `missing`, review the candidate list in the inventory report and decide whether a manual integration is needed.
