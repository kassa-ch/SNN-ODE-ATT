# Local Canonical GitHub Sync Plan

## Source Roles

`C:\Users\kassa\Desktop\SNNODEATT` is the canonical source for this project. Local files in this path define the intended code, configuration, lightweight manifests, and documentation state.

GitHub repository `git@github.com:kassa-ch/SNN-ODE-ATT.git` / `https://github.com/kassa-ch/SNN-ODE-ATT.git` is a remote mirror and the source that servers should clone or pull from. GitHub should not be treated as an independent source of truth when it differs from the local canonical project.

The server reached through `ssh -p 31708 root@connect.bjb1.seetacloud.com` is a training worker only. Its role is to clone or pull GitHub, install environments, train models, generate experiment outputs, and return results to the local `Results` tree. Main code should not be manually edited on the server.

## Files To Commit

Commit source code, lightweight configuration, documentation, and compact audit outputs:

- `README.md`, `requirements.txt`, and `.gitignore`.
- `Scrips/**/*.py`, including public dataset converters, loaders, configs, training wrappers, and utilities.
- `Models/**/*.py`, `Distance/**/*.py`, `Detect/**/*.py`, and `experiments/**/*.py`.
- Experiment YAML files under `Scrips/config/experiments/`.
- Markdown documentation under `docs/`.
- Dataset-level `Data/**/README.md` and compact `Data/**/manifests/*.json`.
- Lightweight `Results/manifests/*.yaml`, `Results/manifests/*.json`, `Results/reports/*.md`, and `Results/tables/*.csv`.

## Files Not To Commit

Keep raw data, generated sample CSVs, checkpoints, caches, and large payloads local:

- `Data/**/Origin_Datasets/`
- `Data/**/preprocess_data/`
- `Data/**/processed/`
- `Data/**/*.pt`, `Data/**/*.pth`, `Data/**/*.npy`, `Data/**/*.npz`
- `Results/models/checkpoints/`
- `Results/models/best_models/`
- `Results/hidden_cache/`
- `Results/benchmarks/**/raw/`
- `saved_models/`, `logs/`, `detection_logs/`, `score_plots/`, `score_results/`
- Python caches, virtual environments, local `.env*`, private keys, and `secrets/`

## Data And Training Artifacts

Large public datasets and converted per-sample CSVs stay under local `Data/` and should not be pushed. If a server needs data, copy or mount the data intentionally outside Git. GitHub should contain only scripts, protocols, and manifests that describe how to reproduce preprocessing.

Training checkpoints, hidden caches, raw benchmark payloads, and full result dumps should remain local or be transferred through an explicit artifact path. Only compact summaries, reports, and manifests should be committed.

## Server Result Return

Server-generated experiment outputs should be copied back into:

```text
C:\Users\kassa\Desktop\SNNODEATT\Results
```

After return, keep heavy files in ignored subdirectories such as `Results/models/checkpoints/`, `Results/models/best_models/`, `Results/hidden_cache/`, or `Results/benchmarks/**/raw/`. Promote only compact CSV summaries, reports, or manifests into commit-eligible locations.

## GitHub Initialization And History Safety

If the local canonical project is not already a Git repository, initialize only after review:

```powershell
git init
git remote add origin git@github.com:kassa-ch/SNN-ODE-ATT.git
git fetch origin
```

Before making the first local commit or setting upstream branches, inspect the fetched remote history:

```powershell
git branch -r
git log --oneline --decorate --all -n 20
```

If the remote already has meaningful history, avoid overwriting it blindly. Prefer creating a local branch, comparing with `origin/main` or `origin/master`, and using an explicit merge, rebase, or replacement plan approved by the project owner. Do not force push without explicit approval.

## Recommended Commit Split

1. `config/env completion`
2. `wafer baseline archive`
3. `public dataset benchmark scaffold`
4. `TEP converter and preprocessing protocol`
5. `Distance modular calculator and demo1 integration`
6. `public training/detection smoke wrappers`

This split keeps environment/config work, dataset scaffolding, TEP conversion, distance integration, and smoke wrappers independently reviewable.
