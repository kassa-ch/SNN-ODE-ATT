# Phase R0S Repository Prune Report

## 1. Target Directory Structure

The requested top-level structure is:

```text
Data/
Models/
Distance/
Scrips/
Detect/
Results/
docs/
```

Standard root files retained: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`, `requirements.txt`, and `environment.yml`.

## 2. Delete Strategy

Phase R0S used a conservative prune strategy:

- Move valid code/config/docs into requested target directories first.
- Use `git mv` for tracked migrations.
- Use `git rm` for tracked deletion.
- Keep active compatibility paths only when deletion would break imports or validated smoke tests.
- Do not move large data, checkpoints, payloads, hidden cache, or runtime outputs into Git.

## 3. KEEP List

| Path | Reason |
|---|---|
| `Data/` | Requested dataset scaffold and future BTD payload landing area |
| `Models/` | Requested statistical/neural model mapping area |
| `Distance/` | Requested distance/scoring mapping area |
| `Scrips/` | Requested script/config/train/data-loader organization |
| `Detect/` | Requested demo/detection organization |
| `Results/` | Requested result/checkpoint/log/report organization |
| `docs/` | Audit, migration, and experiment reports |
| root metadata files | Public repo metadata and environment requirements |

## 4. MIGRATE_THEN_DELETE List

| Source | Destination | Status |
|---|---|---|
| `configs/` | `Scrips/config/` | migrated |
| `scripts/prepare_*.py` | `Scrips/data_loader/` | migrated |
| `scripts/train_exp*.py` | `Scrips/train/` | migrated |
| `scripts/eval_*.py`, `scripts/audit_sinkhorn.py` | `Scrips/main/` | migrated |
| `scripts/ops/*.sh` | `Scrips/utils/ops/` | migrated |
| `demos/` | `Detect/demo*/` | migrated |
| `checkpoints/README.md` | `Results/models/checkpoints/README.md` | migrated |
| root report markdown files | `Results/reports/` | migrated |
| `tests/` | `Scrips/tests/` | migrated |

## 5. DELETE List

| Path | Reason |
|---|---|
| `Archive/` | Duplicate old-name dataset scaffolds; canonical names now live under `Data/` |
| `checkpoints/.gitkeep` | Top-level checkpoint scaffold replaced by `Results/models/checkpoints/` |
| untracked `__pycache__/` and old empty directories | Runtime artifacts or empty directories after migration |

## 6. REVIEW_MANUALLY List

| Path | Reason | Decision |
|---|---|---|
| `src/` | Active import root for `snnodeatt`; wrappers and tests still import it | retained as compatibility implementation path |
| `experiments/` | Active Phase 1/2/2R smoke pipeline and anomaly benchmark scripts import this package | retained as compatibility experiment path |

## 7. Actual Deleted Paths

- `Archive/`
- `checkpoints/.gitkeep`
- untracked old empty directories: `configs/`, `demos/`, `scripts/`, `tests/`
- untracked runtime caches: `__pycache__/`, ignored `experiments/*/runs/`

Tracked deleted file count: 13.

## 8. Actual Migrated Paths

Tracked migrated file count: 49.

Main migration groups:

- configuration YAML files to `Scrips/config/`
- prepare scripts to `Scrips/data_loader/`
- train wrappers to `Scrips/train/`
- evaluation/audit wrappers to `Scrips/main/`
- ops shell scripts to `Scrips/utils/ops/`
- demo scripts to `Detect/demo*/`
- test scripts to `Scrips/tests/`
- root reports to `Results/reports/`

## 9. Retained Old Paths and Reasons

`src/` and `experiments/` remain the only non-target top-level implementation paths.

They are retained because removing them in this pass would break active imports:

- `Models/*` wrappers import `snnodeatt.*` from `src/`.
- `Distance/*` wrappers import `snnodeatt.scoring.*` from `src/`.
- Phase 1/2/2R smoke scripts import `experiments.multidata_snnodeatt_distance_benchmark.*`.
- The migrated smoke tests still validate the active package through `src/`.

This is why the repository status is `YES_WITH_COMPATIBILITY_WRAPPERS`, not `YES_FULLY`.

## 10. .gitignore Update

`.gitignore` now explicitly ignores:

- `Data/**/raw/*`
- `Data/**/processed/*`
- binary payload/model/data extensions such as `.pt`, `.pth`, `.ckpt`, `.npy`, `.npz`, `.pkl`, `.h5`, `.mat`, `.parquet`
- runtime outputs under `Results/models/checkpoints/`, `Results/models/best_models/`, `Results/hidden_cache/`, `Results/logs/`
- run/cache directories such as `runs/`, `cache/`, `hidden_cache/`, `checkpoints/`, and experiment run folders

It preserves README and `.gitkeep` files for tracked structure.

## 11. Cleanup Tree

See `docs/repo_tree_after_prune.txt` for the generated directory tree.

## 12. Validation Results

Validation was run after pruning:

- `python -m compileall Models Distance Scrips Detect`: PASS
- import smoke for model, statistical, distance, and Scrips wrappers: PASS
- migrated smoke tests under `Scrips/tests/`: PASS
- sensitive scan: PASS, no matches
- `>50MB` large file scan: PASS, no files found
- `git status --short`: expected migrations/deletions/new reports only before commit

The final command outputs are summarized in the Phase R0S completion response.

## 13. Large File and Sensitive Scan

No large files over 50MB were intentionally added. No secrets were expected or intentionally added.

## 14. Remaining Non-target Paths

Remaining non-target implementation paths:

- `src/`
- `experiments/`

Both are retained because of active import compatibility. They should be migrated in a future Phase R0T only after a package/import redesign.

## 15. Does the Repository Now Follow the Requested Top-level Structure?

YES_WITH_COMPATIBILITY_WRAPPERS
