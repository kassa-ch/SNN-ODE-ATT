# Server Backfill Phase R0S

## Repository pruning and cleanup

Phase R0S pruned the public GitHub repository toward the requested top-level research structure:

- `Data/`
- `Models/`
- `Distance/`
- `Scrips/`
- `Detect/`
- `Results/`
- `docs/`

The prior Phase R0 server backfill source paths were not accessible from this local checkout:

- `/autodl-tmp/半导体晶圆项目ATT/`
- `/root/autodl-tmp/半导体晶圆项目ATT/`

Therefore Phase R0S did not copy additional server files. It reorganized the code already present in GitHub, moved low-risk legacy paths into the requested structure, and kept compatibility paths only where active imports would otherwise break.

### Moved into target structure

- `configs/` -> `Scrips/config/`
- `scripts/prepare_*.py` -> `Scrips/data_loader/`
- `scripts/train_exp*.py` -> `Scrips/train/`
- `scripts/eval_*.py` and `scripts/audit_sinkhorn.py` -> `Scrips/main/`
- `scripts/ops/*.sh` -> `Scrips/utils/ops/`
- `demos/` -> `Detect/demo*/`
- `checkpoints/README.md` -> `Results/models/checkpoints/README.md`
- root setup/report markdown files -> `Results/reports/`
- `tests/` -> `Scrips/tests/`

### Removed

- `Archive/` was removed because it only contained duplicate legacy dataset scaffold names already represented by canonical `Data/ST-AWFD/` and `Data/Bosch_Production_Line/`.
- The top-level `checkpoints/.gitkeep` placeholder was removed after checkpoint documentation moved under `Results/models/checkpoints/`.

### Compatibility paths retained

- `src/` remains because `snnodeatt` is the active package-safe implementation imported by model wrappers, distance wrappers, tests, and smoke scripts.
- `experiments/` remains because Phase 1/2/2R smoke scripts and anomaly-score benchmark scripts still import `experiments.*` modules directly.

These two paths are retained deliberately and are documented as compatibility implementation paths, not as new research-facing structure.
