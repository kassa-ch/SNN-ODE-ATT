# Phase R0S Repository Prune Plan

This plan is generated before pruning non-target top-level paths. The requested top-level research structure is:

- `Data/`
- `Models/`
- `Distance/`
- `Scrips/`
- `Detect/`
- `Results/`
- `docs/`

The repository also keeps standard root files such as `README.md`, `LICENSE`, `.gitignore`, requirements/environment files, and Git metadata.

## Classification Table

| path | current_status | reason | migration_target | deletion_safe | dependency_check | notes |
|---|---|---|---|---|---|---|
| `Data/` | KEEP | Requested dataset landing area | n/a | NO | Contains README/.gitkeep scaffold only | Heavy data remains ignored |
| `Models/` | KEEP | Requested model mapping area | n/a | NO | Wrappers point to active implementation | Server backfill path unavailable in Phase R0 |
| `Distance/` | KEEP | Requested distance mapping area | n/a | NO | Wrappers point to active scoring implementation | No new distance math added |
| `Scrips/` | KEEP | Requested script/logical pipeline area | n/a | NO | Will receive legacy scripts/config/tests | User spelling preserved |
| `Detect/` | KEEP | Requested demo/detection area | n/a | NO | Will receive demo scripts | Demo code remains lightweight |
| `Results/` | KEEP | Requested results management area | n/a | NO | Will receive root historical reports/checkpoint README | Heavy runtime outputs ignored |
| `docs/` | KEEP | Requested docs/report area | n/a | NO | Stores audit and migration reports | Historical reports retained |
| `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `environment.yml`, `.gitattributes` | KEEP | Necessary repository files | n/a | NO | Required public repo metadata/config | `.gitattributes` kept for Git LFS patterns |
| `configs/` | MIGRATE_THEN_DELETE | Valid config scaffold but non-target top-level path | `Scrips/config/` | YES after move | No active imports require top-level `configs/`; docs/scripts references will be updated | YAML files only |
| `scripts/` | MIGRATE_THEN_DELETE | Valid CLI/ops scaffold but non-target top-level path | `Scrips/data_loader/`, `Scrips/train/`, `Scrips/main/`, `Scrips/utils/ops/` | YES after move | Docs references will be updated; files are wrappers/scaffolds | Server ops scripts moved under `Scrips/utils/ops/` |
| `demos/` | MIGRATE_THEN_DELETE | Valid demo scripts but non-target top-level path | `Detect/demo1/` ... `Detect/demo4/` | YES after move | README references will be updated | Demo scripts are lightweight |
| `checkpoints/` | MIGRATE_THEN_DELETE | Checkpoint placeholder but non-target top-level path | `Results/models/checkpoints/` | YES after move | No code imports top-level checkpoint scaffold | Checkpoint binaries remain ignored |
| root historical reports | MIGRATE_THEN_DELETE | Reports should not live at repo root | `Results/reports/` | YES after move | No imports | Includes setup/GitHub/anomaly benchmark reports |
| `tests/` | MIGRATE_THEN_DELETE | Verification code is useful but non-target top-level path | `Scrips/tests/` | YES after updating test path helpers | Tests import active `snnodeatt` package via `src/` path | Test entry becomes `python Scrips/tests/test_*.py` or pytest discovery |
| `Archive/` | DELETE | Phase R0 archive contains only old duplicate dataset scaffolds already represented under `Data/` | n/a | YES | No imports | Removed to match target top-level structure |
| `src/` | REVIEW_MANUALLY | Active package-safe implementation used by tests, wrappers, and smoke scripts | Potential future migration to `Scrips/snnodeatt/` or packaged install | NO | `Models/*`, `Distance/*`, `experiments/*`, and tests import `snnodeatt` from `src/` | Kept as compatibility implementation path in this pass |
| `experiments/` | REVIEW_MANUALLY | Active Phase 1/2/2R smoke pipeline and anomaly benchmark scripts | Potential future migration to `Scrips/train/` and `Results/benchmarks/` | NO | Phase smoke scripts import `experiments.multidata_snnodeatt_distance_benchmark` | Kept as compatibility experiment path in this pass |

## Delete Strategy

1. Move low-risk tracked files with `git mv`.
2. Remove only paths whose valid code has been moved or whose contents are duplicate placeholders.
3. Keep `src/` and `experiments/` temporarily because deleting them would break active imports and previously validated smoke scripts.
4. Update `.gitignore` so raw data, payloads, checkpoints, hidden caches, and runtime outputs stay out of Git.
5. Run compile/import/tests/sensitive/large-file checks after pruning.
