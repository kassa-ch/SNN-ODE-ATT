# TEP Preprocessing Protocol

## Layout

TEP raw files should be archived under:

```text
Data/TEP/Origin_Datasets/raw/
```

`Origin_Datasets` is the immutable raw archive. Do not rewrite, normalize, move, or delete files there during preprocessing. The converter writes trainer-ready samples to:

```text
Data/TEP/preprocess_data/
```

Each output CSV is one time-series sample with shape `[T,D]`. The CSV body contains numeric feature values only; time, run id, label, and non-numeric columns are excluded. A header row is written to match the current `TimeSeriesDataset` CSV reader contract.

## Label Rules

If a source file contains a `fault_id`, `fault`, `label`, `target`, `class`, `idv`, or similar label column:

- `0` means normal.
- Any positive numeric value means abnormal.

If there is no label column, the converter infers labels from the file name, sheet name, and nearby source path tokens:

- `normal`, `nofault`, `d00`, `fault0` mean normal.
- `fault`, `d01`, `fault01`, `idv` mean abnormal.

TEP public data already contains native normal/fault semantics, so no augmented anomalies are generated.

## Uniform And Nonuniform Views

TEP uses a single `Data/TEP/preprocess_data/` directory. Uniform and nonuniform experiments do not create duplicate datasets:

- `Scrips/config/experiments/tep_uniform.yaml` sets `apply_poisson_sampling: false`.
- `Scrips/config/experiments/tep_nonuniform.yaml` sets `apply_poisson_sampling: true`.

## Dependencies

Excel `.xlsx` conversion uses `pandas.read_excel` with `openpyxl`. Install the missing dependency in the active environment with:

```powershell
pip install openpyxl
```

The converter reports `MISSING_DEPENDENCY: openpyxl` and exits non-fatally when `.xlsx` candidates exist but `openpyxl` is unavailable.

## Commands

Check inputs and dependencies without generating sample CSVs:

```powershell
python -B Scrips/data_loader/public_datasets/convert_tep.py --check-only --input Data/TEP/Origin_Datasets --output Data/TEP/preprocess_data
```

Convert real raw files after placing them under `Data/TEP/Origin_Datasets/raw/`:

```powershell
python -B Scrips/data_loader/public_datasets/convert_tep.py --input Data/TEP/Origin_Datasets --output Data/TEP/preprocess_data --window-size 128 --stride 64 --min-window-size 32
```

Audit generated preprocess CSVs:

```powershell
python -B Scrips/data_loader/public_datasets/audit_tep_preprocess.py --check-only --data_dir Data/TEP/preprocess_data
```

The converter writes `Data/TEP/manifests/tep_preprocess_manifest.json`. The audit writes `Results/manifests/tep_preprocess_audit.json`.
