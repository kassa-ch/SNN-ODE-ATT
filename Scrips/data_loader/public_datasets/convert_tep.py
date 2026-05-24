from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "Data" / "TEP"
DEFAULT_INPUT_DIR = DATASET_DIR / "Origin_Datasets" / "raw"
DEFAULT_OUTPUT_DIR = DATASET_DIR / "preprocess_data"
DEFAULT_PROCESSED_DIR = DATASET_DIR / "processed"
DEFAULT_MANIFEST_PATH = DATASET_DIR / "manifests" / "tep_preprocess_manifest.json"

SUPPORTED_READ_EXTENSIONS = {".xlsx", ".xls", ".csv", ".mat"}
DISCOVERY_EXTENSIONS = SUPPORTED_READ_EXTENSIONS | {".txt"}

NORMAL_PATTERNS = [
    r"(^|[^a-z0-9])normal([^a-z0-9]|$)",
    r"nofault",
    r"(^|[^a-z0-9])d0*0([^0-9]|$)",
    r"fault[_\s-]*0([^0-9]|$)",
]
ABNORMAL_PATTERNS = [
    r"idv",
    r"faults?",
    r"(^|[^a-z0-9])d0*[1-9][0-9]*([^0-9]|$)",
    r"fault[_\s-]*0*[1-9][0-9]*([^0-9]|$)",
]


@dataclass
class WindowSample:
    name: str
    matrix: object
    label: int
    label_name: str
    source_path: str
    sheet_or_run: str


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _dependency_status() -> Dict[str, str]:
    return {
        "numpy": "available" if _has_module("numpy") else "missing",
        "pandas": "available" if _has_module("pandas") else "missing",
        "openpyxl": "available" if _has_module("openpyxl") else "missing",
        "xlrd": "available" if _has_module("xlrd") else "missing",
        "scipy": "available" if _has_module("scipy") else "missing",
        "torch": "available" if _has_module("torch") else "missing",
    }


def _safe_name(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "sample"


def _normalize_col(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _is_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 2048:
            return False
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return False
    return bool(lines) and lines[0].startswith("version https://git-lfs.github.com/spec/")


def _discover_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in DISCOVERY_EXTENSIONS
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _discover_external_candidates(input_dir: Path) -> List[Path]:
    if not DATASET_DIR.exists():
        return []
    return sorted(path for path in _discover_files(DATASET_DIR) if not _is_relative_to(path, input_dir))


def _infer_time_columns(columns: Iterable[object]) -> List[object]:
    out = []
    exact = {
        "time",
        "timestamp",
        "datetime",
        "date",
        "index",
        "idx",
        "sample",
        "samples",
        "step",
        "t",
    }
    for col in columns:
        norm = _normalize_col(col)
        raw = str(col).strip().lower()
        if norm in exact or raw.startswith("unnamed:"):
            out.append(col)
    return out


def _infer_run_columns(columns: Iterable[object]) -> List[object]:
    out = []
    exact = {
        "run",
        "runid",
        "runindex",
        "trial",
        "trialid",
        "case",
        "caseid",
        "batch",
        "batchid",
        "sequence",
        "sequenceid",
        "seq",
        "seqid",
        "simulation",
        "simulationid",
        "sampleid",
    }
    for col in columns:
        norm = _normalize_col(col)
        if norm in exact or norm.endswith("runid"):
            out.append(col)
    return out


def _infer_label_columns(columns: Iterable[object]) -> List[object]:
    out = []
    exact = {
        "fault",
        "faultid",
        "faultnumber",
        "faultno",
        "faulttype",
        "fault_id",
        "label",
        "labels",
        "target",
        "class",
        "y",
        "idv",
        "attack",
        "isfault",
        "isfaulty",
        "abnormal",
        "isabnormal",
    }
    for col in columns:
        if _normalize_col(col) in {_normalize_col(item) for item in exact}:
            out.append(col)
    return out


def _infer_label_from_context(path: Path, sheet_name: str) -> Tuple[Optional[str], str]:
    parent_tokens = " ".join(part.lower() for part in path.parts[-5:-1])
    context = f"{path.stem.lower()} {str(sheet_name).lower()} {parent_tokens}"
    for pattern in NORMAL_PATTERNS:
        if re.search(pattern, context):
            return "normal", "filename_or_sheet_normal"
    for pattern in ABNORMAL_PATTERNS:
        if re.search(pattern, context):
            return "abnormal", "filename_or_sheet_abnormal"
    return None, "unresolved"


def _load_numpy_pandas():
    import numpy as np
    import pandas as pd

    return np, pd


def _iter_dataframes(path: Path) -> Iterator[Tuple[str, object]]:
    _, pd = _load_numpy_pandas()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, sep=None, engine="python")
        yield "csv", frame
        return
    if suffix == ".xlsx":
        excel = pd.ExcelFile(path, engine="openpyxl")
        for sheet_name in excel.sheet_names:
            yield str(sheet_name), excel.parse(sheet_name=sheet_name)
        return
    if suffix == ".xls":
        excel = pd.ExcelFile(path)
        for sheet_name in excel.sheet_names:
            yield str(sheet_name), excel.parse(sheet_name=sheet_name)
        return
    if suffix == ".mat":
        import scipy.io

        np, pd = _load_numpy_pandas()
        mat = scipy.io.loadmat(path)
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            if arr.ndim != 2 or min(arr.shape) == 0:
                continue
            if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 64:
                arr = arr.T
            columns = [f"feature_{idx:03d}" for idx in range(arr.shape[1])]
            yield key, pd.DataFrame(arr, columns=columns)


def _label_values(series, default_label: Optional[str]):
    np, pd = _load_numpy_pandas()
    numeric = pd.to_numeric(series, errors="coerce")
    values = []
    for raw, num in zip(series.tolist(), numeric.tolist()):
        if isinstance(num, (int, float)) and math.isfinite(float(num)):
            values.append(1 if float(num) > 0 else 0)
            continue
        text = str(raw).strip().lower()
        if any(re.search(pattern, text) for pattern in NORMAL_PATTERNS):
            values.append(0)
        elif any(re.search(pattern, text) for pattern in ABNORMAL_PATTERNS):
            values.append(1)
        elif text in {"0", "false", "no"}:
            values.append(0)
        elif text in {"1", "true", "yes"}:
            values.append(1)
        else:
            values.append(1 if default_label == "abnormal" else 0)
    return np.asarray(values, dtype=int)


def _window_ranges(n_rows: int, window_size: int, stride: int, min_window_size: int) -> List[Tuple[int, int]]:
    if n_rows < min_window_size:
        return []
    if n_rows <= window_size:
        return [(0, n_rows)]
    ranges = [(start, start + window_size) for start in range(0, n_rows - window_size + 1, stride)]
    tail_start = ranges[-1][1] if ranges else 0
    if n_rows - tail_start >= min_window_size:
        ranges.append((tail_start, n_rows))
    return ranges


def _write_matrix_csv(path: Path, header: Sequence[str], matrix, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        return "skipped_existing"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in matrix:
            writer.writerow([f"{float(value):.10g}" for value in row])
    return "written"


def _is_float_token(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _detect_skiprows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            first_row = next(csv.reader(handle), [])
    except Exception:
        return 0
    if not first_row:
        return 0
    return 0 if all(_is_float_token(token) for token in first_row) else 1


def _read_output_csv_shape(path: Path) -> Optional[Tuple[int, int]]:
    import numpy as np

    try:
        arr = np.loadtxt(path, delimiter=",", skiprows=_detect_skiprows(path), dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return int(arr.shape[0]), int(arr.shape[1])
    except Exception:
        return None


def _scan_output_csvs(output_dir: Path) -> Dict[str, object]:
    normal_paths = sorted(output_dir.glob("*_normal.csv")) if output_dir.exists() else []
    abnormal_paths = sorted(output_dir.glob("*_abnormal.csv")) if output_dir.exists() else []
    lengths = Counter()
    dims = Counter()
    for path in normal_paths + abnormal_paths:
        shape = _read_output_csv_shape(path)
        if shape is None:
            continue
        lengths[str(shape[0])] += 1
        dims[str(shape[1])] += 1
    return {
        "n_normal_csv": len(normal_paths),
        "n_abnormal_csv": len(abnormal_paths),
        "feature_dim_distribution": dict(sorted(dims.items(), key=lambda item: int(item[0]))),
        "length_distribution": dict(sorted(lengths.items(), key=lambda item: int(item[0]))),
    }


def _process_dataframe(
    *,
    frame,
    source_path: Path,
    sheet_name: str,
    output_dir: Path,
    window_size: int,
    stride: int,
    min_window_size: int,
    overwrite: bool,
    counters: Dict[str, Counter],
    skipped_unreadable_files: List[Dict[str, str]],
) -> List[WindowSample]:
    np, pd = _load_numpy_pandas()
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        skipped_unreadable_files.append({"path": str(source_path), "sheet": sheet_name, "reason": "empty_sheet"})
        return []

    columns = list(frame.columns)
    label_cols = _infer_label_columns(columns)
    time_cols = _infer_time_columns(columns)
    run_cols = _infer_run_columns(columns)
    excluded = set(label_cols + time_cols + run_cols)

    feature_data = {}
    feature_headers = []
    for col in columns:
        if col in excluded:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        header = _safe_name(col)
        while header in feature_headers:
            header = f"{header}_{len(feature_headers)}"
        feature_headers.append(header)
        feature_data[col] = numeric

    if not feature_data:
        skipped_unreadable_files.append({"path": str(source_path), "sheet": sheet_name, "reason": "no_numeric_features"})
        return []

    numeric_df = pd.DataFrame(feature_data, index=frame.index).replace([np.inf, -np.inf], np.nan)
    if numeric_df.isna().any().any():
        numeric_df = numeric_df.ffill().bfill().fillna(0.0)

    default_label, label_rule = _infer_label_from_context(source_path, sheet_name)
    if label_cols:
        labels = _label_values(frame[label_cols[0]], default_label)
        label_rule = f"column:{label_cols[0]}"
    else:
        if default_label is None:
            skipped_unreadable_files.append({"path": str(source_path), "sheet": sheet_name, "reason": "unable_to_infer_label"})
            return []
        labels = np.full(len(frame), 1 if default_label == "abnormal" else 0, dtype=int)

    run_col = run_cols[0] if run_cols else None
    groups = frame.groupby(run_col, sort=False, dropna=False).groups.items() if run_col else [("sheet", frame.index)]
    source_stem = _safe_name(source_path.stem)
    samples: List[WindowSample] = []

    for run_value, index_values in groups:
        index_list = list(index_values)
        if not index_list:
            continue
        matrix = numeric_df.loc[index_list].to_numpy(dtype=float)
        group_labels = labels[frame.index.get_indexer(index_list)]
        sheet_or_run = _safe_name(sheet_name if not run_col else f"{sheet_name}_run_{run_value}")
        for win_idx, (start, end) in enumerate(_window_ranges(len(index_list), window_size, stride, min_window_size)):
            window = matrix[start:end]
            if window.ndim != 2 or window.shape[0] < min_window_size:
                continue
            window_labels = group_labels[start:end]
            label = int(window_labels.max() > 0)
            label_name = "abnormal" if label else "normal"
            output_name = f"tep_{source_stem}_{sheet_or_run}_win{win_idx:05d}_{label_name}.csv"
            status = _write_matrix_csv(output_dir / output_name, feature_headers, window, overwrite)
            samples.append(
                WindowSample(
                    name=output_name,
                    matrix=window,
                    label=label,
                    label_name=label_name,
                    source_path=str(source_path),
                    sheet_or_run=sheet_or_run,
                )
            )
            if status == "written":
                counters["csv_written"][label_name] += 1
            else:
                counters["csv_skipped_existing"][label_name] += 1
            counters["label_rules"][label_rule] += 1

    if not samples:
        skipped_unreadable_files.append({"path": str(source_path), "sheet": sheet_name, "reason": "no_windows_written"})
    return samples


def _split_samples(samples: Sequence[WindowSample], seed: int, val_ratio: float, test_normal_ratio: float) -> List[str]:
    import numpy as np

    rng = np.random.default_rng(seed)
    splits = ["test" if sample.label else "train" for sample in samples]
    normal_indices = [idx for idx, sample in enumerate(samples) if sample.label == 0]
    shuffled = np.asarray(normal_indices, dtype=int)
    rng.shuffle(shuffled)

    n_normal = len(shuffled)
    n_test = int(round(n_normal * test_normal_ratio))
    n_val = int(round(n_normal * val_ratio))
    if n_normal >= 3:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
    if n_test + n_val > n_normal:
        n_val = max(0, n_normal - n_test)

    test_normal = set(int(idx) for idx in shuffled[:n_test])
    val_normal = set(int(idx) for idx in shuffled[n_test : n_test + n_val])
    for idx in normal_indices:
        if idx in test_normal:
            splits[idx] = "test"
        elif idx in val_normal:
            splits[idx] = "val"
        else:
            splits[idx] = "train"
    return splits


def _make_uniform_payload(samples: Sequence[WindowSample], splits: Sequence[str], seed: int) -> Dict[str, object]:
    import numpy as np
    import torch

    max_t = max(sample.matrix.shape[0] for sample in samples)
    dim = samples[0].matrix.shape[1]
    x = np.zeros((len(samples), max_t, dim), dtype=np.float32)
    mask = np.zeros((len(samples), max_t), dtype=np.float32)
    delta_t = np.zeros((len(samples), max_t), dtype=np.float32)
    time = np.zeros((len(samples), max_t), dtype=np.float32)
    label = np.asarray([sample.label for sample in samples], dtype=np.int64)

    for idx, sample in enumerate(samples):
        rows = sample.matrix.shape[0]
        x[idx, :rows, :] = sample.matrix.astype(np.float32)
        mask[idx, :rows] = 1.0
        delta_t[idx, :rows] = 1.0
        time[idx, :rows] = np.arange(rows, dtype=np.float32)

    return {
        "x": torch.from_numpy(x),
        "mask": torch.from_numpy(mask),
        "delta_t": torch.from_numpy(delta_t),
        "time": torch.from_numpy(time),
        "label": torch.from_numpy(label),
        "split": list(splits),
        "meta": {
            "dataset_name": "TEP",
            "view": "uniform",
            "seed": seed,
            "n_samples": len(samples),
            "max_length": max_t,
            "feature_dim": dim,
            "label_meaning": {"0": "normal", "1": "abnormal"},
            "source_files": sorted({sample.source_path for sample in samples}),
        },
    }


def _make_nonuniform_payload(samples: Sequence[WindowSample], splits: Sequence[str], seed: int) -> Dict[str, object]:
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    max_t = max(sample.matrix.shape[0] for sample in samples)
    dim = samples[0].matrix.shape[1]
    x = np.zeros((len(samples), max_t, dim), dtype=np.float32)
    mask = np.zeros((len(samples), max_t), dtype=np.float32)
    delta_t = np.zeros((len(samples), max_t), dtype=np.float32)
    time = np.zeros((len(samples), max_t), dtype=np.float32)
    label = np.asarray([sample.label for sample in samples], dtype=np.int64)

    for idx, sample in enumerate(samples):
        rows = sample.matrix.shape[0]
        keep = rng.random(rows) < 0.8
        if rows:
            keep[0] = True
            keep[-1] = True
        kept_positions = np.flatnonzero(keep)
        x[idx, kept_positions, :] = sample.matrix[kept_positions].astype(np.float32)
        mask[idx, kept_positions] = 1.0
        time[idx, kept_positions] = kept_positions.astype(np.float32)
        previous = None
        for pos in kept_positions:
            delta_t[idx, pos] = 1.0 if previous is None else float(pos - previous)
            previous = pos

    return {
        "x": torch.from_numpy(x),
        "mask": torch.from_numpy(mask),
        "delta_t": torch.from_numpy(delta_t),
        "time": torch.from_numpy(time),
        "label": torch.from_numpy(label),
        "split": list(splits),
        "meta": {
            "dataset_name": "TEP",
            "view": "nonuniform",
            "seed": seed,
            "n_samples": len(samples),
            "max_length": max_t,
            "feature_dim": dim,
            "label_meaning": {"0": "normal", "1": "abnormal"},
            "sampling": "Bernoulli thinning over observed uniform grid; labels unchanged.",
            "source_files": sorted({sample.source_path for sample in samples}),
        },
    }


def _write_btd_payloads(
    samples: Sequence[WindowSample],
    processed_output: Path,
    seed: int,
    val_ratio: float,
    test_normal_ratio: float,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    import torch

    processed_output.mkdir(parents=True, exist_ok=True)
    splits = _split_samples(samples, seed=seed, val_ratio=val_ratio, test_normal_ratio=test_normal_ratio)
    uniform = _make_uniform_payload(samples, splits, seed)
    nonuniform = _make_nonuniform_payload(samples, splits, seed)
    uniform_path = processed_output / "tep_uniform_btd.pt"
    nonuniform_path = processed_output / "tep_nonuniform_btd.pt"
    torch.save(uniform, uniform_path)
    torch.save(nonuniform, nonuniform_path)

    split_counts: Dict[str, Dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        labels = [sample.label_name for sample, split in zip(samples, splits) if split == split_name]
        split_counts[split_name] = dict(Counter(labels))
    return {"uniform": str(uniform_path), "nonuniform": str(nonuniform_path)}, split_counts


def _write_manifest(manifest_path: Path, manifest: Dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def convert_tep(args: argparse.Namespace) -> Dict[str, object]:
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    processed_output = Path(args.processed_output)
    manifest_path = Path(args.manifest)

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_btd:
        processed_output.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    dependencies = _dependency_status()
    source_files = _discover_files(input_dir)
    source_lfs = [path for path in source_files if _is_lfs_pointer(path)]
    external_candidates = _discover_external_candidates(input_dir)
    external_lfs = [path for path in external_candidates if _is_lfs_pointer(path)]
    candidate_files = source_files + external_candidates

    skipped_lfs_pointer_files = [{"path": str(path), "scope": "input"} for path in source_lfs]
    source_candidate_lfs_pointer_files = [{"path": str(path), "scope": "source_candidate"} for path in external_lfs]
    skipped_unreadable_files: List[Dict[str, str]] = []
    counters = {"csv_written": Counter(), "csv_skipped_existing": Counter(), "label_rules": Counter()}
    samples: List[WindowSample] = []
    status = "OK"
    readiness = "READY"

    has_real_source = any(path not in source_lfs and path.suffix.lower() in SUPPORTED_READ_EXTENSIONS for path in source_files)
    has_lfs_block = bool(source_lfs or (not source_files and external_lfs))
    has_xlsx_candidate = any(path.suffix.lower() == ".xlsx" for path in candidate_files)

    if has_lfs_block:
        status = "TEP_RAW_BLOCKED_BY_LFS_POINTER"
        readiness = "LFS_POINTER_NOT_REAL_DATA"
        print("TEP_RAW_BLOCKED_BY_LFS_POINTER")
    elif not source_files:
        status = "MISSING_RAW"
        readiness = "MISSING_RAW"

    if has_xlsx_candidate and dependencies["openpyxl"] == "missing":
        print("MISSING_OPENPYXL")
        if status == "OK":
            status = "MISSING_OPENPYXL"
            readiness = "MISSING_DEPENDENCY"
    else:
        print("OPENPYXL_OK" if dependencies["openpyxl"] == "available" else "OPENPYXL_NOT_REQUIRED_OR_MISSING")

    if dependencies["pandas"] == "missing" or dependencies["numpy"] == "missing":
        missing = [name for name in ("pandas", "numpy") if dependencies[name] == "missing"]
        status = f"MISSING_DEPENDENCY:{','.join(missing)}"
        readiness = "MISSING_DEPENDENCY"
    if any(path.suffix.lower() == ".mat" for path in source_files) and dependencies["scipy"] == "missing":
        if status == "OK":
            status = "MISSING_SCIPY_FOR_MAT"
            readiness = "MISSING_DEPENDENCY"

    can_process = (
        not args.check_only
        and has_real_source
        and dependencies["pandas"] == "available"
        and dependencies["numpy"] == "available"
    )

    if can_process:
        for path in source_files:
            suffix = path.suffix.lower()
            if path in source_lfs:
                continue
            if suffix == ".txt":
                skipped_unreadable_files.append({"path": str(path), "reason": "unsupported_txt"})
                continue
            if suffix == ".xlsx" and dependencies["openpyxl"] == "missing":
                skipped_unreadable_files.append({"path": str(path), "reason": "missing_openpyxl"})
                continue
            if suffix == ".xls" and dependencies["xlrd"] == "missing":
                skipped_unreadable_files.append({"path": str(path), "reason": "missing_xlrd"})
                continue
            if suffix == ".mat" and dependencies["scipy"] == "missing":
                skipped_unreadable_files.append({"path": str(path), "reason": "missing_scipy"})
                continue
            if suffix not in SUPPORTED_READ_EXTENSIONS:
                skipped_unreadable_files.append({"path": str(path), "reason": f"unsupported_extension:{suffix}"})
                continue
            try:
                for sheet_name, frame in _iter_dataframes(path):
                    samples.extend(
                        _process_dataframe(
                            frame=frame,
                            source_path=path,
                            sheet_name=sheet_name,
                            output_dir=output_dir,
                            window_size=args.window_size,
                            stride=args.stride,
                            min_window_size=args.min_window_size,
                            overwrite=args.overwrite,
                            counters=counters,
                            skipped_unreadable_files=skipped_unreadable_files,
                        )
                    )
            except Exception as exc:
                skipped_unreadable_files.append({"path": str(path), "reason": f"read_error:{type(exc).__name__}:{exc}"})

    btd_payload_paths: Dict[str, str] = {}
    split_counts: Dict[str, Dict[str, int]] = {}
    if args.write_btd and not args.check_only and samples:
        if dependencies["torch"] == "available":
            btd_payload_paths, split_counts = _write_btd_payloads(
                samples,
                processed_output=processed_output,
                seed=args.seed,
                val_ratio=args.val_ratio,
                test_normal_ratio=args.test_normal_ratio,
            )
        else:
            skipped_unreadable_files.append({"path": str(processed_output), "reason": "missing_torch_for_btd"})
    elif args.write_btd:
        print("BTD_NOT_GENERATED")

    output_scan = _scan_output_csvs(output_dir)
    manifest = {
        "dataset_name": "TEP",
        "status": status,
        "raw_readiness": readiness,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "processed_output": str(processed_output),
        "source_files": [str(path) for path in source_files],
        "source_candidate_paths": [str(path) for path in external_candidates],
        "skipped_lfs_pointer_files": skipped_lfs_pointer_files,
        "source_candidate_lfs_pointer_files": source_candidate_lfs_pointer_files,
        "n_skipped_lfs_pointer_files": len(skipped_lfs_pointer_files),
        "n_source_candidate_lfs_pointer_files": len(source_candidate_lfs_pointer_files),
        "skipped_unreadable_files": skipped_unreadable_files,
        "n_normal_csv": output_scan["n_normal_csv"],
        "n_abnormal_csv": output_scan["n_abnormal_csv"],
        "n_csv_written_this_run": int(sum(counters["csv_written"].values())),
        "n_csv_skipped_existing_this_run": int(sum(counters["csv_skipped_existing"].values())),
        "feature_dim_distribution": output_scan["feature_dim_distribution"],
        "length_distribution": output_scan["length_distribution"],
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "min_window_size": int(args.min_window_size),
        "label_rule": {
            "column_rule": "fault_id/label numeric values: 0=normal, >0=abnormal",
            "filename_rule": "normal,nofault,d00,fault0 => normal; fault,d01-d21,fault01,idv => abnormal",
            "observed": dict(counters["label_rules"]),
        },
        "btd_payload_paths": btd_payload_paths,
        "split_counts": split_counts,
        "dependency_status": dependencies,
        "check_only": bool(args.check_only),
        "write_btd_requested": bool(args.write_btd),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_normal_ratio": float(args.test_normal_ratio),
        "created_at": datetime.now().isoformat(),
    }
    _write_manifest(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TEP raw files into sample CSVs and optional BTD payloads.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_DIR), help="Input raw directory or third-party TEP package directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output preprocess_data directory.")
    parser.add_argument("--processed-output", default=str(DEFAULT_PROCESSED_DIR), help="Output directory for BTD .pt payloads.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Manifest JSON path.")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--min-window-size", type=int, default=32)
    parser.add_argument("--write-btd", action="store_true", help="Write uniform and nonuniform BTD payloads after real conversion.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-normal-ratio", type=float, default=0.2)
    parser.add_argument("--check-only", action="store_true", help="Discover inputs and dependencies without writing samples or BTD payloads.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSVs.")
    args = parser.parse_args()
    if args.window_size <= 0:
        parser.error("--window-size must be positive")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.min_window_size <= 0:
        parser.error("--min-window-size must be positive")
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be in [0, 1)")
    if not 0 <= args.test_normal_ratio < 1:
        parser.error("--test-normal-ratio must be in [0, 1)")
    if args.val_ratio + args.test_normal_ratio >= 1:
        parser.error("--val-ratio + --test-normal-ratio must be < 1")
    return args


def main() -> None:
    args = parse_args()
    manifest = convert_tep(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "raw_readiness": manifest["raw_readiness"],
                "source_files": len(manifest["source_files"]),
                "source_candidate_paths": len(manifest["source_candidate_paths"]),
                "skipped_lfs_pointer_files": manifest["n_skipped_lfs_pointer_files"],
                "source_candidate_lfs_pointer_files": manifest["n_source_candidate_lfs_pointer_files"],
                "n_normal_csv": manifest["n_normal_csv"],
                "n_abnormal_csv": manifest["n_abnormal_csv"],
                "btd_payload_paths": manifest["btd_payload_paths"],
                "manifest": str(args.manifest),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
