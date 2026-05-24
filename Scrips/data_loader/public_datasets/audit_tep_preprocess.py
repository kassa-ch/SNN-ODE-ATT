from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPS_DIR = PROJECT_ROOT / "Scrips"
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data" / "TEP" / "preprocess_data"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "Data" / "TEP" / "processed"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "Results" / "manifests" / "tep_preprocess_audit.json"


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


def _read_numeric_csv(path: Path) -> Tuple[np.ndarray, int]:
    skiprows = _detect_skiprows(path)
    arr = np.loadtxt(path, delimiter=",", skiprows=skiprows, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr, skiprows


def _label_from_name(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith("_normal.csv"):
        return "normal"
    if lower.endswith("_abnormal.csv"):
        return "abnormal"
    return "unknown"


def _loader_batch_check(paths: List[Path], batch_size: int) -> Dict[str, object]:
    if not paths:
        return {"status": "SKIPPED_EMPTY"}
    try:
        sys.path.insert(0, str(SCRIPS_DIR))
        from torch.utils.data import DataLoader
        from data_loader import TimeSeriesDataset, custom_collate

        selected = [str(path) for path in paths[:batch_size]]
        dataset = TimeSeriesDataset(selected, apply_poisson_sampling=False)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(selected)), shuffle=False, collate_fn=custom_collate)
        x, mask, delta_t, batch_paths = next(iter(loader))
        return {
            "status": "OK",
            "x_shape": list(x.shape),
            "mask_shape": list(mask.shape),
            "delta_t_shape": list(delta_t.shape),
            "paths": list(batch_paths),
            "note": "Current TimeSeriesDataset appends positional encoding dimensions to x.",
        }
    except Exception as exc:
        return {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}


def _audit_csv_dir(data_dir: Path, batch_size: int) -> Dict[str, object]:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(path for path in data_dir.glob("*.csv") if path.is_file())
    label_counts = Counter(_label_from_name(path) for path in csv_paths)
    length_distribution: Counter = Counter()
    dim_distribution: Counter = Counter()
    invalid_files = []
    nan_inf_files = []
    header_rows = Counter()
    valid_paths: List[Path] = []

    if not csv_paths:
        return {
            "status": "NOT_READY",
            "data_dir": str(data_dir),
            "n_csv": 0,
            "n_normal": 0,
            "n_abnormal": 0,
            "n_unknown": 0,
            "length_distribution": {},
            "feature_dim_distribution": {},
            "header_rows_distribution": {},
            "invalid_files": [],
            "nan_inf_files": [],
            "loader_batch_check": {"status": "SKIPPED_EMPTY"},
        }

    for path in csv_paths:
        try:
            arr, skiprows = _read_numeric_csv(path)
            header_rows[str(skiprows)] += 1
            if arr.ndim != 2:
                invalid_files.append({"path": str(path), "reason": f"not_2d:{arr.ndim}"})
                continue
            if arr.shape[0] <= 0 or arr.shape[1] <= 0:
                invalid_files.append({"path": str(path), "reason": f"empty_shape:{list(arr.shape)}"})
                continue
            if not np.isfinite(arr).all():
                nan_inf_files.append({"path": str(path), "shape": list(arr.shape)})
            length_distribution[str(int(arr.shape[0]))] += 1
            dim_distribution[str(int(arr.shape[1]))] += 1
            valid_paths.append(path)
        except Exception as exc:
            invalid_files.append({"path": str(path), "reason": f"{type(exc).__name__}:{exc}"})

    loader_batch_check = _loader_batch_check(valid_paths, batch_size)
    status = "OK"
    if invalid_files or nan_inf_files or loader_batch_check.get("status") == "ERROR":
        status = "ISSUES_FOUND"

    return {
        "status": status,
        "data_dir": str(data_dir),
        "n_csv": len(csv_paths),
        "n_normal": int(label_counts["normal"]),
        "n_abnormal": int(label_counts["abnormal"]),
        "n_unknown": int(label_counts["unknown"]),
        "length_distribution": dict(sorted(length_distribution.items(), key=lambda item: int(item[0]))),
        "feature_dim_distribution": dict(sorted(dim_distribution.items(), key=lambda item: int(item[0]))),
        "header_rows_distribution": dict(header_rows),
        "invalid_files": invalid_files,
        "nan_inf_files": nan_inf_files,
        "loader_batch_check": loader_batch_check,
    }


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _shape(value) -> Optional[List[int]]:
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return None


def _to_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _audit_payload(path: Path) -> Dict[str, object]:
    required = {"x", "mask", "delta_t", "time", "label", "split", "meta"}
    if not path.exists():
        return {"status": "NOT_READY", "path": str(path), "exists": False}
    try:
        payload = _torch_load(path)
    except Exception as exc:
        return {"status": "ERROR", "path": str(path), "exists": True, "error": f"{type(exc).__name__}:{exc}"}

    missing = sorted(required - set(payload.keys())) if isinstance(payload, dict) else sorted(required)
    if missing:
        return {"status": "ERROR", "path": str(path), "exists": True, "missing_keys": missing}

    x_shape = _shape(payload["x"])
    mask_shape = _shape(payload["mask"])
    delta_shape = _shape(payload["delta_t"])
    time_shape = _shape(payload["time"])
    label_shape = _shape(payload["label"])
    split = list(payload["split"])
    label = _to_list(payload["label"])
    errors = []

    if not x_shape or len(x_shape) != 3:
        errors.append("x_not_NTD")
    if not mask_shape or len(mask_shape) != 2:
        errors.append("mask_not_NT")
    if not delta_shape or len(delta_shape) != 2:
        errors.append("delta_t_not_NT")
    if not time_shape or len(time_shape) != 2:
        errors.append("time_not_NT")
    if x_shape and mask_shape and x_shape[:2] != mask_shape:
        errors.append("x_mask_shape_mismatch")
    if mask_shape and delta_shape and mask_shape != delta_shape:
        errors.append("mask_delta_t_shape_mismatch")
    if mask_shape and time_shape and mask_shape != time_shape:
        errors.append("mask_time_shape_mismatch")
    if x_shape and label_shape and label_shape[0] != x_shape[0]:
        errors.append("label_N_mismatch")
    if x_shape and len(split) != x_shape[0]:
        errors.append("split_N_mismatch")

    split_counts = Counter(split)
    invalid_splits = sorted(set(split) - {"train", "val", "test"})
    if invalid_splits:
        errors.append(f"invalid_split:{invalid_splits}")

    label_counts = Counter(int(item) for item in label)
    abnormal_outside_test = [
        idx for idx, (item_label, item_split) in enumerate(zip(label, split)) if int(item_label) > 0 and item_split != "test"
    ]
    if abnormal_outside_test:
        errors.append("abnormal_outside_test")

    return {
        "status": "OK" if not errors else "ISSUES_FOUND",
        "path": str(path),
        "exists": True,
        "keys": sorted(payload.keys()),
        "x_shape": x_shape,
        "mask_shape": mask_shape,
        "delta_t_shape": delta_shape,
        "time_shape": time_shape,
        "label_shape": label_shape,
        "split_counts": dict(split_counts),
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "abnormal_outside_test_count": len(abnormal_outside_test),
        "meta": payload.get("meta", {}),
        "errors": errors,
    }


def _audit_processed_dir(processed_dir: Path) -> Dict[str, object]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    uniform = _audit_payload(processed_dir / "tep_uniform_btd.pt")
    nonuniform = _audit_payload(processed_dir / "tep_nonuniform_btd.pt")
    statuses = {uniform["status"], nonuniform["status"]}
    if "ERROR" in statuses or "ISSUES_FOUND" in statuses:
        status = "ISSUES_FOUND"
    elif "NOT_READY" in statuses:
        status = "NOT_READY"
    else:
        status = "OK"
    return {
        "status": status,
        "processed_dir": str(processed_dir),
        "uniform": uniform,
        "nonuniform": nonuniform,
    }


def audit_preprocess(args: argparse.Namespace) -> Dict[str, object]:
    data_dir = Path(args.data_dir)
    processed_dir = Path(args.processed_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_audit = _audit_csv_dir(data_dir, args.batch_size)
    btd_audit = _audit_processed_dir(processed_dir)

    if csv_audit["status"] == "NOT_READY" or btd_audit["status"] == "NOT_READY":
        status = "NOT_READY"
    elif csv_audit["status"] == "OK" and btd_audit["status"] == "OK":
        status = "OK"
    else:
        status = "ISSUES_FOUND"

    audit = {
        "dataset_name": "TEP",
        "status": status,
        "csv_preprocess": csv_audit,
        "btd_payloads": btd_audit,
        "check_only": bool(args.check_only),
        "created_at": datetime.now().isoformat(),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)

    if status == "NOT_READY":
        print("NOT_READY")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "n_csv": csv_audit["n_csv"],
                "n_normal": csv_audit["n_normal"],
                "n_abnormal": csv_audit["n_abnormal"],
                "length_distribution": csv_audit["length_distribution"],
                "feature_dim_distribution": csv_audit["feature_dim_distribution"],
                "uniform_btd_status": btd_audit["uniform"]["status"],
                "nonuniform_btd_status": btd_audit["nonuniform"]["status"],
                "audit": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit TEP preprocess_data CSVs and BTD payloads.")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--processed_dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--output", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    audit_preprocess(parse_args())


if __name__ == "__main__":
    main()
