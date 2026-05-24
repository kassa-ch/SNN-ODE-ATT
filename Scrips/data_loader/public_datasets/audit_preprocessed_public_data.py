import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPS_ROOT = PROJECT_ROOT / "Scrips"
if str(SCRIPS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPS_ROOT))

from data_loader import TimeSeriesDataset, custom_collate


DEFAULT_DATASETS = {
    "HAI": PROJECT_ROOT / "Data" / "HAI" / "preprocess_data",
    "ST-AWFD": PROJECT_ROOT / "Data" / "ST-AWFD" / "preprocess_data",
    "TEP": PROJECT_ROOT / "Data" / "TEP" / "preprocess_data",
    "SWaT": PROJECT_ROOT / "Data" / "SWaT" / "preprocess_data",
    "WADI": PROJECT_ROOT / "Data" / "WADI" / "preprocess_data",
    "Bosch": PROJECT_ROOT / "Data" / "Bosch" / "preprocess_data",
}
MANIFEST_PATH = PROJECT_ROOT / "Results" / "manifests" / "public_preprocessed_dataset_audit.json"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def _read_csv_shape_and_finite(path: Path):
    try:
        arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return None, False, f"Expected 2D array, got ndim={arr.ndim}"
        finite = bool(np.isfinite(arr).all())
        return tuple(arr.shape), finite, None
    except Exception as exc:
        return None, False, str(exc)


def _verify_batch_shape(paths, apply_poisson_sampling=False):
    if not paths:
        return None
    from torch.utils.data import DataLoader

    selected = [str(p) for p in paths[:2]]
    dataset = TimeSeriesDataset(selected, apply_poisson_sampling=apply_poisson_sampling)
    loader = DataLoader(dataset, batch_size=len(selected), shuffle=False, collate_fn=custom_collate)
    x, mask, delta_t, file_paths = next(iter(loader))
    return {
        "x": list(x.shape),
        "mask": list(mask.shape),
        "delta_t": list(delta_t.shape),
        "files": [Path(p).name for p in file_paths],
    }


def audit_dataset(name, data_dir, batch_check=True):
    data_dir = Path(data_dir)
    result = {
        "dataset": name,
        "data_dir": _rel(data_dir),
        "status": "missing",
        "normal_count": 0,
        "abnormal_count": 0,
        "total_csv": 0,
        "nan_or_inf_found": False,
        "invalid_files": [],
        "t_distribution": {},
        "d_distribution": {},
        "batch_shape_uniform": None,
        "notes": [],
    }

    if not data_dir.exists():
        result["notes"].append("preprocess_data directory missing")
        return result

    normal = sorted(data_dir.glob("*_normal.csv"))
    abnormal = sorted(data_dir.glob("*_abnormal.csv"))
    files = sorted(data_dir.glob("*.csv"))
    result["normal_count"] = len(normal)
    result["abnormal_count"] = len(abnormal)
    result["total_csv"] = len(files)

    if not files:
        result["status"] = "missing"
        result["notes"].append("no CSV files found")
        return result

    t_counter = Counter()
    d_counter = Counter()
    for path in files:
        shape, finite, error = _read_csv_shape_and_finite(path)
        if shape is None:
            result["invalid_files"].append({"path": _rel(path), "error": error})
            continue
        t_counter[str(shape[0])] += 1
        d_counter[str(shape[1])] += 1
        if not finite:
            result["nan_or_inf_found"] = True
            result["invalid_files"].append({"path": _rel(path), "error": "NaN or Inf found"})

    result["t_distribution"] = dict(sorted(t_counter.items(), key=lambda kv: int(kv[0])))
    result["d_distribution"] = dict(sorted(d_counter.items(), key=lambda kv: int(kv[0])))

    if batch_check and normal:
        try:
            result["batch_shape_uniform"] = _verify_batch_shape(normal)
        except Exception as exc:
            result["invalid_files"].append({"path": data_dir.as_posix(), "error": f"batch shape check failed: {exc}"})

    result["status"] = "passed" if not result["invalid_files"] and normal and abnormal else "failed"
    if not abnormal:
        result["notes"].append("no abnormal CSV files found")
    if not normal:
        result["notes"].append("no normal CSV files found")
    return result


def write_manifest(results, check_only):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "check_only": bool(check_only),
        "shape_contract": "CSV [T,D] -> TimeSeriesDataset adds 2 positional dims -> batch [B,T,D+2]",
        "datasets": results,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS.keys()))
    return parser.parse_args()


def main():
    args = parse_args()
    results = []
    for name in args.datasets:
        data_dir = DEFAULT_DATASETS.get(name)
        if data_dir is None:
            data_dir = PROJECT_ROOT / "Data" / name / "preprocess_data"
        print(f"[AUDIT] {name}: {data_dir}")
        results.append(audit_dataset(name, data_dir))

    manifest = write_manifest(results, check_only=args.check_only)
    print(json.dumps({
        "manifest": _rel(MANIFEST_PATH),
        "datasets": {
            item["dataset"]: {
                "status": item["status"],
                "normal": item["normal_count"],
                "abnormal": item["abnormal_count"],
                "nan_or_inf_found": item["nan_or_inf_found"],
                "batch_shape_uniform": item["batch_shape_uniform"],
            }
            for item in manifest["datasets"]
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
