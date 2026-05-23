#!/usr/bin/env python3
"""Discover real-data candidates for Phase 2R.

The discovery is intentionally conservative: benchmark summaries, configs, and
model outputs are recorded as files but are not treated as trainable BTD
payloads unless they contain enough evidence for features, labels, and usable
split/time metadata.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
except Exception:  # pragma: no cover - optional in some deployments
    np = None

try:
    import torch
except Exception:  # pragma: no cover - optional in some deployments
    torch = None


DATASET_PRIORITY = {
    "wafer": 0,
    "skab": 1,
    "tep": 2,
    "hai": 3,
    "swat": 4,
    "wadi": 5,
    "st_awfd": 6,
    "bosch": 7,
}

SCAN_EXTENSIONS = {
    ".pt",
    ".pth",
    ".pkl",
    ".npz",
    ".npy",
    ".csv",
    ".parquet",
    ".h5",
    ".hdf5",
    ".mat",
    ".json",
    ".yaml",
    ".yml",
}

EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    "runs",
    "checkpoints",
    "cache",
    "logs",
    "outputs",
    "outputs_exp4_local",
}

FEATURE_KEYS = {"x", "X", "data", "features", "sequence", "sequences", "values", "signals"}
LABEL_KEYS = {"label", "labels", "y", "target", "targets", "anomaly", "fault", "class"}
SPLIT_KEYS = {"split", "splits", "set", "phase", "fold"}
TIME_KEYS = {"time", "times", "timestamp", "timestamps", "tau", "delta_t", "dt"}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def infer_dataset(path: Path) -> str:
    text = rel(path).lower()
    if "wafer" in text:
        return "wafer"
    if "skab" in text:
        return "SKAB"
    if "tep" in text:
        return "TEP"
    if "hai" in text:
        return "HAI"
    if "swat" in text:
        return "SWaT"
    if "wadi" in text:
        return "WADI"
    if "st_awfd" in text or "st-awfd" in text:
        return "ST_AWFD"
    if "bosch" in text:
        return "BoschProductionLine"
    return "unknown"


def should_skip(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    if parts & EXCLUDE_PARTS:
        return True
    return path.suffix.lower() not in SCAN_EXTENSIONS


def shape_of(value: Any) -> str:
    if hasattr(value, "shape"):
        return str(tuple(int(x) for x in value.shape))
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return f"list[{len(value)}, ...]"
        return f"list[{len(value)}]"
    return type(value).__name__


def inspect_pt(path: Path, max_load_bytes: int) -> dict[str, Any]:
    if torch is None:
        return {"shape_if_readable": "torch unavailable", "keys": []}
    if path.stat().st_size > max_load_bytes:
        return {"shape_if_readable": "skipped_large_pt", "keys": []}
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        keys = list(obj.keys())
        shapes = {str(k): shape_of(v) for k, v in obj.items() if k in FEATURE_KEYS | LABEL_KEYS | SPLIT_KEYS | TIME_KEYS or hasattr(v, "shape")}
        return {"shape_if_readable": json.dumps(shapes, ensure_ascii=False), "keys": [str(k) for k in keys]}
    return {"shape_if_readable": shape_of(obj), "keys": []}


def inspect_npz(path: Path) -> dict[str, Any]:
    if np is None:
        return {"shape_if_readable": "numpy unavailable", "keys": []}
    data = np.load(path, allow_pickle=False)
    keys = list(data.files)
    shapes = {k: str(tuple(data[k].shape)) for k in keys}
    return {"shape_if_readable": json.dumps(shapes, ensure_ascii=False), "keys": keys}


def inspect_npy(path: Path) -> dict[str, Any]:
    if np is None:
        return {"shape_if_readable": "numpy unavailable", "keys": []}
    arr = np.load(path, mmap_mode="r")
    return {"shape_if_readable": str(tuple(arr.shape)), "keys": []}


def inspect_csv(path: Path, max_rows: int = 100) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = 0
        for _ in reader:
            rows += 1
            if rows >= max_rows:
                break
    return {"shape_if_readable": f"header_cols={len(header)}, preview_rows={rows}", "keys": header}


def inspect_json(path: Path, max_load_bytes: int) -> dict[str, Any]:
    if path.stat().st_size > max_load_bytes:
        return {"shape_if_readable": "skipped_large_json", "keys": []}
    obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(obj, dict):
        return {"shape_if_readable": f"dict_keys={len(obj)}", "keys": list(obj.keys())}
    if isinstance(obj, list):
        return {"shape_if_readable": f"list_len={len(obj)}", "keys": []}
    return {"shape_if_readable": type(obj).__name__, "keys": []}


def inspect_file(path: Path, max_load_bytes: int = 20 * 1024 * 1024) -> dict[str, Any]:
    try:
        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth", ".pkl"}:
            return inspect_pt(path, max_load_bytes)
        if suffix == ".npz":
            return inspect_npz(path)
        if suffix == ".npy":
            return inspect_npy(path)
        if suffix == ".csv":
            return inspect_csv(path)
        if suffix == ".json":
            return inspect_json(path, max_load_bytes)
        if suffix in {".yaml", ".yml"}:
            return {"shape_if_readable": "config_text", "keys": []}
        return {"shape_if_readable": "metadata_only", "keys": []}
    except Exception as exc:
        return {"shape_if_readable": f"read_error: {type(exc).__name__}: {exc}", "keys": []}


def classify(path: Path, inspection: dict[str, Any]) -> dict[str, Any]:
    keys = {str(k) for k in inspection.get("keys", [])}
    lower_keys = {k.lower() for k in keys}
    suffix = path.suffix.lower()
    rel_path = rel(path).lower()
    inferred = infer_dataset(path)
    is_result_like = any(token in rel_path for token in ["benchmark", "diagnostic", "metrics", "score_", "summary", "config"])
    has_feature_key = bool(lower_keys & {k.lower() for k in FEATURE_KEYS})
    has_label = bool(lower_keys & LABEL_KEYS)
    has_split = bool(lower_keys & SPLIT_KEYS)
    has_time = bool(lower_keys & TIME_KEYS)
    csv_maybe_features = suffix == ".csv" and not is_result_like and len(keys) >= 4
    has_x = has_feature_key or suffix in {".npy"} or csv_maybe_features

    usable = has_x and has_label and not is_result_like
    if usable:
        reason = "contains feature-like data and labels; split can be loaded or generated if absent"
    elif is_result_like:
        reason = "result/config/diagnostic file, not a trainable data payload"
    elif not has_x:
        reason = "no feature-like x/data/features evidence"
    elif not has_label:
        reason = "no label evidence"
    else:
        reason = "insufficient BTD payload evidence"

    return {
        "candidate_name": path.stem,
        "file_or_dir_path": rel(path),
        "file_type": suffix.lstrip(".") or "unknown",
        "size": path.stat().st_size,
        "inferred_dataset": inferred,
        "has_x_or_features": "YES" if has_x else "NO",
        "has_label": "YES" if has_label else "NO",
        "has_split": "YES" if has_split else "NO",
        "has_time_or_delta_t": "YES" if has_time else "NO",
        "shape_if_readable": inspection.get("shape_if_readable", "UNKNOWN"),
        "usable_for_phase2r": "YES" if usable else "NO",
        "reason": reason,
    }


def discover(root: Path = ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or should_skip(path):
            continue
        inspection = inspect_file(path)
        rows.append(classify(path, inspection))
    rows.sort(key=lambda r: (r["usable_for_phase2r"] != "YES", DATASET_PRIORITY.get(r["inferred_dataset"].lower(), 99), r["file_or_dir_path"]))
    return rows


def write_csv_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = ROOT / "experiments" / "multidata_snnodeatt_distance_benchmark" / "runs" / "phase2r_real_single_dataset_smoke" / "discovery"
    rows = discover()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "real_data_discovery.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv_report(out_dir / "real_data_discovery.csv", rows)
    selected = next((r for r in rows if r["usable_for_phase2r"] == "YES"), None)
    print(json.dumps({
        "candidate_count": len(rows),
        "usable_count": sum(1 for r in rows if r["usable_for_phase2r"] == "YES"),
        "selected_real_dataset": selected,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
