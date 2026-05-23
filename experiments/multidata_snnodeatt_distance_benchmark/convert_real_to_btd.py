#!/usr/bin/env python3
"""Convert a real data candidate into the canonical BTD payload format."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch


FEATURE_KEYS = ["x", "X", "data", "features", "sequence", "sequences", "values", "signals"]
LABEL_KEYS = ["label", "labels", "y", "target", "targets", "anomaly", "fault", "class"]
SPLIT_KEYS = ["split", "splits", "set", "phase", "fold"]
TIME_KEYS = ["time", "times", "timestamp", "timestamps", "tau"]
DT_KEYS = ["delta_t", "dt"]
ID_KEYS = ["sample_id", "sample_ids", "id", "ids"]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def first_key(payload: dict[str, Any], names: list[str]) -> str | None:
    lower = {str(k).lower(): k for k in payload.keys()}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_source(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".pkl"}:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            raise ValueError(f"{path} must contain a dict-like payload.")
        return obj
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k].tolist() if data[k].dtype == object else data[k] for k in data.files}
    if suffix == ".csv":
        return load_long_csv(path)
    raise ValueError(f"Unsupported source extension: {suffix}")


def load_long_csv(path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)
    lower = {c.lower(): c for c in fieldnames}
    sample_col = lower.get("sample_id") or lower.get("id")
    time_col = lower.get("time") or lower.get("timestamp") or lower.get("tau")
    label_col = lower.get("label") or lower.get("target") or lower.get("anomaly") or lower.get("fault")
    split_col = lower.get("split") or lower.get("set") or lower.get("phase")
    if sample_col is None or time_col is None or label_col is None:
        raise ValueError("CSV conversion requires sample_id/id, time/timestamp/tau, and label/target/anomaly/fault columns.")
    excluded = {sample_col, time_col, label_col}
    if split_col:
        excluded.add(split_col)
    feature_cols = [c for c in fieldnames if c not in excluded]
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row[sample_col], []).append(row)
    xs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    times: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    sample_ids: list[str] = []
    max_t = max(len(v) for v in groups.values())
    for sid, group in groups.items():
        group = sorted(group, key=lambda r: float(r[time_col]))
        arr = np.asarray([[float(r[c]) for c in feature_cols] for r in group], dtype=np.float32)
        t = np.asarray([float(r[time_col]) for r in group], dtype=np.float32)
        x_pad = np.zeros((max_t, arr.shape[1]), dtype=np.float32)
        mask = np.zeros(max_t, dtype=np.float32)
        time_pad = np.zeros(max_t, dtype=np.float32)
        x_pad[: len(group)] = arr
        mask[: len(group)] = 1.0
        time_pad[: len(group)] = t
        xs.append(x_pad)
        masks.append(mask)
        times.append(time_pad)
        labels.append(map_label(group[-1][label_col]))
        splits.append(str(group[-1].get(split_col, "unknown")) if split_col else "unknown")
        sample_ids.append(sid)
    return {"x": np.stack(xs), "mask": np.stack(masks), "time": np.stack(times), "label": np.asarray(labels), "split": splits, "sample_id": sample_ids}


def map_label(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value != 0)
    if isinstance(value, (float, np.floating)):
        return int(float(value) != 0.0)
    text = str(value).strip().lower()
    if text in {"0", "normal", "false", "no", "benign", "ok"}:
        return 0
    if text in {"1", "anomaly", "abnormal", "true", "yes", "fault", "attack"}:
        return 1
    try:
        return int(float(text) != 0.0)
    except Exception:
        raise ValueError(f"Cannot map label value to binary normal/anomaly: {value!r}")


def to_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype)


def ensure_ntd(x: torch.Tensor, layout: str) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() != 3:
        raise ValueError(f"x must be [N,T,D], [N,D,T], or [N,T], got {tuple(x.shape)}")
    layout = layout.upper()
    if layout == "NTD":
        return x
    if layout == "NDT":
        return x.transpose(1, 2)
    if layout == "AUTO":
        return x
    raise ValueError(f"Unknown layout: {layout}")


def generate_split(label: torch.Tensor, seed: int = 42) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    split = [""] * int(label.numel())
    for cls in [0, 1]:
        idx = torch.where(label == cls)[0]
        if len(idx) == 0:
            continue
        idx = idx[torch.randperm(len(idx), generator=g)]
        n = len(idx)
        n_train = max(1, int(math.floor(n * 0.6))) if cls == 0 else 0
        n_val = max(1, int(math.floor(n * 0.2))) if cls == 0 else 0
        if cls == 0 and n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        for j, i in enumerate(idx.tolist()):
            if cls == 0 and j < n_train:
                split[i] = "train_normal"
            elif cls == 0 and j < n_train + n_val:
                split[i] = "val_normal"
            elif cls == 0:
                split[i] = "test_normal"
            else:
                split[i] = "test_abnormal"
    return split


def normalize_split(split_values: Any, label: torch.Tensor) -> list[str]:
    if split_values is None:
        return generate_split(label)
    out: list[str] = []
    for raw, y in zip(list(split_values), label.tolist()):
        s = str(raw).strip().lower()
        suffix = "abnormal" if int(y) == 1 else "normal"
        if "train" in s:
            base = "train"
        elif "val" in s or "valid" in s:
            base = "val"
        elif "test" in s or "eval" in s:
            base = "test"
        else:
            base = "test" if int(y) == 1 else "train"
        out.append(f"{base}_{suffix}")
    return out


def build_uniform_time(n: int, t: int) -> tuple[torch.Tensor, torch.Tensor]:
    if t <= 1:
        time = torch.zeros(n, t)
    else:
        time = torch.linspace(0.0, 1.0, t).view(1, t).repeat(n, 1)
    delta_t = torch.zeros_like(time)
    if t > 1:
        delta_t[:, 1:] = time[:, 1:] - time[:, :-1]
    return time, delta_t


def standardize_train_only(x: torch.Tensor, mask: torch.Tensor, split: list[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    train_idx = torch.tensor([i for i, s in enumerate(split) if s.startswith("train")], dtype=torch.long)
    if len(train_idx) == 0:
        raise ValueError("Cannot fit scaler: no train split samples.")
    train_x = x[train_idx]
    train_mask = mask[train_idx].bool()
    points = train_x[train_mask]
    mean = points.mean(dim=0)
    std = points.std(dim=0).clamp_min(1.0e-6)
    x_std = (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    return x_std, {"scaler_fit_split": "train", "feature_mean_shape": list(mean.shape), "feature_std_min": float(std.min().item())}


def convert_to_btd(source_path: Path, output_path: Path, dataset_name: str, layout: str = "AUTO", seed: int = 42) -> dict[str, Any]:
    payload = load_source(source_path)
    x_key = first_key(payload, FEATURE_KEYS)
    label_key = first_key(payload, LABEL_KEYS)
    if x_key is None or label_key is None:
        raise ValueError("Source payload must contain feature and label keys.")
    x = ensure_ntd(to_tensor(payload[x_key], torch.float32), layout)
    label = torch.as_tensor([map_label(v) for v in torch.as_tensor(payload[label_key]).flatten().tolist()], dtype=torch.long)
    if label.numel() != x.shape[0]:
        raise ValueError(f"label length {label.numel()} does not match N={x.shape[0]}")
    n, t, d = x.shape
    mask_key = first_key(payload, ["mask", "valid_mask"])
    if mask_key is not None:
        mask = to_tensor(payload[mask_key], torch.float32)
        if mask.dim() == 3:
            mask = mask[:, :, 0]
    else:
        mask = torch.ones(n, t, dtype=torch.float32)
    time_key = first_key(payload, TIME_KEYS)
    dt_key = first_key(payload, DT_KEYS)
    if time_key is not None:
        time = to_tensor(payload[time_key], torch.float32)
        if time.dim() == 3:
            time = time.squeeze(-1)
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0)
    elif dt_key is not None:
        delta_t = to_tensor(payload[dt_key], torch.float32)
        if delta_t.dim() == 3:
            delta_t = delta_t.squeeze(-1)
        time = torch.cumsum(delta_t, dim=1)
    else:
        time, delta_t = build_uniform_time(n, t)
    delta_t = delta_t * mask
    split_key = first_key(payload, SPLIT_KEYS)
    split = normalize_split(payload.get(split_key) if split_key else None, label)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x, scaler_meta = standardize_train_only(x, mask, split)
    id_key = first_key(payload, ID_KEYS)
    sample_id = list(payload[id_key]) if id_key is not None else [f"{dataset_name}_{i:06d}" for i in range(n)]
    meta = {
        "dataset_name": dataset_name,
        "source_path": rel(source_path),
        "original_shape": str(tuple(to_tensor(payload[x_key], torch.float32).shape)),
        "target_shape": [int(n), int(t), int(d)],
        "D": int(d),
        "T": int(t),
        "N": int(n),
        "normal_count": int((label == 0).sum().item()),
        "anomaly_count": int((label == 1).sum().item()),
        "train_count": sum(1 for s in split if s.startswith("train")),
        "val_count": sum(1 for s in split if s.startswith("val")),
        "test_count": sum(1 for s in split if s.startswith("test")),
        "mask_generated_or_loaded": "loaded" if mask_key else "generated_all_ones",
        "delta_t_generated_or_loaded": "loaded_or_derived" if (time_key or dt_key) else "generated_uniform",
        "time_generated_or_loaded": "loaded" if time_key else ("derived_from_delta_t" if dt_key else "generated_uniform"),
        "label_mapping": "normal=0, anomaly=1",
        "split_mapping": "loaded_and_normalized" if split_key else "generated_seeded_stratified",
        "conversion_notes": "BTD converter does not alter model math; scaler fit uses train split only.",
        **scaler_meta,
    }
    out = {
        "x": x.float(),
        "mask": mask.float(),
        "delta_t": delta_t.float(),
        "time": time.float(),
        "label": label.long(),
        "split": split,
        "sample_id": sample_id,
        "meta": meta,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--layout", default="AUTO", choices=["AUTO", "NTD", "NDT"])
    args = parser.parse_args()
    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    output = Path(args.output) if args.output else ROOT / "experiments" / "multidata_snnodeatt_distance_benchmark" / "processed_btd" / f"{args.dataset_name}_btd.pt"
    if not output.is_absolute():
        output = ROOT / output
    meta = convert_to_btd(source, output, args.dataset_name, layout=args.layout)
    print(json.dumps({"output": rel(output), "meta": meta}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
