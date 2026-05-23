"""Unified BTD processed loader for Phase 1 smoke tests.

The canonical sample format is:
    x:        [T, D]
    mask:     [T]
    delta_t:  [T]
    time:     [T]
    label:    scalar

The collate function pads only at the sequence tail and returns batch tensors:
    x [B,T,D], mask [B,T], delta_t [B,T], time [B,T], label [B].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from snnodeatt.utils.mask import is_tail_padded, normalize_sequence_mask


@dataclass
class BTDShapeReport:
    dataset_name: str
    split: str
    x_shape: tuple[int, ...]
    mask_shape: tuple[int, ...]
    delta_t_shape: tuple[int, ...]
    time_shape: tuple[int, ...]
    label_shape: tuple[int, ...]
    valid_len_min: int
    valid_len_max: int
    padding_tail_check: str
    delta_t_nonnegative: str
    mask_unique_values: list[float]


class BTDProcessedDataset(Dataset):
    """Dataset for processed `.pt`/`.npz` BTD payloads."""

    def __init__(self, payload: dict[str, Any], dataset_name: str = "processed_btd") -> None:
        self.dataset_name = dataset_name
        self.x = torch.as_tensor(payload["x"], dtype=torch.float32)
        self.mask = normalize_sequence_mask(torch.as_tensor(payload["mask"]), dtype=torch.float32)
        self.delta_t = torch.as_tensor(payload["delta_t"], dtype=torch.float32)
        self.time = torch.as_tensor(payload.get("time", payload.get("tau", torch.cumsum(self.delta_t, dim=1))), dtype=torch.float32)
        self.label = torch.as_tensor(payload["label"], dtype=torch.long)
        self.sample_id = list(payload.get("sample_id", [f"{dataset_name}_{i:05d}" for i in range(len(self.label))]))
        self.split = list(payload.get("split", ["unknown"] * len(self.label)))

        if self.x.dim() != 3:
            raise ValueError(f"x must be [N,T,D], got {tuple(self.x.shape)}")
        if self.mask.shape != self.x.shape[:2]:
            raise ValueError(f"mask must be [N,T], got {tuple(self.mask.shape)} for x {tuple(self.x.shape)}")
        if self.delta_t.shape != self.mask.shape:
            raise ValueError("delta_t must match mask shape [N,T].")
        if self.time.shape != self.mask.shape:
            raise ValueError("time/tau must match mask shape [N,T].")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        valid = int(self.mask[idx].sum().item())
        return {
            "x": self.x[idx, :valid],
            "mask": self.mask[idx, :valid],
            "delta_t": self.delta_t[idx, :valid],
            "time": self.time[idx, :valid],
            "label": self.label[idx],
            "sample_id": self.sample_id[idx],
            "split": self.split[idx],
            "dataset_name": self.dataset_name,
        }


class SyntheticBTDDataset(Dataset):
    """Small variable-length BTD dataset for package-safe smoke tests."""

    def __init__(self, n: int = 24, t_max: int = 14, d: int = 5, seed: int = 13) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.rows: list[dict[str, Any]] = []
        splits = ["train_normal"] * 8 + ["val_normal"] * 4 + ["test_normal"] * 6 + ["test_abnormal"] * 6
        for i in range(n):
            split = splits[i % len(splits)]
            label = 1 if split == "test_abnormal" else 0
            length = int(7 + (i * 3) % (t_max - 6))
            delta_t = 0.05 + torch.rand(length, generator=g) * 0.25
            delta_t[0] = 0.0
            time = torch.cumsum(delta_t, dim=0)
            x = 0.15 * torch.randn(length, d, generator=g)
            x = x + torch.sin(time).unsqueeze(-1) * torch.linspace(0.2, 0.8, d)
            if label:
                drift_start = max(1, int(length * 0.55))
                x[drift_start:, : min(2, d)] += torch.linspace(0.4, 1.2, length - drift_start).unsqueeze(-1)
            self.rows.append({
                "x": x.float(),
                "mask": torch.ones(length, dtype=torch.float32),
                "delta_t": delta_t.float(),
                "time": time.float(),
                "label": torch.tensor(label, dtype=torch.long),
                "sample_id": f"synthetic_{i:03d}",
                "split": split,
                "dataset_name": "synthetic_btd",
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate_btd_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_t = max(int(item["x"].shape[0]) for item in batch)
    d = int(batch[0]["x"].shape[-1])
    b = len(batch)
    x = torch.zeros(b, max_t, d, dtype=torch.float32)
    mask = torch.zeros(b, max_t, dtype=torch.float32)
    delta_t = torch.zeros(b, max_t, dtype=torch.float32)
    time = torch.zeros(b, max_t, dtype=torch.float32)
    label = torch.zeros(b, dtype=torch.long)
    sample_id: list[str] = []
    split: list[str] = []
    dataset_names: list[str] = []

    for i, item in enumerate(batch):
        t = int(item["x"].shape[0])
        x[i, :t] = item["x"].float()
        mask[i, :t] = normalize_sequence_mask(item["mask"].view(1, -1), dtype=torch.float32)[0]
        delta_t[i, :t] = item["delta_t"].float()
        time[i, :t] = item["time"].float()
        label[i] = item["label"].long()
        sample_id.append(str(item.get("sample_id", i)))
        split.append(str(item.get("split", "unknown")))
        dataset_names.append(str(item.get("dataset_name", "unknown")))

    return {
        "x": x,
        "mask": mask,
        "delta_t": delta_t,
        "time": time,
        "label": label,
        "sample_id": sample_id,
        "split": split,
        "dataset_name": dataset_names,
    }


def load_btd_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k].tolist() if data[k].dtype == object else data[k] for k in data.files}
    raise ValueError(f"Unsupported BTD payload extension: {path.suffix}")


def build_synthetic_btd_loader(batch_size: int = 6, seed: int = 13) -> DataLoader:
    return DataLoader(
        SyntheticBTDDataset(seed=seed),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_btd_batch,
        drop_last=False,
    )


def shape_report(batch: dict[str, Any], dataset_name: str = "synthetic_btd", split: str = "mixed") -> BTDShapeReport:
    mask = normalize_sequence_mask(batch["mask"], dtype=torch.float32)
    valid_len = mask.sum(dim=1).to(torch.long)
    return BTDShapeReport(
        dataset_name=dataset_name,
        split=split,
        x_shape=tuple(batch["x"].shape),
        mask_shape=tuple(mask.shape),
        delta_t_shape=tuple(batch["delta_t"].shape),
        time_shape=tuple(batch["time"].shape),
        label_shape=tuple(batch["label"].shape),
        valid_len_min=int(valid_len.min().item()),
        valid_len_max=int(valid_len.max().item()),
        padding_tail_check="PASS" if is_tail_padded(mask) else "FAIL",
        delta_t_nonnegative="PASS" if bool((batch["delta_t"] >= 0).all()) else "FAIL",
        mask_unique_values=sorted(float(v) for v in torch.unique(mask).tolist()),
    )
