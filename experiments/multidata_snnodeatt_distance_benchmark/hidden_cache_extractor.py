"""Hidden-cache extraction utilities for SNNODEATT Phase 1 smoke tests."""

from __future__ import annotations

from typing import Any

import torch

from snnodeatt.utils.mask import normalize_sequence_mask


OUTPUT_NAMES = ["recons", "preds", "pooled_latent", "mem_seq", "mem_reset_seq", "rate_seq", "h_seq"]


def forward_to_dict(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor, delta_t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run SNNODEATT and map its legacy tuple output to a named dictionary."""

    outputs = model(x, mask=mask, delta_t=delta_t)
    if not isinstance(outputs, tuple) or len(outputs) != 7:
        raise ValueError(f"Expected 7-tuple SNNODEATT output, got {type(outputs)} len={len(outputs) if isinstance(outputs, tuple) else 'NA'}")
    return dict(zip(OUTPUT_NAMES, outputs))


def _pad_time_tensor(tensor: torch.Tensor, target_t: int) -> torch.Tensor:
    if tensor.shape[1] == target_t:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[1] = target_t - tensor.shape[1]
    pad = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=1)


def extract_hidden_cache(
    model: torch.nn.Module,
    dataloader,
    *,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Extract per-time hidden trajectories into a scoring-ready cache."""

    model = model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = batch["x"].to(device)
            mask = normalize_sequence_mask(batch["mask"], dtype=torch.float32, device=device)
            delta_t = batch["delta_t"].to(device)
            out = forward_to_dict(model, x, mask, delta_t)
            rows.append({
                "mem_seq": out["mem_seq"].detach().cpu(),
                "mem_reset_seq": out["mem_reset_seq"].detach().cpu(),
                "rate_seq": out["rate_seq"].detach().cpu(),
                "h_seq": out["h_seq"].detach().cpu(),
                "pooled_latent": out["pooled_latent"].detach().cpu(),
                "mask": mask.detach().cpu(),
                "delta_t": delta_t.detach().cpu(),
                "time": batch["time"].detach().cpu(),
                "label": batch["label"].detach().cpu(),
                "sample_id": list(batch["sample_id"]),
                "split": list(batch["split"]),
            })

    if not rows:
        raise ValueError("No batches were processed.")

    max_t = max(row["mask"].shape[1] for row in rows)
    cache: dict[str, Any] = {
        "sample_id": [],
        "split": [],
    }
    for key in ["mem_seq", "mem_reset_seq", "rate_seq", "h_seq", "mask", "delta_t", "time"]:
        cache[key] = torch.cat([_pad_time_tensor(row[key], max_t) for row in rows], dim=0)
    cache["pooled_latent"] = torch.cat([row["pooled_latent"] for row in rows], dim=0)
    cache["label"] = torch.cat([row["label"] for row in rows], dim=0)
    for row in rows:
        cache["sample_id"].extend(row["sample_id"])
        cache["split"].extend(row["split"])
    return cache


def cache_smoke_summary(cache: dict[str, Any]) -> dict[str, Any]:
    mask = normalize_sequence_mask(cache["mask"], dtype=torch.float32)
    valid_len = mask.sum(dim=1)
    tensor_shapes = {
        key: tuple(value.shape)
        for key, value in cache.items()
        if torch.is_tensor(value)
    }
    hidden = cache["mem_reset_seq"]
    return {
        "cache_keys": sorted(cache.keys()),
        "tensor_shapes": tensor_shapes,
        "hidden_dtype": str(hidden.dtype),
        "valid_len_min": float(valid_len.min().item()),
        "valid_len_max": float(valid_len.max().item()),
        "hidden_contains_nan": bool(torch.isnan(hidden).any()),
        "delta_t_contains_negative": bool((cache["delta_t"] < 0).any()),
    }
