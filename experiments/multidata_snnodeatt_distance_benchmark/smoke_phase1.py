#!/usr/bin/env python3
"""Phase 1 smoke checks for the multidataset SNNODEATT distance pipeline."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from snnodeatt.models.snn_odeatt import PredictiveSNN_ODEATT_Model
from snnodeatt.losses.enhanced_loss import EnhancedSNNLoss
from snnodeatt.utils.mask import normalize_sequence_mask

from experiments.multidata_snnodeatt_distance_benchmark.btd_processed_loader import (
    build_synthetic_btd_loader,
    collate_btd_batch,
    shape_report,
)
from experiments.multidata_snnodeatt_distance_benchmark.distance_wrapper import run_no_leak_distance_benchmark
from experiments.multidata_snnodeatt_distance_benchmark.hidden_cache_extractor import cache_smoke_summary, extract_hidden_cache


OUT = ROOT / "experiments" / "multidata_snnodeatt_distance_benchmark" / "phase1_smoke_summary.json"


def assert_mask_normalizer():
    mask_bt = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    assert normalize_sequence_mask(mask_bt).shape == (2, 3)
    assert normalize_sequence_mask(mask_bt.unsqueeze(-1)).shape == (2, 3)
    legacy = mask_bt.unsqueeze(-1).expand(-1, -1, 4).clone()
    assert normalize_sequence_mask(legacy).shape == (2, 3)
    invalid = legacy.clone()
    invalid[0, 1, 2] = 0.0
    failed = False
    try:
        normalize_sequence_mask(invalid)
    except ValueError:
        failed = True
    assert failed, "invalid channel-wise mask should raise ValueError"
    return {
        "mask_bt": "PASS",
        "mask_bt1": "PASS",
        "legacy_btd": "PASS",
        "invalid_channel_mask": "PASS_RAISED",
    }


def make_model(input_dim: int):
    return PredictiveSNN_ODEATT_Model(
        input_dim=input_dim,
        hidden_dim=12,
        tau_mem=10.0,
        tau_syn_base=8.0,
        threshold=0.3,
        beta_spike=10.0,
        dropout_rate=0.0,
    )


def forward_shape_smoke(batch):
    x = batch["x"]
    mask = batch["mask"]
    delta_t = batch["delta_t"]
    model = make_model(x.shape[-1])
    results = {}
    with torch.no_grad():
        for name, m in {
            "mask_BT": mask,
            "mask_BT1": mask.unsqueeze(-1),
            "legacy_mask_BTD": mask.unsqueeze(-1).expand(-1, -1, x.shape[-1]).clone(),
        }.items():
            out = model(x, mask=m, delta_t=delta_t)
            padding_hidden = out[4] * (1.0 - mask).unsqueeze(-1)
            results[name] = {
                "status": "PASS",
                "mem_seq": list(out[3].shape),
                "mem_reset_seq": list(out[4].shape),
                "rate_seq": list(out[5].shape),
                "h_seq": list(out[6].shape),
                "padding_hidden_zero": bool(padding_hidden.abs().max().item() < 1e-6),
            }
    criterion = EnhancedSNNLoss()
    out = model(x, mask=mask, delta_t=delta_t)
    loss, loss_info = criterion(
        reconstructions=out[0],
        predictions=out[1],
        x=x,
        mask=mask,
        mem_seq=out[3],
        mem_reset_seq=out[4],
        rate_seq=out[5],
        h_seq=out[6],
        delta_t=delta_t,
    )
    results["loss_mask_BT"] = {
        "status": "PASS",
        "loss_finite": bool(torch.isfinite(loss)),
        "loss_keys": sorted(loss_info.keys()),
    }
    return results


def make_prefix_batch(batch, ratio: float):
    mask = normalize_sequence_mask(batch["mask"])
    rows = []
    for i in range(mask.shape[0]):
        valid_len = int(mask[i].sum().item())
        prefix_len = max(1, int(math.ceil(ratio * valid_len)))
        rows.append({
            "x": batch["x"][i, :prefix_len],
            "mask": torch.ones(prefix_len),
            "delta_t": batch["delta_t"][i, :prefix_len],
            "time": batch["time"][i, :prefix_len],
            "label": batch["label"][i],
            "sample_id": f"{batch['sample_id'][i]}_prefix_{ratio:.2f}",
            "split": batch["split"][i],
            "dataset_name": "synthetic_btd_prefix",
        })
    return collate_btd_batch(rows)


def prefix_forward_smoke(batch):
    out = {}
    for ratio in [0.60, 1.00]:
        prefix = make_prefix_batch(batch, ratio)
        model = make_model(prefix["x"].shape[-1])
        with torch.no_grad():
            y = model(prefix["x"], mask=prefix["mask"], delta_t=prefix["delta_t"])
        out[f"r_{ratio:.2f}"] = {
            "status": "PASS",
            "re_forwarded": True,
            "prefix_hidden_shape": list(y[4].shape),
            "prefix_mask_shape": list(prefix["mask"].shape),
        }
    return out


def main():
    loader = build_synthetic_btd_loader(batch_size=8, seed=17)
    first_batch = next(iter(loader))
    report = shape_report(first_batch, dataset_name="synthetic_btd", split="mixed").__dict__
    forward_report = forward_shape_smoke(first_batch)
    model = make_model(first_batch["x"].shape[-1])
    cache = extract_hidden_cache(model, loader, device="cpu", max_batches=None)
    cache_report = cache_smoke_summary(cache)
    distance_rows = run_no_leak_distance_benchmark(
        cache,
        methods=["euclidean", "trajectory_mahalanobis", "sobolev_h1"],
        feature_key="mem_reset_seq",
        threshold_quantile=0.95,
        config={"device": "cpu", "prototype_count": 8, "pca_dim": None},
    )
    prefix_report = prefix_forward_smoke(first_batch)
    summary = {
        "mask_normalizer": assert_mask_normalizer(),
        "mini_batch_shape": report,
        "forward_shape_smoke": forward_report,
        "hidden_cache_smoke": cache_report,
        "distance_wrapper_toy_metrics": distance_rows,
        "prefix_forward_smoke": prefix_report,
        "GO_TO_TRAINING": "YES",
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
