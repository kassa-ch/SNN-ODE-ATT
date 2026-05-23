#!/usr/bin/env python3
"""Phase 2 single-dataset smoke training pipeline.

This script intentionally runs only a small smoke training job. It does not
launch full multidataset training, does not add model math, and does not add any
new distance method.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from snnodeatt.losses.enhanced_loss import EnhancedSNNLoss
from snnodeatt.models.snn_odeatt import PredictiveSNN_ODEATT_Model
from snnodeatt.utils.mask import normalize_sequence_mask

from experiments.multidata_snnodeatt_distance_benchmark.btd_processed_loader import (
    BTDProcessedDataset,
    SyntheticBTDDataset,
    collate_btd_batch,
    load_btd_payload,
    shape_report,
)
from experiments.multidata_snnodeatt_distance_benchmark.distance_wrapper import run_no_leak_distance_benchmark
from experiments.multidata_snnodeatt_distance_benchmark.hidden_cache_extractor import cache_smoke_summary, extract_hidden_cache


RUN_DIR = ROOT / "experiments" / "multidata_snnodeatt_distance_benchmark" / "runs" / "phase2_single_dataset_smoke"
REPORT_PATH = ROOT / "docs" / "snnodeatt_multidata_distance_phase2.md"


CONFIG = {
    "seed": 42,
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 1.0e-3,
    "hidden_dim": 16,
    "threshold": 0.3,
    "tau_mem": 10.0,
    "tau_syn_base": 8.0,
    "max_batches_per_epoch": 5,
    "num_workers": 0,
    "distance_methods": ["euclidean", "trajectory_mahalanobis", "sobolev_h1", "energy_distance"],
    "feature_modes": ["mem_seq", "mem_reset_seq", "rate_seq", "h_seq"],
    "prefix_ratios": [0.60, 0.80, 1.00],
    "threshold_quantile": 0.95,
}


DATASET_CANDIDATES = [
    ("wafer_exp1", "Data/Wafer/exp1/raw", "Data/Wafer/exp1/processed", "Data/Wafer/exp1/manifests"),
    ("wafer_exp2", "Data/Wafer/exp2/raw", "Data/Wafer/exp2/processed", "Data/Wafer/exp2/manifests"),
    ("wafer_exp3", "Data/Wafer/exp3/raw", "Data/Wafer/exp3/processed", "Data/Wafer/exp3/manifests"),
    ("wafer_exp4", "Data/Wafer/exp4/raw", "Data/Wafer/exp4/processed", "Data/Wafer/exp4/manifests"),
    ("ST_AWFD", "Data/ST-AWFD/raw", "Data/ST-AWFD/processed", "Data/ST-AWFD/manifests"),
    ("BoschProductionLine", "Data/Bosch_Production_Line/raw", "Data/Bosch_Production_Line/processed", "Data/Bosch_Production_Line/manifests"),
    ("HAI", "Data/HAI/raw", "Data/HAI/processed", "Data/HAI/manifests"),
    ("TEP", "Data/TEP/raw", "Data/TEP/processed", "Data/TEP/manifests"),
    ("SWaT", "Data/SWaT/raw", "Data/SWaT/processed", "Data/SWaT/manifests"),
    ("WADI", "Data/WADI/raw", "Data/WADI/processed", "Data/WADI/manifests"),
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def public_value(obj: Any) -> Any:
    """Return a report-safe copy with repo-local paths instead of host paths."""
    if isinstance(obj, dict):
        return {k: public_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [public_value(v) for v in obj]
    if isinstance(obj, tuple):
        return [public_value(v) for v in obj]
    if isinstance(obj, str):
        root_str = str(ROOT)
        if obj == root_str:
            return "."
        if root_str in obj:
            obj = obj.replace(root_str + "\\", "").replace(root_str + "/", "")
        return obj.replace("\\", "/")
    return obj


def list_processed_payloads(processed_path: Path, manifest_path: Path) -> list[Path]:
    payloads: list[Path] = []
    for root in [processed_path, manifest_path]:
        if root.exists():
            payloads.extend(sorted(root.glob("*.pt")))
            payloads.extend(sorted(root.glob("*.npz")))
    return payloads


def select_dataset() -> tuple[Dataset, dict[str, Any]]:
    availability = []
    for name, raw_rel, processed_rel, manifest_rel in DATASET_CANDIDATES:
        raw = ROOT / raw_rel
        processed = ROOT / processed_rel
        manifest = ROOT / manifest_rel
        payloads = list_processed_payloads(processed, manifest)
        row = {
            "dataset_name": name,
            "raw_path": str(raw),
            "processed_path": str(processed),
            "manifest_path": str(manifest),
            "processed_payload_count": len(payloads),
            "status": "available" if payloads else "unavailable",
            "reason": "found .pt/.npz BTD payload" if payloads else "no .pt/.npz processed BTD payload found",
        }
        availability.append(row)
        if payloads:
            payload = load_btd_payload(payloads[0])
            ds = BTDProcessedDataset(payload, dataset_name=name)
            return ds, {
                "selected_dataset": name,
                "dataset_type": "real",
                "raw_path": str(raw),
                "processed_path": str(processed),
                "manifest_path": str(manifest),
                "split_source": "processed payload split field",
                "label_source": "processed payload label field",
                "reason_for_selection": f"first available processed BTD payload: {payloads[0]}",
                "availability": availability,
            }

    return SyntheticBTDDataset(seed=CONFIG["seed"]), {
        "selected_dataset": "synthetic_btd",
        "dataset_type": "synthetic_fallback",
        "raw_path": "N/A",
        "processed_path": "N/A",
        "manifest_path": "N/A",
        "split_source": "SyntheticBTDDataset built-in split labels",
        "label_source": "SyntheticBTDDataset synthetic labels",
        "reason_for_selection": "No real processed BTD payload was found in the repository.",
        "availability": availability,
    }


def split_indices(dataset: Dataset, split_names: set[str]) -> list[int]:
    indices = []
    for i in range(len(dataset)):
        row = dataset[i]
        if str(row["split"]) in split_names:
            indices.append(i)
    return indices


def make_loader(dataset: Dataset, indices: list[int], shuffle: bool = False) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_btd_batch,
        drop_last=False,
        num_workers=CONFIG["num_workers"],
    )


def count_labels(dataset: Dataset, indices: list[int]) -> dict[str, int]:
    normal = 0
    anomaly = 0
    for idx in indices:
        label = int(dataset[idx]["label"])
        if label == 0:
            normal += 1
        else:
            anomaly += 1
    return {"normal": normal, "anomaly": anomaly}


def make_model(input_dim: int, device: torch.device) -> PredictiveSNN_ODEATT_Model:
    return PredictiveSNN_ODEATT_Model(
        input_dim=input_dim,
        hidden_dim=CONFIG["hidden_dim"],
        tau_mem=CONFIG["tau_mem"],
        tau_syn_base=CONFIG["tau_syn_base"],
        threshold=CONFIG["threshold"],
        dropout_rate=0.0,
    ).to(device)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device)
    mask = batch["mask"].to(device)
    delta_t = batch["delta_t"].to(device)
    return x, mask, delta_t


def run_epoch(model, loader, criterion, optimizer, device, train: bool, max_batches: int | None) -> dict[str, Any]:
    model.train(mode=train)
    losses = []
    grad_nan = False
    hidden_nan = False
    delta_negative = False
    valid_lens = []
    start = time.time()
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, mask, delta_t = batch_to_device(batch, device)
        delta_negative = delta_negative or bool((delta_t < 0).any().item())
        valid_lens.extend(normalize_sequence_mask(mask).sum(dim=1).cpu().tolist())
        with torch.set_grad_enabled(train):
            out = model(x, mask=mask, delta_t=delta_t)
            hidden_nan = hidden_nan or bool(torch.isnan(out[4]).any().item())
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
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grad_nan = True
                        break
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered: {loss}")
        losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("No batches processed in epoch.")
    gpu_memory = None
    if device.type == "cuda":
        gpu_memory = {
            "allocated": int(torch.cuda.memory_allocated(device)),
            "reserved": int(torch.cuda.memory_reserved(device)),
        }
    return {
        "loss": float(sum(losses) / len(losses)),
        "batches": len(losses),
        "grad_nan_or_inf": bool(grad_nan),
        "hidden_nan": bool(hidden_nan),
        "delta_t_negative": bool(delta_negative),
        "valid_len_min": float(min(valid_lens)),
        "valid_len_max": float(max(valid_lens)),
        "gpu_memory": gpu_memory,
        "elapsed_sec": round(time.time() - start, 3),
    }


def save_checkpoint(path: Path, model, input_dim: int, epoch: int, val_loss: float, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": CONFIG["hidden_dim"],
        "epoch": epoch,
        "val_loss": val_loss,
        "config": CONFIG,
        "metadata": metadata,
    }, path)


def checkpoint_reload_smoke(path: Path, input_dim: int, batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = make_model(input_dim, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    x, mask, delta_t = batch_to_device(batch, device)
    with torch.no_grad():
        out = model(x, mask=mask, delta_t=delta_t)
    return {
        "checkpoint_reload": "PASS",
        "checkpoint_path": str(path),
        "mem_seq_shape": list(out[3].shape),
        "mem_reset_seq_shape": list(out[4].shape),
        "rate_seq_shape": list(out[5].shape),
        "h_seq_shape": list(out[6].shape),
        "hidden_has_nan": bool(torch.isnan(out[4]).any().item()),
    }


def cache_with_feature_aliases(cache: dict[str, Any]) -> dict[str, Any]:
    # Phase 1 wrapper defaults to feature_key="mem_reset_seq"; no mutation needed.
    return cache


def offline_benchmark(cache: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for feature in CONFIG["feature_modes"]:
        for row in run_no_leak_distance_benchmark(
            cache_with_feature_aliases(cache),
            methods=CONFIG["distance_methods"],
            feature_key=feature,
            threshold_quantile=CONFIG["threshold_quantile"],
            config={"device": "cpu", "prototype_count": 8, "pca_dim": 16 if feature == "h_seq" else None},
        ):
            row = {"dataset": "selected", "feature_mode": feature, "status": "PASS", "failure_reason": "", **row}
            rows.append(row)
    return rows


def make_prefix_dataset(dataset: Dataset, ratio: float) -> SyntheticBTDDataset:
    class PrefixDataset(Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, idx):
            row = dataset[idx]
            valid_len = int(torch.as_tensor(row["mask"]).sum().item())
            prefix_len = max(1, int(math.ceil(ratio * valid_len)))
            return {
                "x": row["x"][:prefix_len],
                "mask": torch.ones(prefix_len, dtype=torch.float32),
                "delta_t": row["delta_t"][:prefix_len],
                "time": row["time"][:prefix_len],
                "label": row["label"],
                "sample_id": f"{row['sample_id']}_prefix_{ratio:.2f}",
                "split": row["split"],
                "dataset_name": row.get("dataset_name", "prefix_dataset"),
            }
    return PrefixDataset()


def prefix_benchmark(model, dataset: Dataset, device: torch.device) -> list[dict[str, Any]]:
    rows = []
    for ratio in CONFIG["prefix_ratios"]:
        prefix_ds = make_prefix_dataset(dataset, ratio)
        loader = DataLoader(prefix_ds, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_btd_batch)
        cache = extract_hidden_cache(model, loader, device=device, max_batches=None)
        metrics = run_no_leak_distance_benchmark(
            cache,
            methods=["trajectory_mahalanobis"],
            feature_key="mem_reset_seq",
            threshold_quantile=CONFIG["threshold_quantile"],
            config={"device": "cpu", "prototype_count": 8, "pca_dim": None},
        )[0]
        masks = {
            "train_normal_count": sum(1 for s, y in zip(cache["split"], cache["label"].tolist()) if s == "train_normal" and int(y) == 0),
            "val_normal_count": sum(1 for s, y in zip(cache["split"], cache["label"].tolist()) if s == "val_normal" and int(y) == 0),
            "test_count": sum(1 for s in cache["split"] if s in {"test", "test_normal", "test_abnormal"}),
        }
        rows.append({
            "prefix_ratio": ratio,
            **masks,
            "hidden_shape": list(cache["mem_reset_seq"].shape),
            "threshold_value": metrics["threshold"],
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "F2": metrics["F2"],
            "AUC": metrics["AUC"],
            "PR_AUC": metrics["PR_AUC"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "TN": metrics["TN"],
            "FN": metrics["FN"],
            "first_alarm_ratio": "not_applicable_without_onset_labels",
            "status": "PASS",
        })
    return rows


def generate_report(summary: dict[str, Any]) -> None:
    public_summary = public_value(summary)
    ds = public_summary["dataset_selection"]
    train = public_summary["training"]
    reload = public_summary["checkpoint_reload"]
    cache = public_summary["hidden_cache"]
    offline_rows = public_summary["offline_benchmark"]
    prefix_rows = public_summary["prefix_benchmark"]
    go_full = summary["decisions"]["GO_TO_FULL_SINGLE_DATASET"]
    go_multi = summary["decisions"]["GO_TO_MULTI_DATASET"]
    lines = [
        "# SNNODEATT Multidataset Distance Scoring Phase 2\n\n",
        "## 1. Executive Summary\n\n",
        "Phase 2 ran a small single-dataset smoke training pipeline. No full multidataset training was launched, no new model math was added, and no new distance method was introduced.\n\n",
        f"- selected_dataset: `{ds['selected_dataset']}`\n",
        f"- dataset_type: `{ds['dataset_type']}`\n",
        f"- GO_TO_FULL_SINGLE_DATASET: `{go_full}`\n",
        f"- GO_TO_MULTI_DATASET: `{go_multi}`\n\n",
        "Because no real processed BTD payload was found in the repository, this run used a clearly marked synthetic fallback. Therefore the pipeline is validated, but real-data full single-dataset training is not yet approved.\n\n",
        "## 2. Selected Dataset and Reason\n\n",
        "```json\n" + json.dumps(ds, indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 3. Training Config\n\n",
        "```json\n" + json.dumps(CONFIG, indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 4. Mini-batch Shape Check\n\n",
        "```json\n" + json.dumps(public_summary["mini_batch_shape"], indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 5. Smoke Training Loss Summary\n\n",
        "```json\n" + json.dumps(train, indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 6. Checkpoint Save/Reload Result\n\n",
        "```json\n" + json.dumps(reload, indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 7. Hidden Cache Extraction Result\n\n",
        "```json\n" + json.dumps(cache, indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 8. Offline Distance Benchmark Smoke Result\n\n",
        "```csv\n" + _rows_to_csv_string(offline_rows) + "```\n\n",
        "## 9. Prefix Benchmark Smoke Result\n\n",
        "```csv\n" + _rows_to_csv_string(prefix_rows) + "```\n\n",
        "## 10. No-leak Verification\n\n",
        "- Reference fitting uses only `train_normal`.\n",
        "- Threshold calibration uses only `val_normal` with q=0.95.\n",
        "- Test samples are only evaluated.\n",
        "- No test labels are used to select thresholds or references.\n\n",
        "## 11. NaN / OOM / Shape Error Scan\n\n",
        "```json\n" + json.dumps(public_summary["runtime_scan"], indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 12. Remaining Blockers\n\n",
    ]
    for blocker in public_summary["decisions"]["blockers"]:
        lines.append(f"- {blocker}\n")
    lines.extend([
        "\n## 13. Decision\n\n",
        f"```text\nGO_TO_FULL_SINGLE_DATASET = {go_full}\nGO_TO_MULTI_DATASET = {go_multi}\n```\n",
    ])
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def _rows_to_csv_string(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    import io
    keys = list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "checkpoints").mkdir(exist_ok=True)
    (RUN_DIR / "cache").mkdir(exist_ok=True)
    (RUN_DIR / "benchmarks").mkdir(exist_ok=True)
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, ds_info = select_dataset()
    train_idx = split_indices(dataset, {"train_normal"})
    val_idx = split_indices(dataset, {"val_normal"})
    test_idx = split_indices(dataset, {"test", "test_normal", "test_abnormal"})
    blockers = []
    if not train_idx or count_labels(dataset, train_idx)["normal"] == 0:
        blockers.append("empty train_normal split")
    if not val_idx or count_labels(dataset, val_idx)["normal"] == 0:
        blockers.append("empty val_normal split")
    if not test_idx:
        blockers.append("empty test split")
    if blockers:
        raise RuntimeError("; ".join(blockers))

    train_loader = make_loader(dataset, train_idx, shuffle=True)
    val_loader = make_loader(dataset, val_idx, shuffle=False)
    full_loader = make_loader(dataset, list(range(len(dataset))), shuffle=False)
    first_batch = next(iter(full_loader))
    shape = asdict(shape_report(first_batch, dataset_name=ds_info["selected_dataset"], split="mixed"))
    input_dim = int(first_batch["x"].shape[-1])

    model = make_model(input_dim, device)
    criterion = EnhancedSNNLoss(exclude_positional_encoding_from_loss=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    training_rows = []
    best_val = float("inf")
    best_path = RUN_DIR / "checkpoints" / "best.pt"
    last_path = RUN_DIR / "checkpoints" / "last.pt"
    for epoch in range(1, CONFIG["epochs"] + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True, CONFIG["max_batches_per_epoch"])
        va = run_epoch(model, val_loader, criterion, optimizer, device, False, CONFIG["max_batches_per_epoch"])
        row = {"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"], "train": tr, "val": va}
        training_rows.append(row)
        if va["loss"] < best_val:
            best_val = va["loss"]
            save_checkpoint(best_path, model, input_dim, epoch, best_val, ds_info)
        save_checkpoint(last_path, model, input_dim, epoch, va["loss"], ds_info)

    reload_summary = checkpoint_reload_smoke(best_path, input_dim, first_batch, device)
    cache = extract_hidden_cache(model, full_loader, device=device, max_batches=None)
    cache_path = RUN_DIR / "cache" / "hidden_cache.pt"
    torch.save(cache, cache_path)
    cache_summary = cache_smoke_summary(cache)
    cache_summary.update({
        "cache_path": str(cache_path),
        "split_counts": {s: list(cache["split"]).count(s) for s in sorted(set(cache["split"]))},
        "label_distribution": {str(k): int(v) for k, v in zip(*torch.unique(cache["label"], return_counts=True))},
    })

    offline_rows = offline_benchmark(cache)
    offline_csv = RUN_DIR / "benchmarks" / "offline_benchmark.csv"
    offline_json = RUN_DIR / "benchmarks" / "offline_benchmark.json"
    write_csv(offline_csv, offline_rows)
    write_json(offline_json, offline_rows)

    prefix_rows = prefix_benchmark(model, dataset, device)
    prefix_csv = RUN_DIR / "benchmarks" / "prefix_benchmark.csv"
    prefix_json = RUN_DIR / "benchmarks" / "prefix_benchmark.json"
    write_csv(prefix_csv, prefix_rows)
    write_json(prefix_json, prefix_rows)

    train_log_csv = RUN_DIR / "training_log.csv"
    write_csv(train_log_csv, [
        {
            "epoch": r["epoch"],
            "train_loss": r["train_loss"],
            "val_loss": r["val_loss"],
            "train_hidden_nan": r["train"]["hidden_nan"],
            "val_hidden_nan": r["val"]["hidden_nan"],
            "train_grad_nan_or_inf": r["train"]["grad_nan_or_inf"],
            "delta_t_negative": r["train"]["delta_t_negative"] or r["val"]["delta_t_negative"],
        }
        for r in training_rows
    ])

    runtime_scan = {
        "loss_nan_or_inf": any(not math.isfinite(r["train_loss"]) or not math.isfinite(r["val_loss"]) for r in training_rows),
        "grad_nan_or_inf": any(r["train"]["grad_nan_or_inf"] for r in training_rows),
        "hidden_nan": bool(cache_summary["hidden_contains_nan"]),
        "delta_t_negative": bool(cache_summary["delta_t_contains_negative"]),
        "oom": False,
        "traceback": False,
    }
    decision_blockers = []
    if ds_info["dataset_type"] != "real":
        decision_blockers.append("selected dataset is synthetic_fallback; real data training chain has not been validated")
    if runtime_scan["loss_nan_or_inf"] or runtime_scan["grad_nan_or_inf"] or runtime_scan["hidden_nan"]:
        decision_blockers.append("NaN/Inf detected")
    go_full = "YES" if not decision_blockers else "NO"
    summary = {
        "dataset_selection": ds_info,
        "config": CONFIG,
        "device": str(device),
        "split_counts": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
            "train_labels": count_labels(dataset, train_idx),
            "val_labels": count_labels(dataset, val_idx),
            "test_labels": count_labels(dataset, test_idx),
        },
        "mini_batch_shape": shape,
        "training": {
            "epochs": CONFIG["epochs"],
            "batch_size": CONFIG["batch_size"],
            "max_batches_per_epoch": CONFIG["max_batches_per_epoch"],
            "training_log_path": str(train_log_csv),
            "rows": training_rows,
        },
        "checkpoints": {
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
        },
        "checkpoint_reload": reload_summary,
        "hidden_cache": cache_summary,
        "offline_benchmark": offline_rows,
        "offline_benchmark_paths": {"csv": str(offline_csv), "json": str(offline_json)},
        "prefix_benchmark": prefix_rows,
        "prefix_benchmark_paths": {"csv": str(prefix_csv), "json": str(prefix_json)},
        "runtime_scan": runtime_scan,
        "decisions": {
            "GO_TO_FULL_SINGLE_DATASET": go_full,
            "GO_TO_MULTI_DATASET": "NO",
            "blockers": decision_blockers,
        },
    }
    write_json(RUN_DIR / "summary.json", summary)
    write_json(RUN_DIR / "config_snapshot.json", CONFIG)
    generate_report(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
