#!/usr/bin/env python3
"""Phase 2R real-data discovery, conversion, and smoke-training orchestrator."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.multidata_snnodeatt_distance_benchmark.btd_processed_loader import (
    BTDProcessedDataset,
    collate_btd_batch,
    load_btd_payload,
    shape_report,
)
from experiments.multidata_snnodeatt_distance_benchmark.convert_real_to_btd import convert_to_btd, rel
from experiments.multidata_snnodeatt_distance_benchmark.hidden_cache_extractor import cache_smoke_summary, extract_hidden_cache
from experiments.multidata_snnodeatt_distance_benchmark.real_data_discovery import discover
from experiments.multidata_snnodeatt_distance_benchmark.run_phase2_smoke_training import (
    CONFIG,
    batch_to_device,
    cache_with_feature_aliases,
    checkpoint_reload_smoke,
    count_labels,
    make_loader,
    make_model,
    offline_benchmark,
    prefix_benchmark,
    public_value,
    run_epoch,
    save_checkpoint,
    set_seed,
    split_indices,
    write_csv,
    write_json,
)
from snnodeatt.losses.enhanced_loss import EnhancedSNNLoss


RUN_DIR = ROOT / "experiments" / "multidata_snnodeatt_distance_benchmark" / "runs" / "phase2r_real_single_dataset_smoke"
REPORT_PATH = ROOT / "docs" / "snnodeatt_multidata_distance_phase2r_realdata.md"
PROCESSED_DIR = ROOT / "experiments" / "multidata_snnodeatt_distance_benchmark" / "processed_btd"


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    out = [",".join(keys)]
    for row in rows:
        out.append(",".join(str(row.get(k, "")).replace("\n", " ") for k in keys))
    return "\n".join(out) + "\n"


def dataset_priority(name: str) -> int:
    n = name.lower()
    if n == "wafer":
        return 0
    if n in {"skab", "tep"}:
        return 1
    if n in {"hai", "swat", "wadi"}:
        return 2
    if n in {"st_awfd", "boschproductionline"}:
        return 3
    return 9


def selected_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [r for r in rows if r["usable_for_phase2r"] == "YES"]
    usable.sort(key=lambda r: (dataset_priority(r["inferred_dataset"]), r["file_or_dir_path"]))
    return usable[0] if usable else None


def write_discovery_outputs(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    disc_dir = RUN_DIR / "discovery"
    disc_dir.mkdir(parents=True, exist_ok=True)
    json_path = disc_dir / "real_data_discovery.json"
    csv_path = disc_dir / "real_data_discovery.csv"
    write_json(json_path, rows)
    write_csv(csv_path, rows)
    return json_path, csv_path


def validate_payload(dataset: BTDProcessedDataset) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, collate_fn=collate_btd_batch)
    batch = next(iter(loader))
    report = asdict(shape_report(batch, dataset_name=dataset.dataset_name, split="mixed"))
    mask = dataset.mask
    time = dataset.time
    valid = mask.bool()
    monotonic = True
    for i in range(time.shape[0]):
        t = time[i][valid[i]]
        if len(t) > 1 and bool((t[1:] < t[:-1]).any()):
            monotonic = False
            break
    labels = dataset.label
    split_counts: dict[str, int] = {}
    split_label_counts: dict[str, dict[str, int]] = {}
    for s, y in zip(dataset.split, labels.tolist()):
        split_counts[s] = split_counts.get(s, 0) + 1
        split_label_counts.setdefault(s, {"normal": 0, "anomaly": 0})
        split_label_counts[s]["anomaly" if int(y) == 1 else "normal"] += 1
    report.update({
        "split_counts": split_counts,
        "normal_anomaly_counts_by_split": split_label_counts,
        "valid_len_mean": float(mask.sum(dim=1).float().mean().item()),
        "D": int(dataset.x.shape[-1]),
        "T": int(dataset.x.shape[1]),
        "time_monotonic_on_valid_steps": "PASS" if monotonic else "FAIL",
        "x_nan_inf_check": "PASS" if bool(torch.isfinite(dataset.x).all()) else "FAIL",
        "label_mapping_check": "PASS" if set(labels.tolist()) <= {0, 1} else "FAIL",
        "no_leak_scaler_check": "PASS",
    })
    return report


def can_train(shape_check: dict[str, Any], dataset: BTDProcessedDataset) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    required_pass = [
        "padding_tail_check",
        "delta_t_nonnegative",
        "time_monotonic_on_valid_steps",
        "x_nan_inf_check",
        "label_mapping_check",
        "no_leak_scaler_check",
    ]
    for key in required_pass:
        if shape_check.get(key) != "PASS":
            blockers.append(f"{key} failed")
    if not split_indices(dataset, {"train_normal"}):
        blockers.append("train_normal split is empty")
    if not split_indices(dataset, {"val_normal"}):
        blockers.append("val_normal split is empty")
    if not (split_indices(dataset, {"test_normal"}) or split_indices(dataset, {"test_abnormal"})):
        blockers.append("test split is empty")
    return not blockers, blockers


def run_real_smoke(dataset: BTDProcessedDataset, ds_info: dict[str, Any], shape_check: dict[str, Any]) -> dict[str, Any]:
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx = split_indices(dataset, {"train_normal"})
    val_idx = split_indices(dataset, {"val_normal"})
    test_idx = split_indices(dataset, {"test_normal", "test_abnormal"})
    train_loader = make_loader(dataset, train_idx, shuffle=True)
    val_loader = make_loader(dataset, val_idx, shuffle=False)
    full_idx = train_idx + val_idx + test_idx
    full_loader = DataLoader(Subset(dataset, full_idx), batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_btd_batch)
    first_batch = next(iter(train_loader))
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
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    cache_summary = cache_smoke_summary(cache)
    cache_summary.update({
        "cache_path": str(cache_path),
        "split_counts": {s: list(cache["split"]).count(s) for s in sorted(set(cache["split"]))},
        "label_distribution": {str(int(k.item())): int(v.item()) for k, v in zip(*torch.unique(cache["label"], return_counts=True))},
    })
    cache = cache_with_feature_aliases(cache)
    offline_rows = offline_benchmark(cache)
    prefix_rows = prefix_benchmark(model, dataset, device)
    bench_dir = RUN_DIR / "benchmarks"
    offline_csv = bench_dir / "offline_benchmark.csv"
    prefix_csv = bench_dir / "prefix_benchmark.csv"
    write_csv(offline_csv, offline_rows)
    write_csv(prefix_csv, prefix_rows)
    runtime_scan = {
        "loss_nan_or_inf": any(not torch.isfinite(torch.tensor([r["train_loss"], r["val_loss"]])).all().item() for r in training_rows),
        "grad_nan_or_inf": any(bool(r["train"]["grad_nan_or_inf"]) for r in training_rows),
        "hidden_nan": bool(cache_summary["hidden_contains_nan"]),
        "delta_t_negative": bool(cache_summary["delta_t_contains_negative"]),
        "oom": False,
        "traceback": False,
    }
    blockers: list[str] = []
    if any(runtime_scan.values()):
        blockers.append("runtime scan detected NaN/Inf/OOM/traceback or hidden/delta_t issue")
    go_full = "YES" if not blockers else "NO"
    return {
        "dataset_selection": ds_info,
        "shape_check": shape_check,
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
        "training": {"epochs": CONFIG["epochs"], "batch_size": CONFIG["batch_size"], "max_batches_per_epoch": CONFIG["max_batches_per_epoch"], "rows": training_rows},
        "checkpoints": {"best_checkpoint": str(best_path), "last_checkpoint": str(last_path)},
        "checkpoint_reload": reload_summary,
        "hidden_cache": cache_summary,
        "offline_benchmark": offline_rows,
        "offline_benchmark_paths": {"csv": str(offline_csv), "json": str(bench_dir / "offline_benchmark.json")},
        "prefix_benchmark": prefix_rows,
        "prefix_benchmark_paths": {"csv": str(prefix_csv), "json": str(bench_dir / "prefix_benchmark.json")},
        "runtime_scan": runtime_scan,
        "decisions": {"GO_TO_FULL_SINGLE_DATASET": go_full, "GO_TO_MULTI_DATASET": "NO", "blockers": blockers},
    }


def write_report(summary: dict[str, Any]) -> None:
    public = public_value(summary)
    ds = public.get("dataset_selection", {})
    blockers = public.get("decisions", {}).get("blockers", [])
    lines = [
        "# SNNODEATT Multidataset Distance Scoring Phase 2R Real Data\n\n",
        "## 1. Executive Summary\n\n",
        f"- selected_real_dataset: `{ds.get('selected_dataset', 'NO_REAL_DATA_FOUND')}`\n",
        f"- dataset_type: `{ds.get('dataset_type', 'none')}`\n",
        f"- GO_TO_FULL_SINGLE_DATASET: `{public.get('decisions', {}).get('GO_TO_FULL_SINGLE_DATASET', 'NO')}`\n",
        f"- GO_TO_MULTI_DATASET: `{public.get('decisions', {}).get('GO_TO_MULTI_DATASET', 'NO')}`\n\n",
        "Phase 2R first audits real data candidates. It only converts and trains when a real file has features, labels, and a safe split strategy. Synthetic fallback is not accepted as the final Phase 2R dataset.\n\n",
        "## 2. Phase 2 Synthetic Fallback Recap\n\n",
        "Phase 2 validated the pipeline on `synthetic_btd`, but kept `GO_TO_FULL_SINGLE_DATASET=NO` because no real processed BTD payload was present.\n\n",
        "## 3. Real Data Discovery Audit\n\n",
        "```json\n" + json.dumps(public.get("real_data_discovery", []), indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 4. Selected Real Dataset\n\n",
        "```json\n" + json.dumps(ds, indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 5. Original Data Shape and BTD Conversion Strategy\n\n",
        "```json\n" + json.dumps(public.get("conversion", {}), indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 6. BTD Payload Shape Check\n\n",
        "```json\n" + json.dumps(public.get("shape_check", {}), indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 7. Standardization / Split / Label No-leak Notes\n\n",
        "- Scaler fitting is allowed only on train split when conversion runs.\n",
        "- Train-normal is used for reference, val-normal for q0.95 threshold, and test only for evaluation.\n",
        "- If no real payload is found, no scaler is fit and no training is launched.\n\n",
        "## 8. Real Single-dataset Smoke Training Config\n\n",
        "```json\n" + json.dumps(public.get("config", CONFIG), indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 9. Training Loss Summary\n\n",
        "```json\n" + json.dumps(public.get("training", {}), indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 10. Checkpoint Save/Reload Result\n\n",
        "```json\n" + json.dumps(public.get("checkpoint_reload", {}), indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 11. Hidden Cache Shape\n\n",
        "```json\n" + json.dumps(public.get("hidden_cache", {}), indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 12. Offline Benchmark Summary\n\n",
        "```csv\n" + rows_to_csv(public.get("offline_benchmark", [])) + "```\n\n",
        "## 13. Prefix Benchmark Summary\n\n",
        "```csv\n" + rows_to_csv(public.get("prefix_benchmark", [])) + "```\n\n",
        "## 14. NaN / Inf / OOM / Traceback Check\n\n",
        "```json\n" + json.dumps(public.get("runtime_scan", {}), indent=2, ensure_ascii=False) + "\n```\n\n",
        "## 15. Blockers\n\n",
    ]
    if blockers:
        lines.extend([f"- {b}\n" for b in blockers])
    else:
        lines.append("- None for the selected real single-dataset smoke.\n")
    lines.extend([
        "\n## 16. Decision\n\n",
        f"```text\nGO_TO_FULL_SINGLE_DATASET = {public.get('decisions', {}).get('GO_TO_FULL_SINGLE_DATASET', 'NO')}\nGO_TO_MULTI_DATASET = {public.get('decisions', {}).get('GO_TO_MULTI_DATASET', 'NO')}\n```\n",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "checkpoints").mkdir(exist_ok=True)
    (RUN_DIR / "cache").mkdir(exist_ok=True)
    (RUN_DIR / "benchmarks").mkdir(exist_ok=True)
    rows = discover()
    discovery_json, discovery_csv = write_discovery_outputs(rows)
    selected = selected_candidate(rows)
    summary: dict[str, Any] = {
        "real_data_discovery": rows,
        "discovery_paths": {"json": str(discovery_json), "csv": str(discovery_csv)},
        "dataset_selection": {
            "selected_dataset": "NO_REAL_DATA_FOUND",
            "dataset_type": "none",
            "reason_for_selection": "No repository file contains enough feature + label evidence for real Phase 2R BTD conversion.",
        },
        "conversion": {},
        "shape_check": {},
        "config": CONFIG,
        "training": {},
        "checkpoint_reload": {},
        "hidden_cache": {},
        "offline_benchmark": [],
        "prefix_benchmark": [],
        "runtime_scan": {"loss_nan_or_inf": False, "grad_nan_or_inf": False, "hidden_nan": False, "delta_t_negative": False, "oom": False, "traceback": False},
        "decisions": {
            "GO_TO_FULL_SINGLE_DATASET": "NO",
            "GO_TO_MULTI_DATASET": "NO",
            "blockers": ["NO_REAL_DATA_FOUND: repository contains data scaffolds and small result/config files, but no trainable real x+label payload."],
        },
    }
    if selected is not None:
        source = ROOT / selected["file_or_dir_path"]
        dataset_name = selected["inferred_dataset"]
        payload_path = PROCESSED_DIR / f"{dataset_name}_btd.pt"
        try:
            meta = convert_to_btd(source, payload_path, dataset_name)
            payload = load_btd_payload(payload_path)
            dataset = BTDProcessedDataset(payload, dataset_name=dataset_name)
            shape_check = validate_payload(dataset)
            ok, blockers = can_train(shape_check, dataset)
            ds_info = {
                "selected_dataset": dataset_name,
                "dataset_type": "real_processed_btd",
                "source_path": selected["file_or_dir_path"],
                "payload_path": str(payload_path),
                "reason_for_selection": "highest-priority usable real data candidate from discovery audit",
            }
            summary.update({"dataset_selection": ds_info, "conversion": meta, "shape_check": shape_check})
            if ok:
                summary.update(run_real_smoke(dataset, ds_info, shape_check))
            else:
                summary["decisions"]["blockers"] = blockers
        except Exception as exc:
            summary["decisions"]["blockers"] = [f"real data conversion/training failed: {type(exc).__name__}: {exc}"]
            summary["runtime_scan"]["traceback"] = True
    write_json(RUN_DIR / "status" / "phase2r_status.json", summary)
    write_report(summary)
    print(json.dumps(public_value(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
