import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPS_ROOT = PROJECT_ROOT / "Scrips"
PUBLIC_DATASET_ROOT = SCRIPS_ROOT / "data_loader" / "public_datasets"
for path in (PROJECT_ROOT, SCRIPS_ROOT, PUBLIC_DATASET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from public_data_resolver import resolve_public_split
from Distance.distance_factory import create_distance_calculator, available_methods


def _rel(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def _read_checkpoint_exists(path):
    if not path:
        return False
    return Path(path).exists()


def _load_model(model_path, model_name="PredictiveSNN_ODEATT_Model", device="cpu"):
    import torch
    from utils import load_model_checkpoint

    model, _checkpoint = load_model_checkpoint(model_path, model_name, device=device)
    model.eval()
    return model


def _make_loader(paths, batch_size=16, max_items=None, apply_poisson_sampling=False):
    from torch.utils.data import DataLoader
    from data_loader import TimeSeriesDataset, custom_collate

    selected = list(paths)
    if max_items is not None:
        selected = selected[:max_items]
    dataset = TimeSeriesDataset(selected, apply_poisson_sampling=apply_poisson_sampling)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)


def _extract_features(model, loader, device="cpu"):
    import torch
    from utils import safe_model_forward

    features = []
    sequences = []
    masks = []
    times = []

    with torch.no_grad():
        for x, mask, delta_t, _paths in loader:
            x = x.to(device)
            mask = mask.to(device)
            delta_t = delta_t.to(device)
            _recon, _pred, z_mean, z_seq, _aux, _h = safe_model_forward(
                model,
                x,
                mask=mask,
                delta_t=delta_t,
                n_samples=1,
            )
            features.append(z_mean.detach().cpu().numpy())
            sequences.append(z_seq.detach().cpu().numpy())
            masks.append(mask.detach().cpu().numpy().mean(axis=-1))
            times.append(delta_t.detach().cpu().numpy())

    return {
        "features": np.concatenate(features, axis=0),
        "sequences": np.concatenate(sequences, axis=0),
        "mask": np.concatenate(masks, axis=0),
        "time": np.concatenate(times, axis=0),
    }


def _threshold_from_train(scores, percentile=99.0):
    return float(np.percentile(np.asarray(scores, dtype=float), percentile))


def _metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores > threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y_true), 1)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "remove_test_extreme_k": 0,
    }


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_scores(path, normal_scores, abnormal_scores, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(normal_scores, bins=40, alpha=0.65, label="test normal")
    plt.hist(abnormal_scores, bins=40, alpha=0.65, label="test abnormal")
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label="threshold")
    plt.xlabel("Distance score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_check_only(args):
    split = resolve_public_split(
        dataset=args.dataset,
        view=args.view,
        data_dir=args.data_dir,
        seed=args.seed,
        save=True,
    )
    calc = create_distance_calculator(args.distance_method)
    summary = {
        "mode": "check-only",
        "dataset": args.dataset,
        "view": args.view,
        "data_dir": _rel(args.data_dir),
        "distance_method": args.distance_method,
        "distance_calculator": calc.__class__.__name__,
        "available_methods": list(available_methods()),
        "split_counts": split["counts"],
        "split_manifest": split.get("manifest_path"),
        "model_checkpoint_exists": _read_checkpoint_exists(args.checkpoint),
        "remove_test_extreme_k": 0,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_detection(args):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required outside --check-only mode.")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)

    split = resolve_public_split(args.dataset, args.view, args.data_dir, seed=args.seed, save=True)
    train_paths = [PROJECT_ROOT / p for p in split["train_normal_paths"]]
    test_normal_paths = [PROJECT_ROOT / p for p in split["test_normal_paths"]]
    test_abnormal_paths = [PROJECT_ROOT / p for p in split["test_abnormal_paths"]]

    model = _load_model(args.checkpoint, device=args.device)
    train_loader = _make_loader(train_paths, batch_size=args.batch_size, max_items=args.max_train)
    test_normal_loader = _make_loader(test_normal_paths, batch_size=args.batch_size, max_items=args.max_test_normal)
    test_abnormal_loader = _make_loader(test_abnormal_paths, batch_size=args.batch_size, max_items=args.max_test_abnormal)

    train_cache = _extract_features(model, train_loader, device=args.device)
    normal_cache = _extract_features(model, test_normal_loader, device=args.device)
    abnormal_cache = _extract_features(model, test_abnormal_loader, device=args.device)

    calc = create_distance_calculator(args.distance_method)
    calc.fit(
        train_cache["features"],
        train_sequences=train_cache["sequences"],
        mask=train_cache["mask"],
        time=train_cache["time"],
    )
    train_scores = calc.calculate_distance(train_cache["features"], sequences=train_cache["sequences"], mask=train_cache["mask"], time=train_cache["time"])
    normal_scores = calc.calculate_distance(normal_cache["features"], sequences=normal_cache["sequences"], mask=normal_cache["mask"], time=normal_cache["time"])
    abnormal_scores = calc.calculate_distance(abnormal_cache["features"], sequences=abnormal_cache["sequences"], mask=abnormal_cache["mask"], time=abnormal_cache["time"])

    threshold = _threshold_from_train(train_scores)
    scores = np.concatenate([normal_scores, abnormal_scores])
    labels = np.array([0] * len(normal_scores) + [1] * len(abnormal_scores))
    metrics = _metrics(labels, scores, threshold)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / args.output_dir / args.dataset / args.view / args.distance_method / timestamp
    score_rows = [
        {"label": int(label), "score": float(score)}
        for label, score in zip(labels, scores)
    ]
    _write_csv(out_dir / "scores.csv", score_rows, ["label", "score"])
    _write_csv(out_dir / "metrics.csv", [metrics], list(metrics.keys()))
    _plot_scores(out_dir / "score_plot.png", normal_scores, abnormal_scores, threshold)
    print(json.dumps({"output_dir": _rel(out_dir), "metrics": metrics}, indent=2))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["HAI", "ST-AWFD"], required=True)
    parser.add_argument("--view", choices=["uniform", "nonuniform"], default="uniform")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--distance_method", default="euclidean")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_train", type=int, default=64)
    parser.add_argument("--max_test_normal", type=int, default=64)
    parser.add_argument("--max_test_abnormal", type=int, default=64)
    parser.add_argument("--output_dir", default="Results/benchmarks/public_detection")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.check_only:
        run_check_only(args)
    else:
        run_detection(args)


if __name__ == "__main__":
    main()
