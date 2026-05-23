"""No-leak wrappers around existing SNN-ODE-ATT anomaly score methods."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from snnodeatt.scoring.score_registry import SCORE_REGISTRY
from snnodeatt.utils.mask import normalize_sequence_mask


def _auc(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_auc(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if (y == 1).sum() == 0:
        return np.nan
    order = np.argsort(-s)
    yy = y[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    precision = tp / np.maximum(1, tp + fp)
    return float((precision * (yy == 1)).sum() / max(1, (y == 1).sum()))


def _metrics(y_true, scores, threshold):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pred = (s > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) else 0.0
    return {
        "Accuracy": float((tp + tn) / max(1, len(y))),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "F2": float(f2),
        "AUC": _auc(y, s),
        "PR_AUC": _pr_auc(y, s),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def split_masks(cache: dict[str, Any]) -> dict[str, torch.Tensor]:
    split = [str(s) for s in cache["split"]]
    labels = torch.as_tensor(cache["label"]).long()
    train_normal = torch.tensor([(s == "train_normal") and int(labels[i]) == 0 for i, s in enumerate(split)], dtype=torch.bool)
    val_normal = torch.tensor([(s == "val_normal") and int(labels[i]) == 0 for i, s in enumerate(split)], dtype=torch.bool)
    test = torch.tensor([s in {"test", "test_normal", "test_abnormal"} for s in split], dtype=torch.bool)
    return {"train_normal": train_normal, "val_normal": val_normal, "test": test}


def make_scoring_cache(cache: dict[str, Any], feature_key: str = "mem_reset_seq") -> dict[str, Any]:
    feature = torch.as_tensor(cache[feature_key]).float()
    if feature.dim() == 2:
        feature = feature.unsqueeze(1)
        mask = torch.ones(feature.shape[:2], dtype=torch.float32)
        delta_t = torch.ones_like(mask)
        tau = torch.zeros_like(mask)
    else:
        mask = normalize_sequence_mask(cache["mask"], dtype=torch.float32)
        delta_t = torch.as_tensor(cache["delta_t"]).float()
        tau = torch.as_tensor(cache.get("time", cache.get("tau", torch.cumsum(delta_t, dim=1)))).float()
    return {
        "m_reset": feature,
        "mask": mask,
        "delta_t": delta_t,
        "tau": tau,
        "label": torch.as_tensor(cache["label"]).long(),
        "split": list(cache["split"]),
        "sample_id": list(cache.get("sample_id", [str(i) for i in range(len(cache["label"]))])),
    }


def run_no_leak_distance_benchmark(
    cache: dict[str, Any],
    *,
    methods: list[str] | None = None,
    feature_key: str = "mem_reset_seq",
    threshold_quantile: float = 0.95,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fit on train normal, calibrate on val normal, evaluate on test only."""

    config = dict(config or {})
    config.setdefault("device", "cpu")
    score_cache = make_scoring_cache(cache, feature_key)
    masks = split_masks(score_cache)
    train_mask = masks["train_normal"]
    val_mask = masks["val_normal"]
    test_mask = masks["test"]
    if not bool(train_mask.any()):
        raise ValueError("No train_normal samples available for reference fitting.")
    if not bool(val_mask.any()):
        raise ValueError("No val_normal samples available for threshold calibration.")
    if not bool(test_mask.any()):
        raise ValueError("No test samples available for evaluation.")

    labels_test = torch.as_tensor(score_cache["label"])[test_mask].cpu().numpy()
    rows: list[dict[str, Any]] = []
    for method in methods or list(SCORE_REGISTRY):
        fn = SCORE_REGISTRY[method]
        val_scores = fn(score_cache, train_mask, val_mask, config).numpy()
        test_scores = fn(score_cache, train_mask, test_mask, config).numpy()
        threshold = float(np.quantile(val_scores, threshold_quantile))
        rows.append({
            "method": method,
            "feature_key": feature_key,
            "threshold_source": "val_normal",
            "reference_source": "train_normal",
            "threshold_quantile": threshold_quantile,
            "threshold": threshold,
            **_metrics(labels_test, test_scores, threshold),
        })
    return rows
