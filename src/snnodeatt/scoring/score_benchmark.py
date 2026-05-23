"""Unified benchmark runner for hidden-trajectory anomaly scores."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .functional_scores import reference_eval_masks
from .score_registry import SCORE_REGISTRY


def make_toy_cache(seed=7):
    """Synthetic cache: abnormal samples drift in the second half."""
    g = torch.Generator().manual_seed(seed)
    n, t, d = 20, 16, 8
    x = 0.25 * torch.randn(n, t, d, generator=g)
    tau = torch.linspace(0, 1, t).repeat(n, 1)
    delta_t = torch.ones(n, t) / t
    mask = torch.ones(n, t)
    labels = torch.zeros(n, dtype=torch.long)
    splits = ["train_normal"] * 8 + ["val_normal"] * 4 + ["test_normal"] * 4 + ["test_abnormal"] * 4
    labels[-4:] = 1
    x[-4:, t // 2 :, :3] += torch.linspace(0.5, 2.5, t // 2).view(1, -1, 1)
    return {
        "m_reset": x,
        "mask": mask,
        "delta_t": delta_t,
        "tau": tau,
        "label": labels,
        "split": splits,
        "sample_id": [f"toy_{i:03d}" for i in range(n)],
    }


def _auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = max(1, (y == 1).sum()); n_neg = max(1, (y == 0).sum())
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    order = np.argsort(-s)
    yy = y[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    prec = tp / np.maximum(1, tp + fp)
    return float((prec * (yy == 1)).sum() / max(1, (y == 1).sum()))


def _metrics(y_true, scores, threshold):
    y = np.asarray(y_true).astype(int); s = np.asarray(scores, dtype=float)
    pred = (s > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    f2 = 5 * precision * recall / max(1e-12, 4 * precision + recall)
    return dict(accuracy=(tp + tn) / max(1, len(y)), precision=precision, recall=recall, f1=f1, f2=f2, roc_auc=_auc(y, s), pr_auc=_pr_auc(y, s), tp=tp, fp=fp, tn=tn, fn=fn)


def run_score_benchmark(cache: dict[str, Any], methods: list[str] | None = None, config: dict[str, Any] | None = None):
    config = config or {}
    methods = methods or list(SCORE_REGISTRY)
    ref_mask, eval_mask = reference_eval_masks(cache, config.get("reference_splits", ("train_normal", "val_normal")), config.get("eval_splits", ("test_normal", "test_abnormal")))
    eval_idx = torch.where(eval_mask)[0].tolist()
    labels = torch.as_tensor(cache["label"])[eval_mask].cpu().numpy()
    sample_ids = [cache.get("sample_id", [str(i) for i in range(len(cache["label"]))])[i] for i in eval_idx]
    q = float(config.get("threshold_quantile", 0.95))
    rows = []
    score_rows = []
    for name in methods:
        fn = SCORE_REGISTRY[name]
        cfg = dict(config)
        ref_scores = fn(cache, ref_mask, ref_mask, cfg).numpy()
        eval_scores = fn(cache, ref_mask, eval_mask, cfg).numpy()
        tau = float(np.quantile(ref_scores, q))
        m = _metrics(labels, eval_scores, tau)
        rows.append({"method": name, "threshold": tau, **m})
        for sid, lab, sc in zip(sample_ids, labels, eval_scores):
            score_rows.append({"method": name, "sample_id": sid, "label": int(lab), "score": float(sc), "threshold": tau})
    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def save_benchmark_outputs(metrics, scores, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics_table.csv", index=False)
    scores.to_csv(out / "scores.csv", index=False)
    best = metrics.sort_values(["roc_auc", "f1"], ascending=False).head(1)
    report = [
        "# Anomaly Score Benchmark Report\n\n",
        "## Metrics (CSV preview)\n\n",
        "```csv\n" + metrics.to_csv(index=False) + "```\n\n",
        "## Best (CSV preview)\n\n",
        "```csv\n" + best.to_csv(index=False) + "```\n",
    ]
    (out / "report.md").write_text("".join(report), encoding="utf-8")
