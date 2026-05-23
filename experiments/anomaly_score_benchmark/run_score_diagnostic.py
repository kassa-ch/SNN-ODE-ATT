#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snnodeatt.scoring.functional_scores import reference_eval_masks
from snnodeatt.scoring.score_benchmark import run_score_benchmark
from snnodeatt.scoring.score_registry import SCORE_REGISTRY

QUANTILES = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
LOCAL_METHODS = [
    "trajectory_mahalanobis",
    "hamiltonian_energy",
    "sobolev_hminus1",
    "sobolev_h1",
    "mmd_cov_kernel",
    "energy_distance",
    "bures",
]


def auc_score(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg))


def pr_auc_score(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(-s)
    yy = y[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    precision = tp / np.maximum(1, tp + fp)
    return float((precision * (yy == 1)).sum() / max(1, (y == 1).sum()))


def metrics(y, score, threshold, direction):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score > threshold).astype(int) if direction == "high" else (score < threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    f2 = 5 * precision * recall / max(1e-12, 4 * precision + recall)
    return dict(accuracy=(tp + tn) / max(1, len(y)), precision=precision, recall=recall, f1=f1, f2=f2, tp=tp, fp=fp, tn=tn, fn=fn)


def dist_summary(score, y):
    score = np.asarray(score, dtype=float)
    y = np.asarray(y).astype(int)
    out = {}
    for prefix, mask in [("normal", y == 0), ("abnormal", y == 1)]:
        vals = score[mask]
        if len(vals) == 0:
            vals = np.array([np.nan])
        out.update({
            f"{prefix}_min": float(np.nanmin(vals)),
            f"{prefix}_q25": float(np.nanquantile(vals, 0.25)),
            f"{prefix}_median": float(np.nanmedian(vals)),
            f"{prefix}_mean": float(np.nanmean(vals)),
            f"{prefix}_q75": float(np.nanquantile(vals, 0.75)),
            f"{prefix}_max": float(np.nanmax(vals)),
        })
    return out


def load_cache(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def split_indices(cache):
    split = [str(x) for x in cache["split"]]
    labels = torch.as_tensor(cache["label"]).long().numpy()
    ref = np.array([(s in ("train_normal", "val_normal")) and labels[i] == 0 for i, s in enumerate(split)])
    ev = np.array([s in ("test_normal", "test_abnormal") for s in split])
    return split, labels, ref, ev


def cache_counts(cache):
    split, labels, _, _ = split_indices(cache)
    return {
        "shape": list(cache["m_reset"].shape),
        "split_counts": pd.Series(split).value_counts().to_dict(),
        "label_counts": pd.Series(labels).value_counts().to_dict(),
    }


def compute_all_scores(cache, methods):
    ref_mask, eval_mask = reference_eval_masks(cache)
    cfg = {"device": "cpu", "prototype_count": 64, "pca_dim": 32, "threshold_quantile": 0.95}
    metrics_df, scores_df = run_score_benchmark(cache, methods, cfg)
    return metrics_df, scores_df


def direction_quantile_grid(cache, scores_df, output_dir):
    split, labels_all, ref_mask, ev_mask = split_indices(cache)
    methods = sorted(scores_df.method.unique())
    ref_scores = {}
    eval_scores = {}
    eval_labels = {}
    rows = []
    dist_rows = []
    # Need train/val normal scores too; recompute each method for ref and eval via registry.
    for method in methods:
        fn = SCORE_REGISTRY[method]
        cfg = {"device": "cpu", "prototype_count": 64, "pca_dim": 32, "threshold_quantile": 0.95}
        ref_t = torch.as_tensor(ref_mask)
        ev_t = torch.as_tensor(ev_mask)
        ref_score = fn(cache, ref_t, ref_t, cfg).numpy()
        ev_score = fn(cache, ref_t, ev_t, cfg).numpy()
        y = labels_all[ev_mask]
        auc_high = auc_score(y, ev_score)
        auc_low = auc_score(y, -ev_score)
        pr_high = pr_auc_score(y, ev_score)
        pr_low = pr_auc_score(y, -ev_score)
        ds = dist_summary(ev_score, y)
        dist_rows.append({
            "method": method,
            "auc_high": auc_high,
            "auc_low": auc_low,
            "pr_auc_high": pr_high,
            "pr_auc_low": pr_low,
            "preferred_direction": "high" if auc_high >= auc_low else "low",
            "abnormal_median_gt_normal": bool(ds["abnormal_median"] > ds["normal_median"]),
            **ds,
        })
        for direction in ["high", "low"]:
            for q in QUANTILES:
                threshold = float(np.quantile(ref_score, q if direction == "high" else 1 - q))
                m = metrics(y, ev_score, threshold, direction)
                rows.append({
                    "method": method,
                    "direction": direction,
                    "quantile": q,
                    "threshold": threshold,
                    **m,
                    "auc": auc_high if direction == "high" else auc_low,
                    "pr_auc": pr_high if direction == "high" else pr_low,
                    **ds,
                })
    grid = pd.DataFrame(rows)
    dist = pd.DataFrame(dist_rows)
    grid.to_csv(output_dir / "direction_quantile_grid.csv", index=False)
    dist.to_csv(output_dir / "score_distribution.csv", index=False)
    return grid, dist


def view_signature(cache):
    mask = torch.as_tensor(cache["mask"]).bool()
    dt = torch.as_tensor(cache["delta_t"]).float()
    tau = torch.as_tensor(cache["tau"]).float()
    rows = []
    for i in range(mask.shape[0]):
        valid = mask[i]
        d = dt[i][valid]
        t = tau[i][valid]
        rows.append([
            float(valid.sum()),
            float(d.mean()),
            float(d.std(unbiased=False)),
            float(d.max()),
            float(torch.quantile(d, 0.9)),
            float(t.min()),
            float(t.max()),
        ])
    phi = np.asarray(rows, dtype=float)
    mu = phi.mean(axis=0)
    sd = phi.std(axis=0) + 1e-8
    return (phi - mu) / sd


def local_reference_grid(cache, output_dir):
    split, labels_all, ref_mask_np, ev_mask_np = split_indices(cache)
    ref_indices = np.where(ref_mask_np)[0]
    ev_indices = np.where(ev_mask_np)[0]
    phi = view_signature(cache)
    y = labels_all[ev_mask_np]
    rows = []
    local_scores_by_method = {}
    global_scores_by_method = {}
    local_ref_scores_by_method = {}
    for method in LOCAL_METHODS:
        fn = SCORE_REGISTRY[method]
        cfg = {"device": "cpu", "prototype_count": 64, "pca_dim": 32}
        ref_mask_t = torch.as_tensor(ref_mask_np)
        ev_mask_t = torch.as_tensor(ev_mask_np)
        global_ref_score = fn(cache, ref_mask_t, ref_mask_t, cfg).numpy()
        global_eval_score = fn(cache, ref_mask_t, ev_mask_t, cfg).numpy()
        global_scores_by_method[method] = global_eval_score
        local_eval = []
        local_ref_diag = []
        # Ref diagnostic: local reference for each ref sample excludes itself.
        for idx in ref_indices:
            d = np.linalg.norm(phi[ref_indices] - phi[idx], axis=1)
            order = ref_indices[np.argsort(d)]
            order = order[order != idx][:15]
            local_mask = np.zeros(len(labels_all), dtype=bool)
            local_mask[order] = True
            sample_mask = np.zeros(len(labels_all), dtype=bool)
            sample_mask[idx] = True
            local_ref_diag.append(float(fn(cache, torch.as_tensor(local_mask), torch.as_tensor(sample_mask), cfg).numpy()[0]))
        for idx in ev_indices:
            d = np.linalg.norm(phi[ref_indices] - phi[idx], axis=1)
            order = ref_indices[np.argsort(d)[:15]]
            local_mask = np.zeros(len(labels_all), dtype=bool)
            local_mask[order] = True
            sample_mask = np.zeros(len(labels_all), dtype=bool)
            sample_mask[idx] = True
            local_eval.append(float(fn(cache, torch.as_tensor(local_mask), torch.as_tensor(sample_mask), cfg).numpy()[0]))
        local_ref_score = np.asarray(local_ref_diag, dtype=float)
        local_eval_score = np.asarray(local_eval, dtype=float)
        local_scores_by_method[method] = local_eval_score
        local_ref_scores_by_method[method] = local_ref_score
        for ref_type, ref_score, eval_score in [
            ("global", global_ref_score, global_eval_score),
            ("local", local_ref_score, local_eval_score),
        ]:
            auc_high = auc_score(y, eval_score)
            auc_low = auc_score(y, -eval_score)
            pr_high = pr_auc_score(y, eval_score)
            pr_low = pr_auc_score(y, -eval_score)
            for direction in ["high", "low"]:
                for q in QUANTILES:
                    threshold = float(np.quantile(ref_score, q if direction == "high" else 1 - q))
                    m = metrics(y, eval_score, threshold, direction)
                    rows.append({
                        "method": method,
                        "reference_type": ref_type,
                        "direction": direction,
                        "quantile": q,
                        "threshold": threshold,
                        **m,
                        "auc": auc_high if direction == "high" else auc_low,
                        "pr_auc": pr_high if direction == "high" else pr_low,
                    })
        # Rich combines normalized global/local ratios with per-direction thresholds.
        for direction in ["high", "low"]:
            for q in QUANTILES:
                tau_g = float(np.quantile(global_ref_score, q if direction == "high" else 1 - q))
                tau_l = float(np.quantile(local_ref_score, q if direction == "high" else 1 - q))
                eps = 1e-8
                if direction == "high":
                    rich = 0.65 * (global_eval_score / (tau_g + eps)) + 0.35 * (local_eval_score / (tau_l + eps))
                    rich_ref = 0.65 * (global_ref_score / (tau_g + eps)) + 0.35 * (local_ref_score / (tau_l + eps))
                else:
                    rich = 0.65 * ((tau_g + eps) / (global_eval_score + eps)) + 0.35 * ((tau_l + eps) / (local_eval_score + eps))
                    rich_ref = 0.65 * ((tau_g + eps) / (global_ref_score + eps)) + 0.35 * ((tau_l + eps) / (local_ref_score + eps))
                threshold = float(np.quantile(rich_ref, q))
                m = metrics(y, rich, threshold, "high")
                rows.append({
                    "method": method,
                    "reference_type": "rich",
                    "direction": direction,
                    "quantile": q,
                    "threshold": threshold,
                    **m,
                    "auc": auc_score(y, rich),
                    "pr_auc": pr_auc_score(y, rich),
                })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "local_reference_grid.csv", index=False)
    return df


def write_distribution_report(grid, dist, local_grid, output_dir):
    top_auc = dist.sort_values(["auc_high", "pr_auc_high"], ascending=False).head(5)
    top_low_auc = dist.sort_values(["auc_low", "pr_auc_low"], ascending=False).head(5)
    top_f1 = grid.sort_values(["f1", "recall", "precision"], ascending=False).head(5)
    top_f2 = grid.sort_values(["f2", "recall", "precision"], ascending=False).head(5)
    fp3 = grid[grid.fp <= 3].sort_values(["recall", "f1", "f2"], ascending=False).head(5)
    md = ["# Score Distribution Summary\n\n"]
    md += ["## Normal vs abnormal distribution\n\n```csv\n", dist.to_csv(index=False), "```\n\n"]
    md += ["## Top by AUC(score)\n\n```csv\n", top_auc.to_csv(index=False), "```\n\n"]
    md += ["## Top by AUC(-score)\n\n```csv\n", top_low_auc.to_csv(index=False), "```\n\n"]
    md += ["## Top by best F1\n\n```csv\n", top_f1.to_csv(index=False), "```\n\n"]
    md += ["## Top by best F2\n\n```csv\n", top_f2.to_csv(index=False), "```\n\n"]
    md += ["## Top Recall under FP <= 3\n\n```csv\n", fp3.to_csv(index=False), "```\n\n"]
    if local_grid is not None and len(local_grid):
        md += ["## Local reference top rows\n\n```csv\n", local_grid.sort_values(["f1", "recall", "precision"], ascending=False).head(10).to_csv(index=False), "```\n"]
    (output_dir / "score_distribution_summary.md").write_text("".join(md), encoding="utf-8")


def write_final_report(cache, grid, dist, local_grid, output_dir, report_path):
    counts = cache_counts(cache)
    best = grid.sort_values(["f1", "recall", "precision", "auc"], ascending=False).iloc[0]
    best_auc = dist.sort_values(["auc_high", "pr_auc_high"], ascending=False).iloc[0]
    best_low_auc = dist.sort_values(["auc_low", "pr_auc_low"], ascending=False).iloc[0]
    best_local = local_grid.sort_values(["f1", "recall", "precision", "auc"], ascending=False).iloc[0] if local_grid is not None and len(local_grid) else None
    any_exceed_aligned = bool((grid.f1 > 0.5714).any() or (local_grid is not None and (local_grid.f1 > 0.5714).any()))
    any_near_best = bool((grid.f1 >= 0.8).any() or (local_grid is not None and (local_grid.f1 >= 0.8).any()))
    direction_reversed = dist[dist.auc_low > dist.auc_high].method.tolist()
    threshold_issue = bool((grid[(grid['quantile'] < 0.95) & (grid['f1'] > 0)].shape[0]) > 0)
    local_improves = False
    if best_local is not None:
        local_improves = bool(best_local.f1 > best.f1)
    decision = "C. 所有方法仍弱，建议停止在 wafer exp4 上继续调分数，转向 HAI/TEP/SWaT/WADI。"
    if any_near_best:
        decision = "A. 有方法接近 best，建议 focused validation。"
    elif any_exceed_aligned:
        decision = "B. 有方法超过 aligned Mahalanobis，但未接近 best，建议小范围优化。"
    lines = []
    lines += ["# Exp4 Score Diagnostic Report\n\n"]
    lines += ["## 1. Executive Summary\n\n"]
    lines += [f"- Default q0.95 high-score failure: thresholds are conservative and most methods rank abnormal below/near normal, giving TP=0 in the previous benchmark.\n"]
    lines += [f"- Direction reversed methods by AUC(-score)>AUC(score): `{direction_reversed}`.\n"]
    lines += [f"- Threshold issue observed: `{threshold_issue}`.\n"]
    lines += [f"- Local/rich reference improves best F1: `{local_improves}`.\n"]
    lines += [f"- Decision: {decision}\n\n"]
    lines += ["## 2. Data and Cache\n\n"]
    lines += [f"- cache path: `/root/autodl-tmp/wafer_att_ctsr_vca/ctsr_vca/cache_sinkhorn_fix/exp4_hidden_cache_aligned.pt`\n"]
    lines += [f"- shape: `{counts['shape']}`\n"]
    lines += [f"- split counts: `{counts['split_counts']}`\n"]
    lines += [f"- label counts: `{counts['label_counts']}`\n\n"]
    lines += ["## 3. Direction Diagnostic\n\n```csv\n", dist[["method","auc_high","auc_low","pr_auc_high","pr_auc_low","preferred_direction","normal_median","abnormal_median","abnormal_median_gt_normal"]].to_csv(index=False), "```\n\n"]
    lines += ["## 4. Threshold Diagnostic\n\n"]
    lines += [f"- Best global grid row: `{best['method']}`, direction=`{best['direction']}`, q=`{best['quantile']}`, TP/FP/TN/FN=`{int(best['tp'])}/{int(best['fp'])}/{int(best['tn'])}/{int(best['fn'])}`, F1=`{best['f1']:.4f}`, F2=`{best['f2']:.4f}`.\n\n"]
    lines += ["```csv\n", grid.sort_values(["f1","recall","precision","auc"], ascending=False).head(15).to_csv(index=False), "```\n\n"]
    lines += ["## 5. Score Distribution\n\n```csv\n", dist.to_csv(index=False), "```\n\n"]
    lines += ["## 6. Local Reference Diagnostic\n\n"]
    if best_local is not None:
        lines += [f"- Best local/rich row: `{best_local['method']}`, reference=`{best_local['reference_type']}`, direction=`{best_local['direction']}`, q=`{best_local['quantile']}`, TP/FP/TN/FN=`{int(best_local['tp'])}/{int(best_local['fp'])}/{int(best_local['tn'])}/{int(best_local['fn'])}`, F1=`{best_local['f1']:.4f}`, AUC=`{best_local['auc']:.4f}`.\n\n"]
        lines += ["```csv\n", local_grid.sort_values(["f1","recall","precision","auc"], ascending=False).head(15).to_csv(index=False), "```\n\n"]
    else:
        lines += ["- Local reference diagnostic was not available.\n\n"]
    lines += ["## 7. Comparison with Baselines\n\n"]
    lines += ["- post-hoc Sinkhorn: `F1=0.2222`\n"]
    lines += ["- aligned Mahalanobis: `F1=0.5714`\n"]
    lines += ["- exp4 current best M1+M2: `TP/FP/TN/FN=6/3/34/0`, `F1=0.8000`\n\n"]
    lines += ["## 8. Root Cause\n\n"]
    lines += ["- threshold issue: lower quantiles recover some positives only if best grid F1 > 0.\n"]
    lines += ["- direction issue: methods with AUC(-score)>AUC(score) are likely inverted for exp4.\n"]
    lines += ["- global reference issue: if local/rich rows improve F1/AUC, global reference is part of the failure.\n"]
    lines += ["- score itself weak / hidden mismatch: if AUC remains low and F1 remains below aligned Mahalanobis, the score geometry does not match exp4 hidden representation.\n\n"]
    lines += ["## 9. Decision\n\n", f"{decision}\n\n"]
    lines += ["## 10. Next Action\n\n"]
    lines += ["1. If continuing, test at most three focused variants from the top local/rich rows, not a wide grid.\n"]
    lines += ["2. Prefer external HAI/TEP/SWaT/WADI validation for density/functional scores.\n"]
    lines += ["3. Keep M1+M2 as current exp4 best unless a diagnostic row reaches comparable F1 without label leakage.\n"]
    report_path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--benchmark_dir", default="experiments/anomaly_score_benchmark/outputs_exp4_local")
    parser.add_argument("--output_dir", default="experiments/anomaly_score_benchmark/outputs_exp4_diagnostic")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache_path)
    methods = list(SCORE_REGISTRY)
    scores_path = Path(args.benchmark_dir) / "scores.csv"
    if not scores_path.exists():
        _metrics, scores_df = compute_all_scores(cache, methods)
        scores_df.to_csv(scores_path, index=False)
    else:
        scores_df = pd.read_csv(scores_path)
    grid, dist = direction_quantile_grid(cache, scores_df, output_dir)
    local_grid = local_reference_grid(cache, output_dir)
    write_distribution_report(grid, dist, local_grid, output_dir)
    compact = grid.sort_values(["f1", "recall", "precision", "auc"], ascending=False).head(40)
    compact.to_csv(ROOT / "experiments/anomaly_score_benchmark/score_diagnostic_exp4.csv", index=False)
    write_final_report(cache, grid, dist, local_grid, output_dir, ROOT / "docs/exp4_score_diagnostic_report.md")
    print("best_global")
    print(grid.sort_values(["f1", "recall", "precision", "auc"], ascending=False).head(10).to_string(index=False))
    print("best_local")
    print(local_grid.sort_values(["f1", "recall", "precision", "auc"], ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    import argparse
    main()
