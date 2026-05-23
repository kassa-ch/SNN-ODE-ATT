#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snnodeatt.scoring.score_benchmark import make_toy_cache, run_score_benchmark, save_benchmark_outputs


DEFAULT_METHODS = [
    "euclidean",
    "sobolev_h1",
    "sobolev_hminus1",
    "trajectory_mahalanobis",
    "mmd_cov_kernel",
    "energy_distance",
    "trace_distance",
    "qre",
    "von_neumann_entropy",
    "purity",
    "bures",
    "hamiltonian_energy",
    "ode_residual_energy",
]


def main():
    parser = argparse.ArgumentParser(description="Run anomaly score benchmark on a hidden cache or toy data.")
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--config", default=str(Path(__file__).with_name("benchmark_config.yaml")))
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--output_dir", default=str(Path(__file__).with_name("outputs")))
    args = parser.parse_args()
    if args.toy or args.cache_path is None:
        cache = make_toy_cache()
    else:
        cache = torch.load(args.cache_path, map_location="cpu", weights_only=False)
    config = {
        "threshold_quantile": 0.95,
        "time_weight": "dt",
        "prototype_count": 16 if args.toy else 64,
        "device": "cpu",
        "pca_dim": None if args.toy else 32,
    }
    metrics, scores = run_score_benchmark(cache, DEFAULT_METHODS, config)
    save_benchmark_outputs(metrics, scores, args.output_dir)
    print(metrics.sort_values(["roc_auc", "f1"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
