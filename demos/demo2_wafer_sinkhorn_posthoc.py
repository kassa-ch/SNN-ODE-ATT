#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from snnodeatt.scoring.sinkhorn import sinkhorn_divergence

def main():
    p = argparse.ArgumentParser(description="Toy Sinkhorn trajectory scoring demo.")
    p.parse_args()
    x = torch.randn(12, 4)
    y = x + 0.05 * torch.randn(12, 4)
    z = x + 2.0
    w = torch.ones(12) / 12
    print("close:", float(sinkhorn_divergence(w, x, w, y)))
    print("far:", float(sinkhorn_divergence(w, x, w, z)))

if __name__ == "__main__":
    main()
