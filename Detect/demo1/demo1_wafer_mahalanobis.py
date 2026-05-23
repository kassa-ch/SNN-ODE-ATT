#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from snnodeatt.scoring.mahalanobis import MahalanobisScorer

def main():
    p = argparse.ArgumentParser(description="Toy Mahalanobis demo.")
    p.parse_args()
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(20, 8))
    test = np.vstack([rng.normal(size=(5, 8)), rng.normal(loc=3, size=(2, 8))])
    scores = MahalanobisScorer().fit(normal).score(test)
    print("toy scores:", np.round(scores, 3).tolist())

if __name__ == "__main__":
    main()
