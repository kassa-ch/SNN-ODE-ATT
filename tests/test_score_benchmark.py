import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from snnodeatt.scoring.score_benchmark import make_toy_cache, run_score_benchmark
from snnodeatt.scoring.score_registry import SCORE_REGISTRY

def test_score_benchmark():
    cache = make_toy_cache()
    metrics, scores = run_score_benchmark(cache, list(SCORE_REGISTRY), {"device": "cpu", "prototype_count": 8, "pca_dim": None})
    assert set(metrics["method"]) == set(SCORE_REGISTRY)
    assert len(scores) > 0
    assert metrics["roc_auc"].max() > 0.5

if __name__ == "__main__":
    test_score_benchmark()
    print("test_score_benchmark OK")
