import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from snnodeatt.scoring.functional_scores import reference_eval_masks, euclidean_latent_score, sobolev_h1_score, negative_sobolev_hminus1_score, covariance_aware_trajectory_energy
from snnodeatt.scoring.score_benchmark import make_toy_cache

def test_functional_scores():
    cache = make_toy_cache()
    ref, ev = reference_eval_masks(cache)
    for fn in [euclidean_latent_score, sobolev_h1_score, negative_sobolev_hminus1_score, covariance_aware_trajectory_energy]:
        s = fn(cache, ref, ev, {"device": "cpu"})
        assert s.shape == (int(ev.sum()),)
        assert torch.isfinite(s).all()

if __name__ == "__main__":
    test_functional_scores()
    print("test_functional_scores OK")
