import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from snnodeatt.scoring.functional_scores import reference_eval_masks
from snnodeatt.scoring.kernel_scores import mmd_covariance_kernel_score, energy_distance_score
from snnodeatt.scoring.score_benchmark import make_toy_cache

def test_kernel_scores():
    cache = make_toy_cache()
    ref, ev = reference_eval_masks(cache)
    for fn in [mmd_covariance_kernel_score, energy_distance_score]:
        s = fn(cache, ref, ev, {"device": "cpu", "prototype_count": 8})
        assert s.shape == (int(ev.sum()),)
        assert torch.isfinite(s).all()
        assert s[-4:].mean() > s[:4].mean()

if __name__ == "__main__":
    test_kernel_scores()
    print("test_kernel_scores OK")
