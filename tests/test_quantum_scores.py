import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from snnodeatt.scoring.functional_scores import reference_eval_masks, time_weights
from snnodeatt.scoring.quantum_scores import build_density_matrix, trace_distance_score, quantum_relative_entropy_score, von_neumann_entropy_score, purity_score, bures_distance_score
from snnodeatt.scoring.score_benchmark import make_toy_cache

def test_quantum_scores():
    cache = make_toy_cache()
    w = time_weights(cache)[0]
    rho = build_density_matrix(cache["m_reset"][0], cache["mask"][0].bool(), w)
    assert torch.allclose(torch.trace(rho), torch.tensor(1.0), atol=1e-4)
    assert torch.linalg.eigvalsh(rho).min() > -1e-6
    assert torch.isfinite(rho).all()
    ref, ev = reference_eval_masks(cache)
    for fn in [trace_distance_score, quantum_relative_entropy_score, von_neumann_entropy_score, purity_score, bures_distance_score]:
        s = fn(cache, ref, ev, {"device": "cpu"})
        assert s.shape == (int(ev.sum()),)
        assert torch.isfinite(s).all()
    same = trace_distance_score(cache, ref, ref, {"device": "cpu"})
    assert same.min() >= -1e-6

if __name__ == "__main__":
    test_quantum_scores()
    print("test_quantum_scores OK")
