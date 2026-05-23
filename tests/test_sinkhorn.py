import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from snnodeatt.scoring.sinkhorn import sinkhorn_divergence

def test_sinkhorn_basic():
    torch.manual_seed(0)
    p = torch.randn(10, 3)
    q_close = p + 0.01 * torch.randn(10, 3)
    q_far = p + 3.0
    w = torch.ones(10) / 10
    spp = sinkhorn_divergence(w, p, w, p)
    sclose = sinkhorn_divergence(w, p, w, q_close)
    sfar = sinkhorn_divergence(w, p, w, q_far)
    sym = abs(float(sinkhorn_divergence(w, p, w, q_close) - sinkhorn_divergence(w, q_close, w, p)))
    assert abs(float(spp)) < 1e-3
    assert float(sfar) > float(sclose)
    assert sym < 1e-3

if __name__ == "__main__":
    test_sinkhorn_basic()
    print("test_sinkhorn OK")
