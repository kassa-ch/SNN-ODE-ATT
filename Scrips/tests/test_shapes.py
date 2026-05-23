import torch

def test_basic_shapes():
    x = torch.randn(4, 10, 6)
    mask = torch.ones(4, 10)
    delta_t = torch.ones(4, 10)
    tau = torch.linspace(0, 1, 10).repeat(4, 1)
    assert x.shape == (4, 10, 6)
    assert mask.shape == delta_t.shape == tau.shape == (4, 10)

if __name__ == "__main__":
    test_basic_shapes()
    print("test_shapes OK")
