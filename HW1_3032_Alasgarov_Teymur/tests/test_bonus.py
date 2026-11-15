"""Bonus (optional): MyAdam must reproduce torch.optim.Adam; MixUp sanity."""
import torch
from bonus import MyAdam, mixup_batch


def test_myadam_matches_torch_adam():
    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.randn(5, 4)); p2 = torch.nn.Parameter(p1.detach().clone())
    o1, o2 = MyAdam([p1], lr=1e-2), torch.optim.Adam([p2], lr=1e-2)
    for _ in range(25):
        for p, o in ((p1, o1), (p2, o2)):
            o.zero_grad()
            ((p ** 3).sum() + (p * 2).pow(2).mean()).backward()
            o.step()
    assert torch.allclose(p1, p2, atol=1e-6)


def test_mixup_shapes_and_label_sums():
    torch.manual_seed(0)
    x, y = torch.randn(10, 6), torch.eye(10)[torch.randint(0, 10, (10,))]
    xm, ym = mixup_batch(x, y, 0.4)
    assert xm.shape == x.shape and ym.shape == y.shape
    assert torch.allclose(ym.sum(1), torch.ones(10), atol=1e-6)
