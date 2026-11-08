"""B5a -- hook-based diagnostics."""
import torch, torch.nn as nn
from data import set_seed
from model import MLP, init_weights
import diagnostics as D


def test_layer_stats_structure_and_hooks_removed():
    set_seed(0)
    m = MLP(20, [32, 32], 5)
    X, y = torch.randn(16, 20), torch.randint(0, 5, (16,))
    s = D.layer_stats(m, X, y)
    assert len(s) == 3 and all({"name", "act_std", "grad_norm"} <= set(d) for d in s)
    assert all(d["act_std"] > 0 and d["grad_norm"] > 0 for d in s)
    assert all(len(mod._forward_hooks) == 0 for mod in m.modules())     # hooks removed
    assert all(p.grad is None for p in m.parameters())                  # gradients cleaned up


def test_layer_stats_grad_vanishes_with_sigmoid_and_bad_init():
    set_seed(0)
    X, y = torch.randn(64, 20), torch.randint(0, 5, (64,))
    bad = MLP(20, [32] * 8, 5, act="sigmoid"); init_weights(bad, "xavier")
    good = MLP(20, [32] * 8, 5, act="relu"); init_weights(good, "he")
    sb, sg = D.layer_stats(bad, X, y), D.layer_stats(good, X, y)
    assert sb[0]["grad_norm"] < 0.05 * sb[-1]["grad_norm"]             # vanishing towards the input
    assert sg[0]["grad_norm"] > 0.05 * sg[-1]["grad_norm"]


def test_dead_relu_fraction():
    m = MLP(10, [20, 20], 3)
    init_weights(m, "he")
    X = torch.randn(200, 10)
    f = D.dead_relu_fraction(m, X)
    assert len(f) == 2 and all(0.2 < v < 0.8 for v in f)               # roughly half of ReLU outputs are zero
    for lin in m.net:
        if isinstance(lin, nn.Linear):
            lin.bias.data.fill_(-100.0)
    assert all(v > 0.99 for v in D.dead_relu_fraction(m, X))           # every unit dead
    assert all(len(mod._forward_hooks) == 0 for mod in m.modules())


def test_overfit_one_batch():
    set_seed(0)
    X, y = torch.randn(16, 20), torch.randint(0, 4, (16,))
    m = MLP(20, [64, 64], 4)
    loss, acc = D.overfit_one_batch(m, X, y, steps=300, lr=1e-2)
    assert loss < 0.05 and acc == 1.0
