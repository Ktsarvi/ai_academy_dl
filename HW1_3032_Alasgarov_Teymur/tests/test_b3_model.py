"""B3 -- the MLP module: structure, behaviour, init statistics."""
import math, pytest, torch, torch.nn as nn
from model import MLP, init_weights, count_params


def test_param_count_and_shapes():
    m = MLP(64, [32, 16], 10)
    assert count_params(m) == (64 * 32 + 32) + (32 * 16 + 16) + (16 * 10 + 10)
    assert m(torch.randn(5, 64)).shape == (5, 10)
    m.net[0].weight.requires_grad_(False)
    assert count_params(m) == count_params(m, trainable_only=False) - 64 * 32


@pytest.mark.parametrize("act,cls", [("relu", nn.ReLU), ("tanh", nn.Tanh), ("gelu", nn.GELU), ("sigmoid", nn.Sigmoid)])
def test_activation_choice(act, cls):
    m = MLP(8, [4, 4], 2, act=act)
    assert sum(isinstance(l, cls) for l in m.net) == 2


def test_dropout_and_norm_layers():
    m = MLP(8, [4, 4], 2, dropout=0.3, norm="batch")
    assert sum(isinstance(l, nn.Dropout) for l in m.net) == 2
    assert sum(isinstance(l, nn.BatchNorm1d) for l in m.net) == 2
    m = MLP(8, [4], 2, norm="layer")
    assert any(isinstance(l, nn.LayerNorm) for l in m.net)
    m = MLP(8, [4], 2)
    assert not any(isinstance(l, (nn.Dropout, nn.BatchNorm1d, nn.LayerNorm)) for l in m.net)


def test_dropout_is_off_in_eval():
    torch.manual_seed(0)
    m = MLP(10, [50], 3, dropout=0.5)
    x = torch.randn(4, 10)
    m.eval()
    assert torch.equal(m(x), m(x))
    m.train()
    assert not torch.equal(m(x), m(x))


def test_he_and_xavier_std():
    torch.manual_seed(0)
    m = MLP(400, [300], 100)
    init_weights(m, "he")
    assert abs(m.net[0].weight.std().item() - math.sqrt(2 / 400)) < 0.003
    init_weights(m, "xavier")
    assert abs(m.net[0].weight.std().item() - math.sqrt(2 / (400 + 300))) < 0.003
    init_weights(m, "normal", std=0.05)
    assert abs(m.net[0].weight.std().item() - 0.05) < 0.003
    assert m.net[0].bias.abs().sum().item() == 0


def test_zeros_init_and_unknown_scheme():
    m = MLP(6, [5], 2)
    init_weights(m, "zeros")
    assert all(float(p.detach().abs().sum()) == 0 for p in m.parameters())
    with pytest.raises(ValueError):
        init_weights(m, "nope")
