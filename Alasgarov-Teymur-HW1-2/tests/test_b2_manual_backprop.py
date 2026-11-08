"""B2 -- manual backprop must agree with autograd and with finite differences; Swish passes gradcheck."""
import pytest, torch
import torch.nn.functional as F
import manual_mlp as mm


def _setup(act, seed=0):
    g = torch.Generator().manual_seed(seed + 1)
    X = torch.randn(7, 5, generator=g, dtype=torch.float64)
    y = torch.randint(0, 3, (7,), generator=g)
    return mm.init_params(5, 6, 3, seed=seed), X, y


@pytest.mark.parametrize("act", ["tanh", "relu"])
def test_forward_shapes_and_cache(act):
    params, X, y = _setup(act)
    logits, cache = mm.forward(params, X, act)
    assert logits.shape == (7, 3)
    assert {"X", "z1", "a1"} <= set(cache)


@pytest.mark.parametrize("act", ["tanh", "relu"])
def test_backward_matches_autograd(act):
    params, X, y = _setup(act)
    logits, cache = mm.forward(params, X, act)
    grads = mm.backward(params, cache, logits, y, act)
    leaf = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    out, _ = mm.forward(leaf, X, act)
    F.cross_entropy(out, y).backward()
    for k in ("W1", "b1", "W2", "b2"):
        assert torch.allclose(grads[k], leaf[k].grad, atol=1e-9), k


def test_numerical_grad_on_quadratic():
    x = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    g = mm.numerical_grad(lambda v: (v ** 2).sum(), x)
    assert torch.allclose(g, 2 * x, atol=1e-6)
    assert torch.equal(x, torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64))   # input untouched


def test_backward_matches_finite_differences():
    params, X, y = _setup("tanh", seed=3)
    logits, cache = mm.forward(params, X, "tanh")
    grads = mm.backward(params, cache, logits, y, "tanh")

    def loss_of_W1(W1):
        p = dict(params, W1=W1)
        return F.cross_entropy(mm.forward(p, X, "tanh")[0], y)

    num = mm.numerical_grad(loss_of_W1, params["W1"])
    assert torch.allclose(grads["W1"], num, atol=1e-6)


def test_swish_gradcheck_and_value():
    x = torch.randn(11, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(mm.swish, (x,))
    assert torch.allclose(mm.swish(x), x * torch.sigmoid(x))
