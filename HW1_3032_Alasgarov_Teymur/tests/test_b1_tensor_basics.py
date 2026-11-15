"""B1 -- tensor basics and Week-1 losses must match PyTorch's reference implementations."""
import pytest, torch
import torch.nn.functional as F
import tensor_basics as tb


def test_standardize():
    X = torch.randn(50, 7) * 3 + 2
    Z = tb.standardize(X, X.mean(0), X.std(0))
    assert Z.shape == X.shape
    assert torch.allclose(Z.mean(0), torch.zeros(7), atol=1e-5)
    assert torch.allclose(Z.std(0), torch.ones(7), atol=1e-4)


def test_one_hot():
    y = torch.tensor([0, 2, 1, 2])
    out = tb.one_hot(y, 3)
    assert out.dtype == torch.float32
    assert torch.equal(out, F.one_hot(y, 3).float())


def test_pairwise_sq_dists():
    A, B = torch.randn(9, 5), torch.randn(4, 5)
    assert torch.allclose(tb.pairwise_sq_dists(A, B), torch.cdist(A, B) ** 2, atol=1e-4)
    assert (tb.pairwise_sq_dists(A, A).diagonal().abs() < 1e-4).all()


def test_softmax_matches_and_is_stable():
    z = torch.randn(6, 5)
    assert torch.allclose(tb.softmax(z), torch.softmax(z, -1), atol=1e-6)
    big = torch.tensor([[1000.0, 1001.0, 999.0]])
    p = tb.softmax(big)
    assert torch.isfinite(p).all() and torch.allclose(p.sum(), torch.tensor(1.0))
    assert torch.allclose(p, torch.softmax(big, -1), atol=1e-6)


def test_log_softmax_and_cross_entropy():
    z, y = torch.randn(32, 10), torch.randint(0, 10, (32,))
    assert torch.allclose(tb.log_softmax(z), F.log_softmax(z, -1), atol=1e-5)
    assert torch.allclose(tb.cross_entropy(z, y), F.cross_entropy(z, y), atol=1e-5)
    big = torch.tensor([[2000.0, 0.0, -2000.0]])
    assert torch.isfinite(tb.cross_entropy(big, torch.tensor([2])))


def test_bce_with_logits_matches_and_is_stable():
    z, y = torch.randn(64) * 4, torch.randint(0, 2, (64,)).float()
    assert torch.allclose(tb.bce_with_logits(z, y), F.binary_cross_entropy_with_logits(z, y), atol=1e-5)
    zz = torch.tensor([500.0, -500.0]); yy = torch.tensor([0.0, 1.0])
    out = tb.bce_with_logits(zz, yy)
    assert torch.isfinite(out) and abs(out.item() - 500.0) < 1e-3


def test_mse_and_shape_guard():
    a, b = torch.randn(8, 3), torch.randn(8, 3)
    assert torch.allclose(tb.mse(a, b), F.mse_loss(a, b), atol=1e-6)
    with pytest.raises(AssertionError):
        tb.mse(torch.randn(8, 1), torch.randn(8))
