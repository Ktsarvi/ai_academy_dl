"""B4 -- schedule, one epoch, evaluation, fit with early stopping."""
import math, pytest, torch, torch.nn as nn
import torch.nn.functional as F
from data import make_loader, set_seed
from model import MLP
import train as T


def _toy(n=256, d=10, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, K, generator=g)
    return X, (X @ w).argmax(1)


def test_warmup_cosine_values():
    assert abs(T.warmup_cosine(0, 10, 100) - 0.1) < 1e-9
    assert abs(T.warmup_cosine(9, 10, 100) - 1.0) < 1e-9
    assert abs(T.warmup_cosine(10, 10, 100) - 1.0) < 1e-9
    assert abs(T.warmup_cosine(55, 10, 100) - 0.5) < 1e-9          # halfway through the cosine
    assert abs(T.warmup_cosine(100, 10, 100)) < 1e-9
    assert abs(T.warmup_cosine(100, 10, 100, min_factor=0.1) - 0.1) < 1e-9
    assert abs(T.warmup_cosine(500, 10, 100, min_factor=0.1) - 0.1) < 1e-9   # clipped after the end
    vals = [T.warmup_cosine(s, 10, 100) for s in range(10, 101)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


def test_accuracy_from_logits():
    assert T.accuracy_from_logits(torch.tensor([[2., 1.], [0., 3.], [5., 0.]]), torch.tensor([0, 1, 1])) == pytest.approx(2 / 3, abs=1e-6)


def test_evaluate_is_eval_mode_and_weighted():
    set_seed(0)
    X, y = _toy(n=103)
    m = MLP(10, [16], 3, dropout=0.5)
    loader = make_loader(X, y, batch_size=32, shuffle=False)       # uneven last batch (103 = 3*32 + 7)
    m.train()
    l1, a1 = T.evaluate(m, loader)
    l2, a2 = T.evaluate(m, loader)
    assert (l1, a1) == (l2, a2)                                    # dropout was off
    assert not m.training                                          # evaluate left the model in eval mode
    with torch.no_grad():
        out = m(X)
    assert abs(l1 - F.cross_entropy(out, y).item()) < 1e-5         # weighted by batch size
    assert abs(a1 - (out.argmax(1) == y).float().mean().item()) < 1e-6


def test_train_one_epoch_learns_and_steps_scheduler():
    set_seed(0)
    X, y = _toy()
    m = MLP(10, [32], 3)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    loader = make_loader(X, y, batch_size=64, seed=0)
    l0, _ = T.evaluate(m, loader)
    tl, ta = T.train_one_epoch(m, loader, opt, scheduler=sched)
    for _ in range(14):
        tl, ta = T.train_one_epoch(m, loader, opt, scheduler=sched)
    assert sched.last_epoch == 15 * 4                              # once per batch: 4 batches x 15 epochs
    assert m.training
    assert tl < l0 * 0.5 and ta > 0.8


def test_clip_norm_limits_update():
    set_seed(0)
    X, y = _toy()
    m = MLP(10, [32], 3)
    before = [p.clone() for p in m.parameters()]
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    T.train_one_epoch(m, make_loader(X[:64], y[:64], batch_size=64), opt, clip_norm=1e-3)
    step = torch.sqrt(sum(((p - b) ** 2).sum() for p, b in zip(m.parameters(), before)))
    assert step.item() <= 1e-3 * 1.0 + 1e-6                        # lr * clipped gradient norm


def test_fit_history_and_best_restore():
    set_seed(0)
    X, y = _toy(n=400)
    tr, va = make_loader(X[:300], y[:300], 50), make_loader(X[300:], y[300:], 100, shuffle=False)
    m = MLP(10, [64], 3)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    h = T.fit(m, tr, va, opt, epochs=20)
    assert len(h["val_loss"]) == 20 and len(h["train_acc"]) == 20
    assert 0 <= h["best_epoch"] < 20
    vl, _ = T.evaluate(m, va)
    assert abs(vl - min(h["val_loss"])) < 1e-6                     # best weights were restored


def test_early_stopping_triggers():
    set_seed(0)
    g = torch.Generator().manual_seed(1)
    X = torch.randn(80, 10, generator=g); y = torch.randint(0, 3, (80,), generator=g)   # pure noise: val cannot improve for long
    tr, va = make_loader(X[:40], y[:40], 20), make_loader(X[40:], y[40:], 40, shuffle=False)
    m = MLP(10, [128], 3)
    h = T.fit(m, tr, va, torch.optim.Adam(m.parameters(), lr=1e-2), epochs=200, patience=3)
    assert len(h["val_loss"]) < 200
    assert len(h["val_loss"]) == h["best_epoch"] + 1 + 3 or len(h["val_loss"]) >= h["best_epoch"] + 4
