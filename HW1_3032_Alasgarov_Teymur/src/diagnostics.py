"""Part B5a -- Diagnostics: look inside a network with forward hooks (Week 1 and Week-1 debugging)."""

import torch
import torch.nn.functional as F
from torch import nn


def layer_stats(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> list:
    """Per-Linear-layer statistics for one batch: the std of the layer's OUTPUT (pre-activation) and
    the L2 norm of its WEIGHT gradient.

    Return a list with one dict per nn.Linear (in network order):
        {"name": <module name>, "act_std": float, "grad_norm": float}
    Use register_forward_hook on every nn.Linear, run model(X) in train mode, compute the
    cross-entropy loss, call backward() once, read weight.grad, and REMOVE the hooks at the end.
    Zero the gradients first so numbers do not accumulate.
    """
    linear_layers = []
    act_stds = {}
    handles = []

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            linear_layers.append((name, mod))

            def make_hook(n):
                def hook_fn(module, inp, out):
                    act_stds[n] = float(out.detach().std().item())

                return hook_fn

            handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        model.train()
        model.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        results = []
        for name, mod in linear_layers:
            grad_norm = (
                float(mod.weight.grad.detach().norm().item())
                if mod.weight.grad is not None
                else 0.0
            )
            results.append(
                {
                    "name": name,
                    "act_std": act_stds.get(name, 0.0),
                    "grad_norm": grad_norm,
                }
            )
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)

    return results


@torch.no_grad()
def dead_relu_fraction(model: nn.Module, X: torch.Tensor) -> list:
    """For every nn.ReLU in the model, the fraction of its outputs that are exactly 0 on batch X
    (one float per ReLU, in order). Use forward hooks in eval mode and remove them afterwards.
    """
    was_training = model.training
    model.eval()
    handles = []
    fractions = []

    for mod in model.modules():
        if isinstance(mod, nn.ReLU):
            idx = len(fractions)
            fractions.append(0.0)

            def make_hook(i):
                def hook_fn(module, inp, out):
                    fractions[i] = float((out == 0.0).float().mean().item())

                return hook_fn

            handles.append(mod.register_forward_hook(make_hook(idx)))

    try:
        model(X)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    return fractions


def overfit_one_batch(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    steps: int = 300,
    lr: float = 1e-2,
):
    """The sanity check from Week 1: train on ONE small batch until the loss is ~0.

    Use Adam(lr) on (X, y) for `steps` full-batch steps in train mode. Return (final_loss, final_accuracy).
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    final_loss = float(loss.item())
    final_acc = float((logits.argmax(dim=-1) == y).float().mean().item())
    return final_loss, final_acc
