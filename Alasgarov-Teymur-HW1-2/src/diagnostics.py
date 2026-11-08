"""Part B5a -- Diagnostics: look inside a network with forward hooks (Week 1 and Week-1 debugging)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_stats(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> list:
    """Per-Linear-layer statistics for one batch: the std of the layer's OUTPUT (pre-activation) and
    the L2 norm of its WEIGHT gradient.

    Return a list with one dict per nn.Linear (in network order):
        {"name": <module name>, "act_std": float, "grad_norm": float}
    Use register_forward_hook on every nn.Linear, run model(X) in train mode, compute the
    cross-entropy loss, call backward() once, read weight.grad, and REMOVE the hooks at the end.
    Zero the gradients first so numbers do not accumulate.
    """
    raise NotImplementedError  # TODO: your code here


@torch.no_grad()
def dead_relu_fraction(model: nn.Module, X: torch.Tensor) -> list:
    """For every nn.ReLU in the model, the fraction of its outputs that are exactly 0 on batch X
    (one float per ReLU, in order). Use forward hooks in eval mode and remove them afterwards."""
    raise NotImplementedError  # TODO: your code here


def overfit_one_batch(model: nn.Module, X: torch.Tensor, y: torch.Tensor, steps: int = 300, lr: float = 1e-2):
    """The sanity check from Week 1: train on ONE small batch until the loss is ~0.

    Use Adam(lr) on (X, y) for `steps` full-batch steps in train mode. Return (final_loss, final_accuracy).
    """
    raise NotImplementedError  # TODO: your code here
