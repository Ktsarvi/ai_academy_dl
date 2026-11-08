"""Part B3 -- An MLP as an nn.Module, with configurable activation, normalisation, dropout and init."""
import math
import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU, "silu": nn.SiLU,
}


class MLP(nn.Module):
    """Fully connected network: [Linear -> (norm) -> activation -> (dropout)] x len(hidden) -> Linear.

    Args:
        in_dim, hidden_dims (list[int]), out_dim: sizes.
        act: key of ACTIVATIONS.
        dropout: probability p; add nn.Dropout(p) after each activation only if p > 0.
        norm: None, "batch" (nn.BatchNorm1d) or "layer" (nn.LayerNorm), placed after each hidden Linear.
    Store the layers in a single nn.Sequential called `self.net` (the tests and diagnostics rely on it).
    forward(x) returns raw LOGITS (no softmax).
    """

    def __init__(self, in_dim, hidden_dims, out_dim, act="relu", dropout=0.0, norm=None):
        super().__init__()
        raise NotImplementedError  # TODO: your code here

    def forward(self, x):
        raise NotImplementedError  # TODO: your code here


def init_weights(model: nn.Module, scheme: str = "he", std: float = 0.01) -> None:
    """Initialise every nn.Linear in `model` IN PLACE; biases are set to zero.

    scheme:
      "he"     weight ~ N(0, 2 / fan_in)                (Week 1.3, for ReLU)
      "xavier" weight ~ N(0, 2 / (fan_in + fan_out))    (for tanh / sigmoid)
      "normal" weight ~ N(0, std^2)
      "zeros"  weight = 0                               (shows why symmetry must be broken)
    Write the variance formulas yourself with weight.shape (out_features, in_features); do not call nn.init.kaiming_* / xavier_*.
    """
    raise NotImplementedError  # TODO: your code here


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Total number of parameters (only those with requires_grad=True if trainable_only)."""
    raise NotImplementedError  # TODO: your code here
