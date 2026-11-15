"""Part B3 -- An MLP as an nn.Module, with configurable activation, normalisation, dropout and init."""

import math

from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
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

    def __init__(
        self, in_dim, hidden_dims, out_dim, act="relu", dropout=0.0, norm=None
    ):
        super().__init__()
        if act not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {act}")
        if norm not in (None, "batch", "layer"):
            raise ValueError(f"Unknown norm: {norm}")

        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(ACTIVATIONS[act]())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def init_weights(model: nn.Module, scheme: str = "he", std: float = 0.01) -> None:
    """Initialise every nn.Linear in `model` IN PLACE; biases are set to zero.

    scheme:
      "he"     weight ~ N(0, 2 / fan_in)                (Week 1.3, for ReLU)
      "xavier" weight ~ N(0, 2 / (fan_in + fan_out))    (for tanh / sigmoid)
      "normal" weight ~ N(0, std^2)
      "zeros"  weight = 0                               (shows why symmetry must be broken)
    Write the variance formulas yourself with weight.shape (out_features, in_features); do not call nn.init.kaiming_* / xavier_*.
    """
    valid_schemes = {"he", "xavier", "normal", "zeros"}
    if scheme not in valid_schemes:
        raise ValueError(f"Unknown scheme: {scheme}")

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            fan_out, fan_in = mod.weight.shape
            if scheme == "he":
                mod.weight.data.normal_(0.0, math.sqrt(2.0 / fan_in))
            elif scheme == "xavier":
                mod.weight.data.normal_(0.0, math.sqrt(2.0 / (fan_in + fan_out)))
            elif scheme == "normal":
                mod.weight.data.normal_(0.0, std)
            elif scheme == "zeros":
                mod.weight.data.zero_()

            if mod.bias is not None:
                mod.bias.data.zero_()


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Total number of parameters (only those with requires_grad=True if trainable_only)."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
