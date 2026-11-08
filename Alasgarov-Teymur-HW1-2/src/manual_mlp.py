"""Part B2 -- Backpropagation by hand in PyTorch, and a custom autograd Function.

A two-layer MLP in batch-first convention:
    z1 = X @ W1 + b1          (n, h)
    a1 = act(z1)              (n, h)       act in {"tanh", "relu"}
    z2 = a1 @ W2 + b2         (n, K)       logits
    L  = mean cross-entropy(z2, y)
You compute the gradients with the backpropagation equations of Week 2 -- NOT with .backward().
The tests compare your gradients with torch.autograd and with finite differences.
"""
import torch


def init_params(d: int, h: int, K: int, seed: int = 0, dtype=torch.float64) -> dict:
    """PROVIDED. He-style random weights, zero biases, as a dict of tensors."""
    g = torch.Generator().manual_seed(seed)
    return {
        "W1": torch.randn(d, h, generator=g, dtype=dtype) * (2.0 / d) ** 0.5,
        "b1": torch.zeros(h, dtype=dtype),
        "W2": torch.randn(h, K, generator=g, dtype=dtype) * (2.0 / h) ** 0.5,
        "b2": torch.zeros(K, dtype=dtype),
    }


def act_forward(z: torch.Tensor, act: str) -> torch.Tensor:
    """PROVIDED."""
    return torch.tanh(z) if act == "tanh" else torch.clamp(z, min=0.0)


def act_derivative(z: torch.Tensor, a: torch.Tensor, act: str) -> torch.Tensor:
    """TODO: derivative f'(z) of the activation. `a = f(z)` is passed in so tanh can reuse it.

    tanh: 1 - a^2        relu: 1 where z > 0 else 0
    """
    raise NotImplementedError  # TODO: your code here


def forward(params: dict, X: torch.Tensor, act: str = "tanh"):
    """Return (logits, cache). `cache` is a dict holding what the backward pass needs: X, z1, a1."""
    raise NotImplementedError  # TODO: your code here


def backward(params: dict, cache: dict, logits: torch.Tensor, y: torch.Tensor, act: str = "tanh") -> dict:
    """Gradients of the MEAN cross-entropy loss w.r.t. W1, b1, W2, b2 (dict with those four keys).

    Steps (n = batch size, Y = one-hot of y):
      delta2 = (softmax(logits) - Y) / n                     # dL/dz2   (Week 2.1, softmax + CE)
      gW2 = a1^T delta2         gb2 = delta2 summed over the batch
      delta1 = (delta2 @ W2^T) * act'(z1)                    # one layer back
      gW1 = X^T delta1          gb1 = delta1 summed over the batch
    """
    raise NotImplementedError  # TODO: your code here


def numerical_grad(f, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Central finite-difference gradient of the scalar function f at tensor x (float64).

    For every entry i: (f(x + eps e_i) - f(x - eps e_i)) / (2 eps). `f` takes the whole tensor and
    returns a scalar. Do not modify x permanently (restore each entry, or work on a clone).
    """
    raise NotImplementedError  # TODO: your code here


class SwishFn(torch.autograd.Function):
    """Custom autograd Function for swish(x) = x * sigmoid(x).

    d/dx swish = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
    Save what you need in forward with ctx.save_for_backward and return grad_output * derivative.
    """

    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError  # TODO: your code here

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError  # TODO: your code here


swish = SwishFn.apply
