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
    if act == "tanh":
        return 1.0 - a**2
    elif act == "relu":
        return (z > 0.0).to(z.dtype)
    else:
        raise ValueError(f"Unknown activation: {act}")


def forward(params: dict, X: torch.Tensor, act: str = "tanh"):
    """Return (logits, cache). `cache` is a dict holding what the backward pass needs: X, z1, a1."""
    z1 = X @ params["W1"] + params["b1"]
    a1 = act_forward(z1, act)
    logits = a1 @ params["W2"] + params["b2"]
    cache = {"X": X, "z1": z1, "a1": a1}
    return logits, cache


def backward(
    params: dict, cache: dict, logits: torch.Tensor, y: torch.Tensor, act: str = "tanh"
) -> dict:
    """Gradients of the MEAN cross-entropy loss w.r.t. W1, b1, W2, b2 (dict with those four keys).

    Steps (n = batch size, Y = one-hot of y):
      delta2 = (softmax(logits) - Y) / n                     # dL/dz2   (Week 2.1, softmax + CE)
      gW2 = a1^T delta2         gb2 = delta2 summed over the batch
      delta1 = (delta2 @ W2^T) * act'(z1)                    # one layer back
      gW1 = X^T delta1          gb1 = delta1 summed over the batch
    """
    max_l = torch.max(logits, dim=-1, keepdim=True).values
    exp_l = torch.exp(logits - max_l)
    probs = exp_l / torch.sum(exp_l, dim=-1, keepdim=True)

    Y = torch.zeros_like(probs)
    Y.scatter_(1, y.unsqueeze(1), 1.0)

    n = logits.shape[0]
    delta2 = (probs - Y) / n
    gW2 = cache["a1"].T @ delta2
    gb2 = delta2.sum(dim=0)
    delta1 = (delta2 @ params["W2"].T) * act_derivative(cache["z1"], cache["a1"], act)
    gW1 = cache["X"].T @ delta1
    gb1 = delta1.sum(dim=0)

    return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}


def numerical_grad(f, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Central finite-difference gradient of the scalar function f at tensor x (float64).

    For every entry i: (f(x + eps e_i) - f(x - eps e_i)) / (2 eps). `f` takes the whole tensor and
    returns a scalar. Do not modify x permanently (restore each entry, or work on a clone).
    """
    grad = torch.zeros_like(x)
    x_copy = x.clone()
    flat_x = x_copy.view(-1)
    flat_grad = grad.view(-1)
    for i in range(flat_x.numel()):
        orig = flat_x[i].item()
        flat_x[i] = orig + eps
        f_plus = f(x_copy)
        if isinstance(f_plus, torch.Tensor):
            f_plus = f_plus.item()
        flat_x[i] = orig - eps
        f_minus = f(x_copy)
        if isinstance(f_minus, torch.Tensor):
            f_minus = f_minus.item()
        flat_grad[i] = (f_plus - f_minus) / (2.0 * eps)
        flat_x[i] = orig
    return grad


class SwishFn(torch.autograd.Function):
    """Custom autograd Function for swish(x) = x * sigmoid(x).

    d/dx swish = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
    Save what you need in forward with ctx.save_for_backward and return grad_output * derivative.
    """

    @staticmethod
    def forward(ctx, x):
        sig_x = torch.sigmoid(x)
        ctx.save_for_backward(x, sig_x)
        return x * sig_x

    @staticmethod
    def backward(ctx, grad_output):
        x, sig_x = ctx.saved_tensors
        dx = sig_x + x * sig_x * (1.0 - sig_x)
        return grad_output * dx


swish = SwishFn.apply
