"""Part B1 -- Tensor basics and the Week-1 losses, written with raw tensor operations.

Rules for this file: use torch tensor operations (indexing, broadcasting, matmul,
logsumexp, exp, log, ...). Do NOT call torch.nn.functional losses, torch.softmax,
torch.log_softmax or torch.nn.functional.one_hot inside your answers -- the tests
compare your results with those reference functions.
"""

import torch


def standardize(
    X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Return (X - mean) / (std + eps). `mean` and `std` have shape (d,) and X has shape (n, d)."""
    return (X - mean) / (std + eps)


def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Integer labels (n,) -> float32 one-hot matrix (n, num_classes). Hint: zeros + scatter_ or indexing."""
    out = torch.zeros(y.shape[0], num_classes, dtype=torch.float32, device=y.device)
    return out.scatter_(1, y.unsqueeze(1), 1.0)


def pairwise_sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between rows of A (n, d) and rows of B (m, d) -> (n, m).

    No Python loops. Hint: ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2 with broadcasting,
    or insert size-1 axes and subtract. Clamp tiny negatives to 0 if you use the expansion.
    """
    return (A.unsqueeze(1) - B.unsqueeze(0)).pow(2).sum(dim=-1)


def softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax: subtract the max along `dim` first. Must not overflow for z ~ 1000."""
    max_z = torch.max(z, dim=dim, keepdim=True).values
    exp_z = torch.exp(z - max_z)
    return exp_z / torch.sum(exp_z, dim=dim, keepdim=True)


def log_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Stable log-softmax via the log-sum-exp trick: z - logsumexp(z). You may use torch.logsumexp."""
    return z - torch.logsumexp(z, dim=dim, keepdim=True)


def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean categorical cross-entropy from raw logits (n, K) and integer labels (n,) -> scalar.

    L = -(1/n) sum_i log softmax(logits_i)[y_i]. Use your own log_softmax and fancy indexing/gather.
    """
    log_p = log_softmax(logits, dim=-1)
    n = logits.shape[0]
    return -log_p[torch.arange(n, device=logits.device), y].mean()


def bce_with_logits(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy from logits z (n,) and float targets y in {0,1} (n,) -> scalar.

    Stable form (never exponentiates a large positive number):
        loss_i = max(z_i, 0) - z_i * y_i + log(1 + exp(-|z_i|))
    Derive why this equals -[y log sigmoid(z) + (1-y) log(1-sigmoid(z))] (Part A2).
    """
    loss = torch.clamp(z, min=0.0) - z * y + torch.log1p(torch.exp(-torch.abs(z)))
    return loss.mean()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error (mean over every element). pred and target must have the same shape."""
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    return torch.mean((pred - target) ** 2)
