"""Bonus (optional): Adam from scratch as a torch.optim.Optimizer, and MixUp."""
import torch


class MyAdam(torch.optim.Optimizer):
    """Adam with bias correction (Kingma & Ba), written as an Optimizer subclass.

        m = b1 m + (1-b1) g          v = b2 v + (1-b2) g^2
        m_hat = m / (1 - b1^t)       v_hat = v / (1 - b2^t)
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)
    Keep per-parameter state in self.state[p] ("step", "m", "v"). Must match torch.optim.Adam.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError  # TODO: your code here


def mixup_batch(x: torch.Tensor, y_onehot: torch.Tensor, alpha: float, generator: torch.Generator = None):
    """MixUp: lam ~ Beta(alpha, alpha); mix each example with a random partner from the same batch.

    Return (x_mixed, y_mixed) with x_mixed = lam x + (1-lam) x[perm] and the same mix for the labels.
    Sample lam with torch.distributions.Beta and the permutation with torch.randperm(generator=generator).
    """
    raise NotImplementedError  # TODO: your code here
