"""Bonus (optional): Adam from scratch as a torch.optim.Optimizer, and MixUp."""

import math

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
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MyAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["v"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                step = state["step"]
                m = state["m"]
                v = state["v"]

                m.mul_(b1).add_(grad, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)

                bias_correction1 = 1.0 - b1**step
                bias_correction2 = 1.0 - b2**step

                step_size = lr / bias_correction1
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(m, denom, value=-step_size)

        return loss


def mixup_batch(
    x: torch.Tensor,
    y_onehot: torch.Tensor,
    alpha: float,
    generator: torch.Generator = None,
):
    """MixUp: lam ~ Beta(alpha, alpha); mix each example with a random partner from the same batch.

    Return (x_mixed, y_mixed) with x_mixed = lam x + (1-lam) x[perm] and the same mix for the labels.
    Sample lam with torch.distributions.Beta and the permutation with torch.randperm(generator=generator).
    """
    lam = torch.distributions.Beta(alpha, alpha).sample()
    perm = torch.randperm(x.size(0), generator=generator, device=x.device)

    x_mixed = lam * x + (1.0 - lam) * x[perm]
    y_mixed = lam * y_onehot + (1.0 - lam) * y_onehot[perm]
    return x_mixed, y_mixed
