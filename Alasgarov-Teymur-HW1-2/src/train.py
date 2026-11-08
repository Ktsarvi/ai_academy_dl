"""Part B4 -- The training loop pieces: schedule, one epoch, evaluation, and fit with early stopping.

Conventions (these are the habits of Week 2 and the Week-2 reading):
  * model.train() while updating, model.eval() + torch.no_grad() while evaluating.
  * optimizer.zero_grad(set_to_none=True) -> loss.backward() -> (clip) -> optimizer.step() -> scheduler.step()
    The scheduler is stepped once per BATCH, not per epoch.
  * Losses and accuracies are averaged over EXAMPLES (weight each batch by its size).
"""
import copy
import math
import torch
import torch.nn.functional as F


def warmup_cosine(step: int, warmup: int, total: int, min_factor: float = 0.0) -> float:
    """Learning-rate MULTIPLIER in [min_factor, 1] for a linear warm-up followed by a cosine decay.

        step < warmup:  (step + 1) / warmup
        otherwise:      min_factor + (1 - min_factor) * 0.5 * (1 + cos(pi * progress)),
                        progress = (step - warmup) / (total - warmup), clipped to [0, 1]
    """
    raise NotImplementedError  # TODO: your code here


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of rows whose arg-max equals the label."""
    raise NotImplementedError  # TODO: your code here


def train_one_epoch(model, loader, optimizer, loss_fn=F.cross_entropy, scheduler=None, clip_norm=None):
    """One pass over `loader`. Returns (mean_loss, mean_accuracy) over all examples seen."""
    raise NotImplementedError  # TODO: your code here


@torch.no_grad()
def evaluate(model, loader, loss_fn=F.cross_entropy):
    """Return (mean_loss, mean_accuracy) in eval mode (dropout off, BatchNorm uses running statistics)."""
    raise NotImplementedError  # TODO: your code here


def fit(model, train_loader, val_loader, optimizer, epochs, scheduler=None, clip_norm=None,
        patience=None, loss_fn=F.cross_entropy, verbose=False):
    """Train for up to `epochs`, return a history dict, and RESTORE the best weights (lowest val loss).

    history keys: train_loss, train_acc, val_loss, val_acc (one entry per epoch run) and best_epoch (0-based).
    Early stopping: if `patience` is not None, stop when val loss has not improved (by > 1e-6) for
    `patience` consecutive epochs. After the loop, load the best state with model.load_state_dict.
    """
    raise NotImplementedError  # TODO: your code here
