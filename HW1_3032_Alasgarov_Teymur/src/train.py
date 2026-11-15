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
    if step < warmup:
        return float(step + 1) / float(warmup)

    if total == warmup:
        progress = 1.0
    else:
        progress = min(max(float(step - warmup) / float(total - warmup), 0.0), 1.0)

    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of rows whose arg-max equals the label."""
    return float((logits.argmax(dim=-1) == y).float().mean().item())


def train_one_epoch(
    model, loader, optimizer, loss_fn=F.cross_entropy, scheduler=None, clip_norm=None
):
    """One pass over `loader`. Returns (mean_loss, mean_accuracy) over all examples seen."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X, y in loader:
        batch_size = X.shape[0]
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_examples += batch_size

    mean_loss = total_loss / total_examples if total_examples > 0 else 0.0
    mean_acc = total_correct / total_examples if total_examples > 0 else 0.0
    return mean_loss, mean_acc


@torch.no_grad()
def evaluate(model, loader, loss_fn=F.cross_entropy):
    """Return (mean_loss, mean_accuracy) in eval mode (dropout off, BatchNorm uses running statistics)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X, y in loader:
        batch_size = X.shape[0]
        logits = model(X)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_examples += batch_size

    mean_loss = total_loss / total_examples if total_examples > 0 else 0.0
    mean_acc = total_correct / total_examples if total_examples > 0 else 0.0
    return mean_loss, mean_acc


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    scheduler=None,
    clip_norm=None,
    patience=None,
    loss_fn=F.cross_entropy,
    verbose=False,
):
    """Train for up to `epochs`, return a history dict, and RESTORE the best weights (lowest val loss).

    history keys: train_loss, train_acc, val_loss, val_acc (one entry per epoch run) and best_epoch (0-based).
    Early stopping: if `patience` is not None, stop when val loss has not improved (by > 1e-6) for
    `patience` consecutive epochs. After the loop, load the best state with model.load_state_dict.
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 0,
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            clip_norm=clip_norm,
        )
        va_loss, va_acc = evaluate(model, val_loader, loss_fn=loss_fn)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Epoch {ep + 1}/{epochs}: train loss={tr_loss:.4f}, acc={tr_acc:.4f} | val loss={va_loss:.4f}, acc={va_acc:.4f}"
            )

        if patience is not None and epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    return history
