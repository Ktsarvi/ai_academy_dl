"""PROVIDED -- do not edit. Data loading, seeding and batching helpers.

The dataset is scikit-learn's 8x8 digits (1797 images, 10 classes, 64 features):
it is built in, so there is nothing to download and everything runs on a CPU.
"""
import os
import numpy as np
import torch
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset

# Tiny networks run FASTEST on a single CPU thread: with many threads PyTorch spends its time synchronising
# (we measured 0.04 s vs 8 s per epoch on a 16-core machine). Set DL_HW_THREADS=N to override.
torch.set_num_threads(int(os.environ.get("DL_HW_THREADS", "1")))


def student_seed(default: int = 0) -> int:
    """Seed from the STUDENT_ID environment variable (digits only), else `default`."""
    sid = "".join(ch for ch in os.environ.get("STUDENT_ID", "") if ch.isdigit())
    return int(sid) % (2**31 - 1) if sid else default


def set_seed(seed: int) -> None:
    """Seed Python-side RNGs that PyTorch code commonly touches."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_digits_splits(seed: int = 0, n_train: int = 1000, n_val: int = 300):
    """Return dict with float32 X* and int64 y* tensors, standardised with TRAIN statistics.

    Split: n_train / n_val / rest (test). `n_train` can be reduced to create an
    overfitting-prone small-data setting (Part B5d).
    """
    X, y = load_digits(return_X_y=True)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    X = torch.tensor(X, dtype=torch.float32)[perm]
    y = torch.tensor(y, dtype=torch.int64)[perm]
    # The validation and test slices are fixed in size, so shrinking n_train only
    # shrinks the training set.
    Xva, yva = X[1000:1300][:n_val], y[1000:1300][:n_val]
    Xte, yte = X[1300:], y[1300:]
    Xtr, ytr = X[:n_train], y[:n_train]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)   # a pixel that is constant in TRAIN has std 0: leave it unscaled,
    f = lambda A: (A - mu) / sd                            # otherwise val/test values blow up by 1e8. Train statistics only.
    return dict(Xtr=f(Xtr), ytr=ytr, Xva=f(Xva), yva=yva, Xte=f(Xte), yte=yte)


def make_loader(X, y, batch_size: int = 64, shuffle: bool = True, seed: int = 0) -> DataLoader:
    """DataLoader over in-memory tensors, with a seeded shuffle generator."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle, generator=g)


def corrupt_labels(y: torch.Tensor, frac: float = 0.25, num_classes: int = 10, seed: int = 0) -> torch.Tensor:
    """Return a copy of y in which a fraction `frac` of the labels is replaced by a DIFFERENT random class.

    Used in Part B5d: noisy labels cannot be learned, only memorised, so overfitting becomes visible.
    """
    g = torch.Generator().manual_seed(seed)
    y = y.clone()
    idx = torch.randperm(len(y), generator=g)[: int(frac * len(y))]
    y[idx] = (y[idx] + torch.randint(1, num_classes, (len(idx),), generator=g)) % num_classes
    return y
