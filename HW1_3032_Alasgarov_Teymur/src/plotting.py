"""PROVIDED -- do not edit. Small plotting helpers for the notebook."""
import matplotlib.pyplot as plt


def plot_histories(histories: dict, key: str = "val_loss", ylabel: str = None, ax=None, title: str = None):
    """Overlay one curve per run. `histories` maps a label to the dict returned by train.fit."""
    ax = ax or plt.gca()
    for name, h in histories.items():
        ax.plot(h[key], label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel or key)
    if title: ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    return ax


def plot_layer_stats(stats_by_run: dict, key: str, ax=None, logy: bool = True, title: str = None):
    """stats_by_run: label -> list of dicts from diagnostics.layer_stats. Plots `key` per layer."""
    ax = ax or plt.gca()
    for name, stats in stats_by_run.items():
        ax.plot(range(1, len(stats) + 1), [s[key] for s in stats], marker="o", label=name)
    ax.set_xlabel("layer (1 = first)"); ax.set_ylabel(key)
    if logy: ax.set_yscale("log")
    if title: ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    return ax
