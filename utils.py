"""
Helper functions utilized during training in train.py.
"""

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(train_loss, val_loss, save_path):
    """Plot and save train/val loss curves to `loss.png`.

    Args:
        train_loss: Sequence of per-epoch training losses.
        val_loss: Sequence of per-epoch validation losses.
        save_path: Directory where `loss.png` will be written.
    """
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss.png"))
    plt.close()


def create_experiment_dir(experiment_num: int) -> str:
    """Create and return an experiment directory under `experiments/`.
    For each run, experiments will have a run#/ folder, which contains that run's loss curves and checkpoints.

    Example:
        `experiment_num=6` -> `experiments/run6`
    """

    save_path = os.path.join("experiments", f"run{experiment_num}")
    os.makedirs(save_path, exist_ok=True)
    return save_path

def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
