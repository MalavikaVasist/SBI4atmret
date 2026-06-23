from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import islice
import pandas as pd


# =========================================================
# RESULT OBJECT
# =========================================================

@dataclass(frozen=True)
class CoverageResult:

    ranks: Optional[np.ndarray]   # None if loaded from disk

    coverage: np.ndarray

    alpha: np.ndarray

    figure: Optional[Any] = None

    save_path: Optional[Path] = None


# =========================================================
# COMPUTE FUNCTION
# =========================================================

def compute_coverage(
    net,
    batch_processor,
    test_loaders,
    test_keys,
    n_batches: int = 128,
    n_samples: int = 1024,
):
    """
    Compute SBC ranks and empirical coverage curve.

    Args:
        net: model with .flow(x) method
        batch_processor: BatchProcessor instance
        test_loaders: iterable of test data loaders
        test_keys: keys for batch_processor.prepare_batch
        n_batches: number of test batches
        n_samples: posterior samples per observation for ranking

    Returns:
        (ranks, coverage, alpha) as numpy arrays
    """

    ranks = []

    with torch.no_grad():
        for batches in islice(zip(*test_loaders), n_batches):
            theta, x = batch_processor.prepare_batch(batches, test_keys)

            posterior = net.flow(x)
            samples = posterior.sample((n_samples,))
            log_p = posterior.log_prob(theta)
            log_p_samples = posterior.log_prob(samples)

            ranks.append(
                (log_p_samples < log_p).float().mean(dim=0).cpu()
            )

    ranks = torch.cat(ranks).numpy()
    alpha = np.linspace(0, 1, 100)
    sorted_ranks = np.sort(ranks.flatten())

    coverage = np.array([
        (sorted_ranks > (1 - a)).mean()
        for a in alpha
    ])

    return ranks, coverage, alpha


# =========================================================
# PLOT FUNCTION
# =========================================================

def plot_coverage(coverage, alpha):
    """
    Plot empirical coverage vs ideal diagonal.
    """

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(alpha, coverage, color="steelblue", label="Empirical")
    ax.plot([0, 1], [0, 1], color="k", linestyle="--", label="Ideal")

    ax.set_xlabel(r"Credibility level $1-\alpha$", fontsize=12)
    ax.set_ylabel(r"Coverage probability", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=12)

    return fig
