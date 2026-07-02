"""
Posterior variability score for OOD detection.

Quantifies disagreement between posteriors from N density estimators
retrieving over the same observation. If posteriors are highly variable,
the observation is potentially out-of-distribution.

The variability score is defined as:

    D_v({q_i}_{i=1}^{N_p}) = 1 / (N_p * (N_p - 1)) * sum_{i != j} D_KL(q_i || q_j)

High D_v → models disagree → data likely OOD.
Low D_v  → models agree → data likely in-distribution.
"""

from typing import List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm


# =========================================================
# RESULT OBJECT
# =========================================================

@dataclass(frozen=True)
class VariabilityResult:
    """Container for posterior variability analysis."""

    # Pairwise KL matrix: (N, N)
    kl_matrix: np.ndarray

    # Scalar variability score D_v
    variability_score: float

    # Per-parameter variability scores: (n_params,)
    per_param_scores: Optional[np.ndarray] = None

    # Figure
    figure: Optional[Any] = None


# =========================================================
# KL DIVERGENCE ESTIMATION
# =========================================================

def pairwise_kl_divergence(
    samples_list: List[torch.Tensor],
    method: str = "gaussian",
) -> np.ndarray:
    """
    Estimate pairwise KL divergence between posteriors.

    Args:
        samples_list: list of N tensors, each (n_samples, n_params)
                      representing samples from each posterior q_i
        method: estimation method
            - "gaussian": fit Gaussian to each posterior, compute KL analytically
            - "mc": Monte Carlo estimate using log_prob if available

    Returns:
        kl_matrix: (N, N) array where kl_matrix[i, j] = D_KL(q_i || q_j)
    """

    n_posteriors = len(samples_list)
    n_params = samples_list[0].shape[-1]

    if method == "gaussian":
        # Fit Gaussian (mean, cov) to each posterior
        means = []
        covs = []
        for samples in samples_list:
            s = samples.double()
            mean = s.mean(dim=0)
            centered = s - mean
            cov = (centered.T @ centered) / (len(s) - 1)
            # Regularize for numerical stability
            cov = cov + 1e-6 * torch.eye(n_params, dtype=torch.float64)
            means.append(mean)
            covs.append(cov)

        # Compute pairwise KL for Gaussians
        # D_KL(N_i || N_j) = 0.5 * (tr(Σ_j^{-1} Σ_i) + (μ_j - μ_i)^T Σ_j^{-1} (μ_j - μ_i)
        #                     - n + ln(det(Σ_j) / det(Σ_i)))
        kl_matrix = np.zeros((n_posteriors, n_posteriors))

        for i in range(n_posteriors):
            for j in range(n_posteriors):
                if i == j:
                    continue

                mu_i, cov_i = means[i], covs[i]
                mu_j, cov_j = means[j], covs[j]

                cov_j_inv = torch.linalg.inv(cov_j)
                diff = mu_j - mu_i

                trace_term = torch.trace(cov_j_inv @ cov_i)
                quad_term = diff @ cov_j_inv @ diff
                logdet_term = torch.logdet(cov_j) - torch.logdet(cov_i)

                kl = 0.5 * (trace_term + quad_term - n_params + logdet_term)
                kl_matrix[i, j] = kl.item()

        return kl_matrix

    elif method == "mc":
        # Monte Carlo KL estimation using kernel density or sample-based approach
        # D_KL(q_i || q_j) ≈ (1/M) * sum_m [log q_i(x_m) - log q_j(x_m)]
        # where x_m ~ q_i
        # Using k-NN density estimation for log q

        from scipy.spatial import KDTree

        kl_matrix = np.zeros((n_posteriors, n_posteriors))

        # Build KD-trees for each posterior
        trees = [KDTree(s.numpy()) for s in samples_list]
        n_samples = samples_list[0].shape[0]
        k = min(5, n_samples - 1)  # k-NN parameter

        for i in range(n_posteriors):
            for j in range(n_posteriors):
                if i == j:
                    continue

                x = samples_list[i].numpy()

                # k-NN distances from q_i samples to q_i and q_j
                d_ii, _ = trees[i].query(x, k=k + 1)  # +1 because self is included
                d_ij, _ = trees[j].query(x, k=k)

                # Take k-th neighbor distance (exclude self for d_ii)
                rho_i = d_ii[:, -1]  # k-th neighbor in q_i (excluding self)
                rho_j = d_ij[:, -1]  # k-th neighbor in q_j

                # KL estimate: (d/n) * sum(log(rho_j / rho_i)) + log(n_j / (n_i - 1))
                # where d = dimension
                valid = (rho_i > 0) & (rho_j > 0)
                if valid.sum() > 0:
                    kl_est = (
                        n_params * np.mean(np.log(rho_j[valid] / rho_i[valid]))
                        + np.log(n_samples / (n_samples - 1))
                    )
                    kl_matrix[i, j] = max(kl_est, 0.0)  # KL >= 0

        return kl_matrix

    else:
        raise ValueError(f"Unknown method: {method}. Use 'gaussian' or 'mc'.")


def pairwise_kl_per_param(
    samples_list: List[torch.Tensor],
) -> np.ndarray:
    """
    Compute per-parameter pairwise KL divergence (1D Gaussian assumption).

    Args:
        samples_list: list of N tensors, each (n_samples, n_params)

    Returns:
        per_param_kl: (N, N, n_params) array
    """

    n_posteriors = len(samples_list)
    n_params = samples_list[0].shape[-1]

    means = [s.mean(dim=0).numpy() for s in samples_list]
    stds = [s.std(dim=0).numpy() for s in samples_list]

    per_param_kl = np.zeros((n_posteriors, n_posteriors, n_params))

    for i in range(n_posteriors):
        for j in range(n_posteriors):
            if i == j:
                continue
            # 1D Gaussian KL: log(σ_j/σ_i) + (σ_i² + (μ_i - μ_j)²) / (2σ_j²) - 0.5
            mu_i, sig_i = means[i], stds[i]
            mu_j, sig_j = means[j], stds[j]

            kl = (
                np.log(sig_j / sig_i)
                + (sig_i**2 + (mu_i - mu_j) ** 2) / (2 * sig_j**2)
                - 0.5
            )
            per_param_kl[i, j] = np.maximum(kl, 0.0)

    return per_param_kl


# =========================================================
# VARIABILITY SCORE
# =========================================================

def posterior_variability_score(kl_matrix: np.ndarray) -> float:
    """
    Compute the posterior variability score D_v.

    D_v = 1 / (N_p * (N_p - 1)) * sum_{i != j} D_KL(q_i || q_j)

    Args:
        kl_matrix: (N, N) pairwise KL divergence matrix

    Returns:
        D_v: scalar variability score
    """
    n = kl_matrix.shape[0]
    if n < 2:
        return 0.0

    # Sum off-diagonal elements
    total = kl_matrix.sum() - np.trace(kl_matrix)
    dv = total / (n * (n - 1))

    return dv


# =========================================================
# FULL OOD SCORE COMPUTATION
# =========================================================

def compute_ood_score(
    base_models: list,
    x_obs: torch.Tensor,
    n_samples: int = 2048,
    method: str = "gaussian",
    device: str = "cuda",
) -> VariabilityResult:
    """
    Compute posterior variability score for OOD detection.

    Runs all base models on x_obs, collects posterior samples,
    computes pairwise KL, and returns the variability score.

    Args:
        base_models: list of frozen EstimatorBase instances
        x_obs: (1, D_obs) observation tensor
        n_samples: number of posterior samples per model
        method: "gaussian" or "mc" for KL estimation
        device: computation device

    Returns:
        VariabilityResult with kl_matrix, variability_score, per_param_scores
    """

    x_obs = x_obs.to(device)
    samples_list = []

    with torch.no_grad():
        for model in base_models:
            x_emb = model.embedding(x_obs)
            posterior = model.flow.flow(x_emb)
            samples = posterior.sample((n_samples,)).squeeze(1).cpu()
            samples_list.append(samples)

    # Full KL matrix
    kl_matrix = pairwise_kl_divergence(samples_list, method=method)
    dv = posterior_variability_score(kl_matrix)

    # Per-parameter scores
    per_param_kl = pairwise_kl_per_param(samples_list)
    # Average off-diagonal per parameter
    n = len(samples_list)
    mask = ~np.eye(n, dtype=bool)
    per_param_scores = per_param_kl[mask].reshape(-1, per_param_kl.shape[-1]).mean(axis=0)

    return VariabilityResult(
        kl_matrix=kl_matrix,
        variability_score=dv,
        per_param_scores=per_param_scores,
    )


# =========================================================
# PLOTTING
# =========================================================

def plot_variability(
    result: VariabilityResult,
    param_names: Optional[List[str]] = None,
    figsize=(12, 5),
) -> plt.Figure:
    """
    Plot the posterior variability analysis.

    Left: KL divergence heatmap between models.
    Right: Per-parameter variability bar chart.

    Args:
        result: VariabilityResult from compute_ood_score
        param_names: optional list of parameter names for labels
        figsize: figure size

    Returns:
        matplotlib Figure
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- KL heatmap ---
    n = result.kl_matrix.shape[0]
    im = ax1.imshow(result.kl_matrix, cmap="Reds", aspect="equal")
    ax1.set_xlabel("Model j")
    ax1.set_ylabel("Model i")
    ax1.set_title(
        f"Pairwise $D_{{KL}}(q_i || q_j)$\n"
        f"$D_v = {result.variability_score:.3f}$"
    )
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels([f"M{i+1}" for i in range(n)])
    ax1.set_yticklabels([f"M{i+1}" for i in range(n)])
    plt.colorbar(im, ax=ax1)

    # --- Per-parameter bar chart ---
    if result.per_param_scores is not None:
        n_params = len(result.per_param_scores)
        x_pos = np.arange(n_params)

        if param_names is None:
            param_names = [f"p{i}" for i in range(n_params)]

        ax2.bar(x_pos, result.per_param_scores, color="steelblue", alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel(r"Mean $D_{KL}$")
        ax2.set_title("Per-parameter variability")
        ax2.axhline(
            result.variability_score, color="red",
            linestyle="--", linewidth=1, label=f"$D_v = {result.variability_score:.3f}$"
        )
        ax2.legend()

    fig.tight_layout()

    return fig
