"""
Embedding-space OOD detection.

Compares the embedding of a target observation against embeddings
from a test/training set using:
1. PCA projection and visualization
2. Mahalanobis distance in embedding space
3. k-NN distance to detect outliers

If the observation's embedding is far from the training distribution,
it is likely OOD.
"""

from dataclasses import dataclass
from typing import Optional, Any, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm


# =========================================================
# RESULT OBJECT
# =========================================================

@dataclass(frozen=True)
class EmbeddingAnalysisResult:
    """Container for embedding-space OOD analysis."""

    # Embeddings of the test set: (N_test, emb_dim)
    test_embeddings: np.ndarray

    # Embedding of the observation: (1, emb_dim)
    obs_embedding: np.ndarray

    # PCA-projected test embeddings: (N_test, n_components)
    test_pca: np.ndarray

    # PCA-projected observation: (1, n_components)
    obs_pca: np.ndarray

    # Mahalanobis distance of observation from test distribution
    mahalanobis_distance: float

    # k-NN distance of observation to test set
    knn_distance: float

    # PCA object for reuse
    pca_model: Any = None

    figure: Optional[Any] = None


# =========================================================
# EXTRACT EMBEDDINGS
# =========================================================

def extract_embeddings(
    model,
    dataloader,
    batch_processor,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract embeddings for an entire dataset.

    Args:
        model: EstimatorBase instance (with .embedding attribute)
        dataloader: data loader yielding batches
        batch_processor: BatchProcessor for preparing batches
        device: computation device
        max_batches: limit number of batches (None = all)

    Returns:
        embeddings: (N, emb_dim) tensor
    """

    model.embedding.eval()
    all_embeddings = []

    with torch.no_grad():
        for i, batches in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            if max_batches is not None and i >= max_batches:
                break

            _, x = batch_processor.prepare_batch(batches)
            emb = model.embedding(x)
            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


def extract_obs_embedding(
    model,
    x_obs: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract embedding for a single observation.

    Args:
        model: EstimatorBase instance
        x_obs: (1, D_obs) observation tensor

    Returns:
        embedding: (1, emb_dim) tensor
    """

    model.embedding.eval()

    with torch.no_grad():
        emb = model.embedding(x_obs.to(device))

    return emb.cpu()


# =========================================================
# PCA ANALYSIS
# =========================================================

def compute_pca(
    test_embeddings: np.ndarray,
    obs_embedding: np.ndarray,
    n_components: int = 2,
) -> tuple:
    """
    Fit PCA on test embeddings and project both test and observation.

    Args:
        test_embeddings: (N, emb_dim)
        obs_embedding: (1, emb_dim)
        n_components: PCA dimensions

    Returns:
        (test_pca, obs_pca, pca_model)
    """

    pca = PCA(n_components=n_components)
    test_pca = pca.fit_transform(test_embeddings)
    obs_pca = pca.transform(obs_embedding)

    return test_pca, obs_pca, pca


# =========================================================
# DISTANCE METRICS
# =========================================================

def mahalanobis_distance(
    test_embeddings: np.ndarray,
    obs_embedding: np.ndarray,
) -> float:
    """
    Compute Mahalanobis distance of observation from the test distribution.

    Args:
        test_embeddings: (N, D)
        obs_embedding: (1, D)

    Returns:
        scalar Mahalanobis distance
    """

    mean = test_embeddings.mean(axis=0)
    centered = test_embeddings - mean
    cov = (centered.T @ centered) / (len(test_embeddings) - 1)

    # Regularize
    cov += 1e-6 * np.eye(cov.shape[0])

    cov_inv = np.linalg.inv(cov)
    diff = obs_embedding.flatten() - mean
    dist = np.sqrt(diff @ cov_inv @ diff)

    return float(dist)


def knn_distance(
    test_embeddings: np.ndarray,
    obs_embedding: np.ndarray,
    k: int = 5,
) -> float:
    """
    Compute mean k-NN distance of observation to the test set.

    Args:
        test_embeddings: (N, D)
        obs_embedding: (1, D)
        k: number of nearest neighbors

    Returns:
        mean distance to k nearest neighbors
    """
    from scipy.spatial import KDTree

    tree = KDTree(test_embeddings)
    dists, _ = tree.query(obs_embedding.flatten(), k=k)

    return float(np.mean(dists))


# =========================================================
# FULL ANALYSIS
# =========================================================

def analyze_embeddings(
    model,
    x_obs: torch.Tensor,
    dataloader,
    batch_processor,
    n_components: int = 2,
    k: int = 5,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> EmbeddingAnalysisResult:
    """
    Full embedding-space OOD analysis.

    Args:
        model: EstimatorBase instance
        x_obs: (1, D_obs) observation
        dataloader: test/train data loader
        batch_processor: BatchProcessor
        n_components: PCA dimensions
        k: k for k-NN distance
        device: computation device
        max_batches: limit batches for speed

    Returns:
        EmbeddingAnalysisResult
    """

    # Extract embeddings
    test_emb = extract_embeddings(
        model, dataloader, batch_processor,
        device=device, max_batches=max_batches,
    ).numpy()

    obs_emb = extract_obs_embedding(model, x_obs, device=device).numpy()

    # PCA
    test_pca, obs_pca, pca_model = compute_pca(
        test_emb, obs_emb, n_components=n_components
    )

    # Distances
    maha_dist = mahalanobis_distance(test_emb, obs_emb)
    knn_dist = knn_distance(test_emb, obs_emb, k=k)

    return EmbeddingAnalysisResult(
        test_embeddings=test_emb,
        obs_embedding=obs_emb,
        test_pca=test_pca,
        obs_pca=obs_pca,
        mahalanobis_distance=maha_dist,
        knn_distance=knn_dist,
        pca_model=pca_model,
    )


# =========================================================
# PLOTTING
# =========================================================

def plot_embedding_pca(
    result: EmbeddingAnalysisResult,
    figsize=(10, 5),
    title: str = "Embedding PCA",
) -> plt.Figure:
    """
    Plot PCA projection of embeddings with observation highlighted.

    Left: 2D scatter of test embeddings + observation.
    Right: histogram of distances from centroid with observation marked.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- PCA scatter ---
    ax1.scatter(
        result.test_pca[:, 0], result.test_pca[:, 1],
        alpha=0.3, s=10, color="steelblue", label="Test set",
    )
    ax1.scatter(
        result.obs_pca[:, 0], result.obs_pca[:, 1],
        s=150, color="red", marker="*", zorder=5, label="Observation",
    )
    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.set_title(title)
    ax1.legend()

    # --- Distance histogram ---
    centroid = result.test_embeddings.mean(axis=0)
    test_dists = np.linalg.norm(result.test_embeddings - centroid, axis=1)
    obs_dist = np.linalg.norm(result.obs_embedding.flatten() - centroid)

    ax2.hist(test_dists, bins=50, alpha=0.7, color="steelblue", label="Test set")
    ax2.axvline(
        obs_dist, color="red", linewidth=2, linestyle="--",
        label=f"Obs (d={obs_dist:.2f})",
    )
    ax2.set_xlabel("L2 distance from centroid")
    ax2.set_ylabel("Count")
    ax2.set_title(
        f"Mahalanobis: {result.mahalanobis_distance:.2f} | "
        f"k-NN: {result.knn_distance:.2f}"
    )
    ax2.legend()

    fig.tight_layout()
    return fig
