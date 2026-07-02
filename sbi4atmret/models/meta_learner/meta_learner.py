"""
Meta-learner ensemble for SBI posterior stacking.

Instead of stacking point estimates, each base model produces posterior
summaries (means, stds, and optionally embeddings), and a small MLP
meta-network is trained to predict improved posterior parameters.

Workflow:
    1. Train N independent SBI models (different seeds/architectures/hyperparameters).
    2. Freeze all of them.
    3. On a held-out validation set, collect each model's posterior summary.
    4. Train a meta-MLP on those summaries → target theta.
    5. At test time, run all base models once and feed their summaries into the meta-MLP.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =========================================================
# META-MLP NETWORK
# =========================================================

class MetaMLP(nn.Module):
    """
    Small MLP that takes concatenated posterior summaries from N base models
    and predicts refined posterior parameters (mean and log-std).

    Input dim:  N * summary_dim_per_model
    Output dim: 2 * n_params (mean + log_std)
    """

    def __init__(
        self,
        input_dim: int,
        n_params: int,
        hidden_dims: List[int] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_params = n_params

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Output: mean and log_std for each parameter
        layers.append(nn.Linear(in_dim, 2 * n_params))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) concatenated summaries

        Returns:
            mean: (B, n_params)
            log_std: (B, n_params)
        """
        out = self.net(x)
        mean, log_std = out.chunk(2, dim=-1)
        return mean, log_std

    def predict(self, x):
        """Return mean and std."""
        mean, log_std = self.forward(x)
        return mean, torch.exp(log_std)

    def loss(self, x, theta_true):
        """
        Gaussian NLL loss for the meta-learner predictions.

        Args:
            x: (B, input_dim) concatenated summaries
            theta_true: (B, n_params) ground truth parameters

        Returns:
            scalar loss
        """
        mean, log_std = self.forward(x)
        # Gaussian NLL: 0.5 * (log(2*pi) + 2*log_std + ((theta - mean)/std)^2)
        var = torch.exp(2 * log_std)
        nll = 0.5 * (2 * log_std + (theta_true - mean) ** 2 / var)
        return nll.mean()


# =========================================================
# POSTERIOR SUMMARY EXTRACTION
# =========================================================

def extract_posterior_summaries(
    base_models: List,
    x_obs: torch.Tensor,
    n_samples: int = 1024,
    include_embeddings: bool = False,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract posterior summaries from a list of frozen base models.

    For each model, computes:
        - posterior mean: (n_params,)
        - posterior std:  (n_params,)
        - (optional) embedding: (emb_dim,)

    Args:
        base_models: list of EstimatorBase instances (frozen)
        x_obs: (1, D_obs) or (B, D_obs) observation tensor
        n_samples: number of posterior samples for computing mean/std
        include_embeddings: whether to include intermediate embeddings
        device: computation device

    Returns:
        summaries: (B, N * summary_dim) concatenated summaries
    """

    x_obs = x_obs.to(device)
    all_summaries = []

    with torch.no_grad():
        for model in base_models:
            # Get embedding
            x_emb = model.embedding(x_obs)

            # Get posterior distribution
            posterior = model.flow.flow(x_emb)

            # Sample and compute statistics
            samples = posterior.sample((n_samples,))  # (n_samples, B, n_params)
            post_mean = samples.mean(dim=0)           # (B, n_params)
            post_std = samples.std(dim=0)             # (B, n_params)

            # Concatenate summary for this model
            summary = torch.cat([post_mean, post_std], dim=-1)  # (B, 2*n_params)

            if include_embeddings:
                summary = torch.cat([summary, x_emb], dim=-1)

            all_summaries.append(summary.cpu())

    # Concatenate across all models: (B, N * summary_dim)
    return torch.cat(all_summaries, dim=-1)


def extract_validation_summaries(
    base_models: List,
    dataloader: DataLoader,
    batch_processor,
    n_samples: int = 256,
    include_embeddings: bool = False,
    device: str = "cuda",
) -> tuple:
    """
    Extract posterior summaries for an entire validation set.

    Args:
        base_models: list of frozen EstimatorBase instances
        dataloader: validation data loader yielding batches
        batch_processor: BatchProcessor for preparing batches
        n_samples: samples per posterior for mean/std
        include_embeddings: include intermediate embeddings
        device: computation device

    Returns:
        (summaries, thetas): both as tensors
            summaries: (N_val, N_models * summary_dim)
            thetas: (N_val, n_params) ground truth
    """

    all_summaries = []
    all_thetas = []

    for batches in tqdm(dataloader, desc="Extracting summaries"):
        theta, x = batch_processor.prepare_batch(batches)

        # x is on device, extract summaries
        summaries = extract_posterior_summaries(
            base_models, x,
            n_samples=n_samples,
            include_embeddings=include_embeddings,
            device=device,
        )

        all_summaries.append(summaries)
        all_thetas.append(theta.cpu())

    return torch.cat(all_summaries, dim=0), torch.cat(all_thetas, dim=0)


# =========================================================
# TRAINING THE META-LEARNER
# =========================================================

def train_meta_learner(
    meta_model: MetaMLP,
    summaries: torch.Tensor,
    thetas: torch.Tensor,
    n_epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.1,
    device: str = "cuda",
    save_path: Optional[Path] = None,
) -> dict:
    """
    Train the meta-learner MLP on extracted summaries.

    Args:
        meta_model: MetaMLP instance
        summaries: (N, input_dim) from extract_validation_summaries
        thetas: (N, n_params) ground truth
        n_epochs: training epochs
        batch_size: batch size
        lr: learning rate
        weight_decay: weight decay
        val_fraction: fraction for validation split
        device: computation device
        save_path: optional path to save best model

    Returns:
        dict with training history
    """

    meta_model = meta_model.to(device)

    # Train/val split
    n_val = int(len(summaries) * val_fraction)
    n_train = len(summaries) - n_val

    perm = torch.randperm(len(summaries))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = TensorDataset(summaries[train_idx], thetas[train_idx])
    val_ds = TensorDataset(summaries[val_idx], thetas[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(
        meta_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        # Train
        meta_model.train()
        train_losses = []
        for s_batch, t_batch in train_loader:
            s_batch, t_batch = s_batch.to(device), t_batch.to(device)
            loss = meta_model.loss(s_batch, t_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        meta_model.eval()
        val_losses = []
        with torch.no_grad():
            for s_batch, t_batch in val_loader:
                s_batch, t_batch = s_batch.to(device), t_batch.to(device)
                loss = meta_model.loss(s_batch, t_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path is not None:
                torch.save(meta_model.state_dict(), save_path)

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Best: {best_val_loss:.4f}"
            )

    # Load best model
    if save_path is not None and save_path.exists():
        meta_model.load_state_dict(torch.load(save_path))

    return history


# =========================================================
# INFERENCE WITH META-LEARNER
# =========================================================

def meta_predict(
    base_models: List,
    meta_model: MetaMLP,
    x_obs: torch.Tensor,
    n_samples: int = 1024,
    include_embeddings: bool = False,
    device: str = "cuda",
) -> tuple:
    """
    Run the full meta-learner pipeline at test time.

    1. Extract posterior summaries from all base models.
    2. Feed into meta-MLP to get refined mean and std.

    Args:
        base_models: list of frozen EstimatorBase instances
        meta_model: trained MetaMLP
        x_obs: (1, D_obs) observation
        n_samples: samples for posterior summary extraction
        include_embeddings: include embeddings in summaries
        device: computation device

    Returns:
        (mean, std): each (1, n_params) — the meta-learner's prediction
    """

    meta_model = meta_model.to(device).eval()

    summaries = extract_posterior_summaries(
        base_models, x_obs,
        n_samples=n_samples,
        include_embeddings=include_embeddings,
        device=device,
    )

    with torch.no_grad():
        mean, std = meta_model.predict(summaries.to(device))

    return mean.cpu(), std.cpu()


# =========================================================
# UTILITY: LOAD FROZEN BASE MODELS
# =========================================================

def load_base_models(
    checkpoint_paths: List[Path],
    model_builder,
    device: str = "cuda",
) -> List:
    """
    Load multiple trained models from checkpoints and freeze them.

    Args:
        checkpoint_paths: list of paths to checkpoint .pth files
        model_builder: callable that returns a fresh model instance
                       (e.g., lambda: BaseModel(config).build())
        device: device to load models onto

    Returns:
        list of frozen EstimatorBase instances
    """

    base_models = []

    for path in checkpoint_paths:
        model = model_builder()
        checkpoint = torch.load(path, map_location=device)
        model.estimator.load_state_dict(checkpoint["estimator"])
        model.estimator.to(device)
        model.estimator.eval()

        # Freeze all parameters
        for param in model.estimator.parameters():
            param.requires_grad = False

        base_models.append(model.estimator)

    return base_models
