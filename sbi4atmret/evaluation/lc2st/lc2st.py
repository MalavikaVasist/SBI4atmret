"""
Local Classifier Two-Sample Test in Normalizing Flow space (l-C2ST-NF).

Tests whether the learned posterior q(θ|x) matches the true posterior p(θ|x)
by training a classifier in the flow's latent space Z to distinguish:
- z ~ base_dist (samples from the flow's base distribution)
- z = T^{-1}(θ; x) (inverse-transformed ground truth θ)

If the posterior is well-calibrated, these should be indistinguishable
(classifier accuracy ≈ 0.5, p-value > α).

Requires: sbi >= 0.22 (for sbi.diagnostics.lc2st.LC2ST_NF)

References:
    - Linhart et al. (2023): "LC2ST: Local Classifier Two-Sample Tests"
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from sbi.diagnostics.lc2st import LC2ST_NF
from sbi.analysis.plot import pp_plot_lc2st


# =========================================================
# RESULT
# =========================================================

@dataclass(frozen=True)
class LC2STResult:
    """Container for l-C2ST evaluation results."""

    # Per-observation p-values
    p_values: list

    # Whether the null hypothesis is rejected per observation
    rejections: list

    # The trained LC2ST_NF object (for further analysis)
    lc2st_nf: Any

    # Figures
    figure_tdist: Optional[Any] = None
    figure_pp: Optional[Any] = None
    figure_pairgrid: Optional[Any] = None


# =========================================================
# EVALUATOR
# =========================================================

class LC2STEvaluator:
    """
    l-C2ST-NF evaluator for posterior calibration testing.

    Expects shared state from BaseEvaluator via __dict__.update:
        - self.net (EstimatorBase)
        - self.theta (posterior samples)
        - self.x_obs (observation on device)
        - self.pipe, self.batch_processor
        - self.test_keys, self.test_loaders
        - self.device
        - self.config
    """

    def collect_calibration_data(
        self,
        n_batches: int = 128,
    ) -> tuple:
        """
        Collect (theta, x, posterior_samples) triples from the test set.

        For each test batch:
        - theta, x come from the batch processor
        - posterior_samples = net.flow_forward(x).sample((1,))

        Args:
            n_batches: number of test batches to use

        Returns:
            (theta, x, post_samples) — all CPU tensors
        """
        from itertools import islice
        from sbi4atmret.datasets.DatasetBase import Dataset

        theta_list = []
        x_list = []
        post_list = []

        self.net.eval()

        with torch.no_grad():
            for batches in tqdm(
                islice(zip(*self.test_loaders), n_batches),
                total=n_batches,
                desc="Collecting l-C2ST data",
            ):
                batch_dict = {
                    key: batches[i]
                    for i, key in enumerate(self.test_keys)
                }

                theta, x = self.batch_processor.prepare_batch(batch_dict)

                # Sample from posterior
                posterior = self.net.flow_forward(x)
                samples = posterior.sample((1,)).squeeze(0)

                theta_list.append(theta.cpu())
                x_list.append(x.cpu())
                post_list.append(samples.cpu())

        theta_all = torch.cat(theta_list, dim=0)
        x_all = torch.cat(x_list, dim=0)
        post_all = torch.cat(post_list, dim=0)

        # Filter non-finite
        mask = (
            torch.isfinite(theta_all).all(dim=-1)
            & torch.isfinite(x_all).all(dim=-1)
            & torch.isfinite(post_all).all(dim=-1)
        )

        return theta_all[mask], x_all[mask], post_all[mask]

    def flow_inverse_transform(self, theta, x):
        """
        Apply the inverse flow transform: θ → z = T^{-1}(θ; x).

        Args:
            theta: (B, D) parameter samples
            x: (B, D_obs) observations (context)

        Returns:
            z: (B, D) latent samples
        """
        batch_size = 1024
        n = theta.shape[0]
        z_list = []

        self.net.eval()

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x_batch = x[start:end].float().to(self.device)
                theta_batch = theta[start:end].float().to(self.device)

                # Get posterior (which wraps the flow transforms)
                posterior = self.net.flow_forward(x_batch)

                # Apply inverse transforms
                z = theta_batch
                for trns in reversed(posterior.transforms):
                    z = trns.inv(z)

                z_list.append(z.cpu())

        return torch.cat(z_list, dim=0)

    def run_lc2st(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        post_samples: torch.Tensor,
        N: int = 50000,
        num_ensemble: int = 10,
        num_trials_null: int = 50,
        num_folds: int = 3,
        classifier: str = "mlp",
        clf_kwargs: Optional[Dict] = None,
        save_path: Optional[Path] = None,
    ) -> LC2STResult:
        """
        Run the l-C2ST-NF test.

        Args:
            theta: (N, D) ground truth parameters from test set
            x: (N, D_obs) corresponding observations
            post_samples: (N, D) posterior samples for those observations
            N: number of samples to use
            num_ensemble: number of classifier ensemble members
            num_trials_null: trials for null distribution
            num_folds: cross-validation folds
            classifier: "mlp" or "random_forest"
            clf_kwargs: classifier hyperparameters
            save_path: directory to save results

        Returns:
            LC2STResult
        """
        n_params = theta.shape[-1]

        if clf_kwargs is None:
            clf_kwargs = {
                "hidden_layer_sizes": (10 * n_params, 10 * n_params),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.0001,
                "batch_size": min(2048, N // 10),
                "learning_rate": "adaptive",
                "learning_rate_init": 1e-3,
                "max_iter": 1000,
                "tol": 1e-4,
                "early_stopping": False,
                "n_iter_no_change": 50,
            }

        # Flow base distribution (standard normal in latent space)
        flow_base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(n_params), torch.eye(n_params)
        )

        # Build LC2ST_NF
        lc2st_nf = LC2ST_NF(
            thetas=theta[:N].float().to(self.device),
            xs=x[:N].float().to(self.device),
            posterior_samples=post_samples[:N].float().to(self.device),
            flow_inverse_transform=self.flow_inverse_transform,
            flow_base_dist=flow_base_dist,
            num_eval=10000,
            num_ensemble=num_ensemble,
            classifier=classifier,
            num_trials_null=num_trials_null,
            clf_kwargs=clf_kwargs,
            num_folds=num_folds,
        )

        # Train under null
        _ = lc2st_nf.train_under_null_hypothesis()

        # Train on observed data
        _ = lc2st_nf.train_on_observed_data()

        # Save if requested
        if save_path is not None:
            import dill as pickle
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / "lc2st_nf.pkl", "wb") as f:
                pickle.dump(lc2st_nf, f)

        return LC2STResult(
            p_values=[],
            rejections=[],
            lc2st_nf=lc2st_nf,
        )

    def evaluate_observations(
        self,
        lc2st_nf,
        x_observations: torch.Tensor,
        obs_labels: list = None,
        conf_alpha: float = 0.05,
    ) -> tuple:
        """
        Evaluate l-C2ST on specific observations (e.g., the real one).

        Args:
            lc2st_nf: trained LC2ST_NF object
            x_observations: (K, D_obs) observations to test
            obs_labels: list of label strings for plotting
            conf_alpha: significance level

        Returns:
            (p_values, rejections, fig_tdist, fig_pp)
        """
        K = len(x_observations)
        if obs_labels is None:
            obs_labels = [f"Obs {i}" for i in range(K)]

        p_values = []
        rejections = []

        # T-statistic distribution plot
        fig_t, axes_t = plt.subplots(1, K, figsize=(4 * K, 3))
        if K == 1:
            axes_t = [axes_t]

        for i in range(K):
            x_o = x_observations[i].to(self.device)

            T_data = lc2st_nf.get_statistic_on_observed_data(x_o=x_o)
            T_null = lc2st_nf.get_statistics_under_null_hypothesis(x_o=x_o)
            p_val = lc2st_nf.p_value(x_o)
            reject = lc2st_nf.reject_test(x_o, alpha=conf_alpha)

            p_values.append(p_val)
            rejections.append(reject)

            quantiles = np.quantile(T_null, [0, 1 - conf_alpha])
            axes_t[i].hist(T_null, bins=50, density=True, alpha=0.5, label="Null")
            axes_t[i].axvline(T_data, color="red", label="Observed")
            axes_t[i].axvline(quantiles[1], color="black", linestyle="--", label="95% CI")
            axes_t[i].set_xlabel("Test statistic")
            axes_t[i].set_title(f"{obs_labels[i]}\np={p_val:.3f}, reject={reject}")

        axes_t[-1].legend(bbox_to_anchor=(1.1, 0.5), loc="center left")
        fig_t.tight_layout()

        # PP plot
        fig_pp, axes_pp = plt.subplots(1, K, figsize=(4 * K, 3))
        if K == 1:
            axes_pp = [axes_pp]

        for i in range(K):
            x_o = x_observations[i].to(self.device)

            probs_data, _ = lc2st_nf.get_scores(
                x_o=x_o, return_probs=True, trained_clfs=lc2st_nf.trained_clfs
            )
            probs_null, _ = lc2st_nf.get_statistics_under_null_hypothesis(
                x_o=x_o, return_probs=True
            )

            pp_plot_lc2st(
                probs=[probs_data],
                probs_null=probs_null,
                conf_alpha=conf_alpha,
                labels=["Classifier probs on data"],
                colors=["red"],
                ax=axes_pp[i],
            )
            axes_pp[i].set_title(f"PP-plot: {obs_labels[i]}")

        axes_pp[-1].legend(bbox_to_anchor=(1.1, 0.5), loc="center left")
        fig_pp.tight_layout()

        return p_values, rejections, fig_t, fig_pp
