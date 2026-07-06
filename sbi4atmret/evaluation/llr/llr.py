"""
Log-Likelihood Ratio (LLR) test for nested model comparison.

Computes the statistic:
    T = log10( r_num / r_den )

where:
    r = posterior(θ*) / prior(θ*)

and θ* is found by optimizing the posterior-to-prior ratio for each model.

Workflow:
    1. For a given observation x, get posteriors from two models (e.g., mixture, cloudfree)
    2. Find the MAP of posterior/prior for each (numerator = complex, denominator = simple)
    3. T = log10(ratio_complex / ratio_simple)
    4. Build a null distribution by simulating observations under the simpler model
    5. Compare T_obs to the null to get a p-value
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, List, Callable

import numpy as np
import torch
from scipy.optimize import minimize
from tqdm import tqdm


# =========================================================
# RESULT
# =========================================================

@dataclass(frozen=True)
class LLRResult:
    """Container for LLR test results."""

    # T-statistic on the real observation
    T_obs: float

    # Null distribution of T under the simpler model
    T_null: Optional[np.ndarray] = None

    # p-value
    p_value: Optional[float] = None

    # MAP parameters for numerator and denominator
    theta_map_num: Optional[np.ndarray] = None
    theta_map_den: Optional[np.ndarray] = None

    # Figure
    figure: Optional[Any] = None


# =========================================================
# EVALUATOR
# =========================================================

class LLREvaluator:
    """
    Log-Likelihood Ratio evaluator for nested model comparison.

    Compares a complex model (numerator, e.g., mixture/cloudy) against
    a simpler model (denominator, e.g., cloudfree) by optimizing the
    posterior-to-prior ratio for each.

    Expects:
        - Two trained estimators (net_num, net_den)
        - Their priors (prior_num, prior_den)
        - An observation x_obs
        - Optionally: a simulator for generating the null distribution
    """

    @staticmethod
    def posterior_prior_ratio(theta, posterior, prior, device="cuda"):
        """
        Compute posterior(θ) / prior(θ).

        Args:
            theta: (D,) tensor
            posterior: distribution with .log_prob()
            prior: distribution with .log_prob()
            device: computation device

        Returns:
            ratio: scalar tensor
        """
        theta_t = theta.float().to(device)
        log_post = posterior.log_prob(theta_t.unsqueeze(0)).squeeze()
        log_prior = prior.log_prob(theta_t.unsqueeze(0)).squeeze()
        ratio = torch.exp(log_post - log_prior)
        return ratio.cpu().detach()

    @staticmethod
    def find_map_ratio(
        posterior,
        prior,
        init_theta: torch.Tensor,
        bounds: List[tuple],
        device: str = "cuda",
        method: str = "Nelder-Mead",
    ) -> tuple:
        """
        Find θ* that maximizes posterior/prior via optimization.

        Args:
            posterior: distribution with .log_prob()
            prior: distribution with .log_prob()
            init_theta: (D,) initial guess
            bounds: list of (lower, upper) per parameter
            device: computation device
            method: scipy optimizer method

        Returns:
            (theta_map, ratio_at_map)
        """

        def objective(theta_np):
            theta = torch.from_numpy(theta_np).float().to(device)
            log_post = posterior.log_prob(theta.unsqueeze(0)).squeeze()
            log_prior = prior.log_prob(theta.unsqueeze(0)).squeeze()
            ratio = (log_post - log_prior).cpu().detach().item()

            if not np.isfinite(ratio):
                return 1e10

            return -ratio  # minimize negative = maximize

        result = minimize(
            objective,
            x0=init_theta.numpy(),
            method=method,
            bounds=bounds,
        )

        theta_map = torch.from_numpy(result.x).float()
        ratio_at_map = LLREvaluator.posterior_prior_ratio(
            theta_map, posterior, prior, device
        )

        return theta_map, ratio_at_map

    def compute_llr(
        self,
        x_obs: torch.Tensor,
        net_num,
        net_den,
        prior_num,
        prior_den,
        bounds_num: List[tuple],
        bounds_den: List[tuple],
        device: str = "cuda",
    ) -> tuple:
        """
        Compute the LLR T-statistic for a single observation.

        T = log10(ratio_num / ratio_den)

        Args:
            x_obs: (1, D_obs) or (D_obs,) observation
            net_num: estimator for the complex (numerator) model
            net_den: estimator for the simple (denominator) model
            prior_num: prior for the complex model
            prior_den: prior for the simple model
            bounds_num: parameter bounds for the complex model
            bounds_den: parameter bounds for the simple model
            device: computation device

        Returns:
            (T, theta_map_num, theta_map_den)
        """
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)

        x_obs = x_obs.float().to(device)

        # Numerator: complex model
        posterior_num = net_num.flow_forward(x_obs)
        samples_num = posterior_num.sample((512,)).squeeze(1).cpu()
        log_p_num = posterior_num.log_prob(samples_num.to(device)).cpu()
        init_num = samples_num[torch.argmax(log_p_num)]

        theta_map_num, ratio_num = self.find_map_ratio(
            posterior_num, prior_num, init_num, bounds_num, device
        )

        # Denominator: simple model
        posterior_den = net_den.flow_forward(x_obs)
        samples_den = posterior_den.sample((512,)).squeeze(1).cpu()
        log_p_den = posterior_den.log_prob(samples_den.to(device)).cpu()
        init_den = samples_den[torch.argmax(log_p_den)]

        theta_map_den, ratio_den = self.find_map_ratio(
            posterior_den, prior_den, init_den, bounds_den, device
        )

        # T-statistic
        if ratio_den > 0 and ratio_num > 0:
            T = torch.log10(ratio_num / ratio_den).item()
        else:
            T = float("nan")

        return T, theta_map_num, theta_map_den

    def build_null_distribution(
        self,
        net_num,
        net_den,
        prior_num,
        prior_den,
        bounds_num: List[tuple],
        bounds_den: List[tuple],
        simulator_den,
        noise_fn: Callable,
        n_simulations: int = 100,
        device: str = "cuda",
    ) -> np.ndarray:
        """
        Build a null distribution of T by simulating under the simpler model.

        1. Sample θ from the simple model's posterior (MAP or samples)
        2. Simulate x_sim = simulator(θ) + noise
        3. Compute T for each x_sim

        Args:
            net_num, net_den: estimators for both models
            prior_num, prior_den: priors for both models
            bounds_num, bounds_den: parameter bounds
            simulator_den: simulator for the simpler model
            noise_fn: callable(x, theta) → x_noisy
            n_simulations: number of null samples
            device: computation device

        Returns:
            T_null: (n_simulations,) array of T-statistics under the null
        """
        # Get MAP from the simpler model on the real observation
        x_obs = self.x_obs.float().to(device)
        posterior_den = net_den.flow_forward(x_obs)
        samples = posterior_den.sample((512,)).squeeze(1).cpu()
        log_p = posterior_den.log_prob(samples.to(device)).cpu()
        theta_map = samples[torch.argmax(log_p)]

        # Simulate from the MAP
        output = simulator_den(theta_map.numpy())
        x_sim_base = torch.from_numpy(output.spectrum).float()

        T_null = []

        for _ in tqdm(range(n_simulations), desc="Building null"):
            # Add noise
            x_noisy = noise_fn(x_sim_base.unsqueeze(0), theta_map)

            # Compute T
            T, _, _ = self.compute_llr(
                x_noisy.squeeze(0), net_num, net_den,
                prior_num, prior_den, bounds_num, bounds_den, device,
            )

            if np.isfinite(T):
                T_null.append(T)

        return np.array(T_null)

    def run(
        self,
        x_obs: torch.Tensor,
        net_num,
        net_den,
        prior_num,
        prior_den,
        bounds_num: List[tuple],
        bounds_den: List[tuple],
        simulator_den=None,
        noise_fn: Callable = None,
        n_null: int = 100,
        device: str = "cuda",
        save_path: Optional[Path] = None,
    ) -> LLRResult:
        """
        Full LLR test: compute T_obs + null distribution + p-value.

        Args:
            x_obs: (D_obs,) observation
            net_num, net_den: trained estimators
            prior_num, prior_den: prior distributions
            bounds_num, bounds_den: parameter bounds
            simulator_den: simulator for null distribution (optional)
            noise_fn: noise function for null distribution (optional)
            n_null: number of null simulations
            device: computation device
            save_path: directory to save results

        Returns:
            LLRResult
        """
        self.x_obs = x_obs

        # Compute T on real observation
        T_obs, theta_num, theta_den = self.compute_llr(
            x_obs, net_num, net_den,
            prior_num, prior_den, bounds_num, bounds_den, device,
        )
        print(f"T_obs = {T_obs:.4f}")

        # Build null distribution
        T_null = None
        p_value = None

        if simulator_den is not None and noise_fn is not None:
            T_null = self.build_null_distribution(
                net_num, net_den, prior_num, prior_den,
                bounds_num, bounds_den, simulator_den, noise_fn,
                n_simulations=n_null, device=device,
            )

            p_value = float((T_null >= T_obs).mean())
            print(f"p-value = {p_value:.4f} (from {len(T_null)} null samples)")

        # Plot
        import matplotlib.pyplot as plt

        fig = None
        if T_null is not None:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(T_null, bins=30, alpha=0.6, color="steelblue", label="Null distribution")
            ax.axvline(T_obs, color="red", linewidth=2, linestyle="--",
                       label=f"T_obs = {T_obs:.3f}")
            ax.set_xlabel("T = log10(ratio_complex / ratio_simple)")
            ax.set_ylabel("Count")
            ax.set_title(f"LLR Test (p = {p_value:.4f})")
            ax.legend()
            plt.tight_layout()

        # Save
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            np.savetxt(save_path / "T_obs.txt", [T_obs])
            if T_null is not None:
                np.savetxt(save_path / "T_null.csv", T_null)
            if fig is not None:
                fig.savefig(save_path / "llr_test.pdf", bbox_inches="tight")

        return LLRResult(
            T_obs=T_obs,
            T_null=T_null,
            p_value=p_value,
            theta_map_num=theta_num.numpy() if theta_num is not None else None,
            theta_map_den=theta_den.numpy() if theta_den is not None else None,
            figure=fig,
        )
