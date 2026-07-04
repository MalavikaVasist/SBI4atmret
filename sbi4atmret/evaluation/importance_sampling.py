"""
Importance Sampling for posterior refinement.

Uses the NPE posterior as proposal distribution to compute
importance weights that correct for any amortization gap:

    w_i = p(x_obs | θ_i) * π(θ_i) / q(θ_i | x_obs)

where θ_i ~ q(θ | x_obs) are samples from the NPE posterior (proposal).

The weighted samples give a corrected posterior that is asymptotically
exact even if the amortized posterior q is imperfect.

Also computes:
- Effective sample size (ESS)
- Log-evidence estimate
- Normalized weights for downstream use (e.g., IS corner plots)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
import pandas as pd
from scipy.special import logsumexp
from tqdm import tqdm

from sbi4atmret.utils.general import instrument_from_simname


# =========================================================
# RESULT OBJECT
# =========================================================

@dataclass(frozen=True)
class ISResult:
    """Container for importance sampling results."""

    theta: torch.Tensor              # (N, D) posterior samples used
    raw_log_weights: np.ndarray      # (N,) unnormalized log-weights
    normalized_weights: np.ndarray   # (N,) normalized weights
    n_eff: float                     # effective sample size
    log_evidence: float              # log-evidence estimate
    log_evidence_std: float          # std of log-evidence


# =========================================================
# WEIGHT NORMALIZATION
# =========================================================

def normalize_log_weights(raw_log_weights: np.ndarray, percentile: float = 100.0) -> np.ndarray:
    """
    Normalize raw log importance weights using log-sum-exp.

    Optionally clips extreme weights to reduce variance.

    Args:
        raw_log_weights: (N,) log w_i = log L + log π - log q
        percentile: upper percentile for clipping (100 = no clip)

    Returns:
        normalized_weights: (N,) summing to N
    """
    if percentile < 100.0:
        threshold = np.percentile(raw_log_weights, percentile)
        raw_log_weights = np.clip(raw_log_weights, None, threshold)

    N = len(raw_log_weights)
    C = -np.min(raw_log_weights)
    normalized = np.exp(
        np.log(N) + (raw_log_weights + C) - logsumexp(raw_log_weights + C)
    )
    return normalized


def compute_ess(normalized_weights: np.ndarray) -> float:
    """Effective sample size from normalized weights."""
    return float(np.sum(normalized_weights) ** 2 / np.sum(normalized_weights ** 2))


# =========================================================
# IMPORTANCE SAMPLER
# =========================================================

class ImportanceSampler:
    """
    Importance sampling using the NPE posterior as proposal.

    Expects shared state from BaseEvaluator via __dict__.update.
    Uses:
        - self.net (estimator with .flow_forward)
        - self.pipe (for split_theta)
        - self.simulator_dict
        - self.batch_processor
        - self.noise
        - self.observation
        - self.domain
        - self.theta (posterior samples)
        - self.x_obs
        - self.posterior
    """

    def compute_log_likelihood(
        self,
        theta: torch.Tensor,
        x_simulated: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute Gaussian log-likelihood for each sample.

        log L(x_obs | θ) = -0.5 Σ_d (x_obs_d - x_sim_d)² / σ_d²

        where σ² includes the heteroscedastic b-factor noise.

        Args:
            theta: (B, D) parameter samples (merged posterior space)
            x_simulated: (B, D_obs) simulated + processed spectra

        Returns:
            log_likelihoods: (B,) numpy array
        """
        x_obs = self.observation.full_observation  # (D_obs,)

        # Compute per-instrument variance
        theta_dict = self.pipe.split_theta(theta)
        all_var = []

        for sim_name in self.simulator_dict.keys():
            instrument = instrument_from_simname(sim_name)
            sigma = self.noise.compute_sigma(theta_dict[sim_name], sim_name)
            # sigma is (B, 1), variance = sigma² * scale²
            var = (sigma ** 2) * (self.domain.scale ** 2)
            all_var.append(var)

        # Merge variances in observation order (same as merge_spec)
        # For simplicity, concatenate and let pipe ordering handle it
        # This is a simplification — full version should use pipe.merge logic
        var_merged = torch.cat(all_var, dim=-1)

        # Compute log-likelihood
        x_obs_tensor = torch.from_numpy(x_obs).float().unsqueeze(0)
        residuals = x_simulated.cpu() - x_obs_tensor
        log_lik = -0.5 * (residuals ** 2 / var_merged.cpu()).sum(dim=-1)

        return log_lik.numpy()

    def simulate_batch(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Simulate spectra for a batch of theta and process through pipe.

        Args:
            theta: (B, D) merged posterior samples

        Returns:
            x_processed: (B, D_obs) processed spectra
        """
        theta_dict = self.pipe.split_theta(theta)
        predictive = {}

        for sim_name, simulator in self.simulator_dict.items():
            spectra = []
            for i in range(theta_dict[sim_name].shape[0]):
                output = simulator(theta_dict[sim_name][i].numpy())
                spectra.append(torch.from_numpy(output.spectrum).float())
            predictive[sim_name] = (theta_dict[sim_name], torch.stack(spectra))

        # Process (spectral transforms, no theta modification, no noise)
        processed = self.batch_processor.process(predictive, mode="eval", add_noise=False)
        _, x = self.batch_processor.merge(processed, mode="eval")

        return x.cpu()

    def run(
        self,
        n_samples: int = 10000,
        batch_size: int = 500,
        percentile: float = 100.0,
        save_path: Optional[Path] = None,
    ) -> ISResult:
        """
        Run importance sampling.

        1. Take theta samples from the posterior (proposal)
        2. Simulate spectra for each
        3. Compute log-likelihood, log-prior, log-proposal
        4. Compute raw log-weights and normalize
        5. Save results

        Args:
            n_samples: number of IS samples (uses self.theta)
            batch_size: simulation batch size
            percentile: weight clipping percentile
            save_path: directory to save results. If None, uses eval_dir/importance_sampling/

        Returns:
            ISResult
        """
        if save_path is None:
            save_path = self.eval_dir / "importance_sampling"
        save_path.mkdir(parents=True, exist_ok=True)

        # Check if already computed
        weights_file = save_path / "normalized_weights.csv"
        theta_file = save_path / "th_withnormweights.csv"
        raw_weights_file = save_path / "raw_log_weights.csv"

        if weights_file.exists() and theta_file.exists() and raw_weights_file.exists():
            # Load from disk
            normalized_weights = pd.read_csv(weights_file, header=None).values.flatten()
            theta_is = torch.from_numpy(pd.read_csv(theta_file).values).float()
            raw_log_weights = pd.read_csv(raw_weights_file, header=None).values.flatten()

            n_eff = compute_ess(normalized_weights)
            N = len(raw_log_weights)
            log_evidence = float(logsumexp(raw_log_weights) - np.log(N))
            log_evidence_std = float(np.sqrt((N - n_eff) / (N * n_eff))) if n_eff > 1 else float("inf")

            return ISResult(
                theta=theta_is,
                raw_log_weights=raw_log_weights,
                normalized_weights=normalized_weights,
                n_eff=n_eff,
                log_evidence=log_evidence,
                log_evidence_std=log_evidence_std,
            )

        # Use posterior samples
        theta_samples = self.theta[:n_samples].cpu()

        # Compute log-proposal: q(θ | x_obs)
        with torch.no_grad():
            log_q = self.posterior.log_prob(
                theta_samples.float().to(self.device)
            ).cpu().numpy()

        # Compute log-prior: π(θ)
        from zuko.distributions import BoxUniform
        prior_lower = torch.tensor([p.lower for p in self.config.prior.parameters])
        prior_upper = torch.tensor([p.upper for p in self.config.prior.parameters])
        prior = BoxUniform(prior_lower, prior_upper)
        log_prior = prior.log_prob(theta_samples).numpy()

        # Simulate and compute log-likelihood in batches
        all_log_lik = []

        for start in tqdm(range(0, n_samples, batch_size), desc="IS simulation"):
            end = min(start + batch_size, n_samples)
            theta_batch = theta_samples[start:end]

            x_sim = self.simulate_batch(theta_batch)
            log_lik = self.compute_log_likelihood(theta_batch, x_sim)
            all_log_lik.append(log_lik)

        log_likelihood = np.concatenate(all_log_lik)

        # Raw log-weights
        raw_log_weights = log_likelihood + log_prior - log_q

        # Filter invalid
        valid_mask = np.isfinite(raw_log_weights)
        theta_valid = theta_samples[valid_mask]
        raw_log_weights_valid = raw_log_weights[valid_mask]

        # Normalize
        normalized_weights = normalize_log_weights(raw_log_weights_valid, percentile=percentile)

        # Statistics
        n_eff = compute_ess(normalized_weights)
        N = len(raw_log_weights_valid)
        log_evidence = float(logsumexp(raw_log_weights_valid) - np.log(N))
        log_evidence_std = float(np.sqrt((N - n_eff) / (N * n_eff))) if n_eff > 1 else float("inf")

        # Save
        pd.DataFrame(normalized_weights).to_csv(weights_file, index=False, header=False)
        pd.DataFrame(theta_valid.numpy()).to_csv(theta_file, index=False)
        pd.DataFrame(raw_log_weights_valid).to_csv(raw_weights_file, index=False, header=False)

        # Save summary
        summary = {
            "n_samples": N,
            "n_eff": n_eff,
            "sampling_efficiency": n_eff / N,
            "log_evidence": log_evidence,
            "log_evidence_std": log_evidence_std,
            "percentile": percentile,
        }
        pd.DataFrame([summary]).to_csv(save_path / "summary.csv", index=False)

        print(f"IS complete: N_eff={n_eff:.1f}, efficiency={n_eff/N:.4f}, "
              f"log Z={log_evidence:.2f} ± {log_evidence_std:.2f}")

        return ISResult(
            theta=theta_valid,
            raw_log_weights=raw_log_weights_valid,
            normalized_weights=normalized_weights,
            n_eff=n_eff,
            log_evidence=log_evidence,
            log_evidence_std=log_evidence_std,
        )
