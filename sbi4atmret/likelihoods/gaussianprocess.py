"""
Gaussian Process correlated noise model.

Generates spatially-correlated noise using a GP kernel over wavelength,
rather than independent Gaussian noise per pixel. This models systematics
like fringing, detector artifacts, or correlated calibration errors.

Config example:
    noise:
      type: likelihoods.gp_noise.GPNoise
      kwargs:
        noise_params:
          miri: bfactor_noise_miri
          gemini: bfactor_noise_gemini
          hst: bfactor_noise_hst
        kernel: squared_exponential
        length_scale: 0.05       # in microns
        amplitude_fraction: 0.1  # GP amplitude as fraction of obs sigma
"""

import torch
import numpy as np
from typing import Dict, Optional

from sbi4atmret.likelihoods.NoiseBase import BaseNoise
from sbi4atmret.utils.general import instrument_from_simname


def _squared_exponential_kernel(wlen, length_scale):
    """
    Compute the squared exponential (RBF) covariance matrix.

    K(i,j) = exp(-0.5 * (w_i - w_j)^2 / l^2)

    Args:
        wlen: (D,) wavelength array in microns
        length_scale: kernel length scale in microns

    Returns:
        K: (D, D) covariance matrix
    """
    diff = wlen[:, None] - wlen[None, :]
    return np.exp(-0.5 * (diff / length_scale) ** 2)


def _matern32_kernel(wlen, length_scale):
    """
    Matérn 3/2 kernel.

    K(i,j) = (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)

    Args:
        wlen: (D,) wavelength array
        length_scale: kernel length scale in microns

    Returns:
        K: (D, D) covariance matrix
    """
    diff = np.abs(wlen[:, None] - wlen[None, :])
    scaled = np.sqrt(3.0) * diff / length_scale
    return (1.0 + scaled) * np.exp(-scaled)


KERNELS = {
    "squared_exponential": _squared_exponential_kernel,
    "rbf": _squared_exponential_kernel,
    "matern32": _matern32_kernel,
}


class GPNoise(BaseNoise):
    """
    Gaussian Process correlated noise model.

    Generates noise samples from a GP prior with a configurable kernel,
    added on top of independent Gaussian noise from the observational errors.

    The total noise for each sample is:
        error = independent_noise + correlated_noise

    where:
        independent_noise = sigma * randn * scale  (same as GaussianNoise)
        correlated_noise  = amplitude * L @ randn * scale
                            (L = Cholesky of kernel matrix)

    Args:
        domain: DomainContext
        noise_params: dict mapping instrument → noise parameter name (for b-factor)
        kernel: kernel name ("squared_exponential", "rbf", "matern32")
        length_scale: GP correlation length in microns
        amplitude_fraction: GP amplitude as fraction of median obs sigma
    """

    def __init__(
        self,
        domain,
        noise_params: Optional[Dict[str, str]] = None,
        kernel: str = "squared_exponential",
        length_scale: float = 0.05,
        amplitude_fraction: float = 0.1,
    ):
        super().__init__(domain)
        self.noise_params = noise_params or {}
        self.length_scale = length_scale
        self.amplitude_fraction = amplitude_fraction

        if kernel not in KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Available: {list(KERNELS.keys())}"
            )
        self.kernel_fn = KERNELS[kernel]

        # Precompute Cholesky factors per instrument
        self._cholesky_cache = {}
        for inst, data in domain.obs_wlens.items():
            wlen = data if isinstance(data, np.ndarray) else np.array(data)
            K = self.kernel_fn(wlen, self.length_scale)
            # Add small jitter for numerical stability
            K += 1e-8 * np.eye(len(wlen))
            L = np.linalg.cholesky(K)
            self._cholesky_cache[inst] = torch.from_numpy(L).float()

    def _sample_gp(self, instrument: str, batch_size: int) -> torch.Tensor:
        """
        Draw correlated noise samples from the GP prior.

        Returns: (B, D) tensor of correlated noise (unit amplitude).
        """
        L = self._cholesky_cache[instrument]
        D = L.shape[0]
        z = torch.randn(batch_size, D)
        # L @ z^T → (D, B) → transpose → (B, D)
        return (L @ z.T).T

    def compute_sigma(self, theta, sim_name):
        """
        Compute per-sample sigma (same as GaussianNoise).
        If noise_params configured: sigma = sqrt(obs_sigma^2 + 10^b)
        Otherwise: sigma = obs_sigma
        """
        instrument = instrument_from_simname(sim_name)

        if self.noise_params and instrument in self.noise_params:
            noise_name = self.noise_params[instrument]
            b_indx = self.domain.sim_param_index[sim_name][noise_name]
            b = torch.unsqueeze(theta[:, b_indx], 1)
            return self.flattening_likelihood(instrument, b)
        else:
            return torch.Tensor(self.domain.obs_noise[instrument]).unsqueeze(0)

    def _apply_noise(self, theta, x, sigma, instrument):
        """
        Apply independent + correlated noise.
        """
        B = x.shape[0]

        # Independent component (same as GaussianNoise)
        independent = sigma * torch.randn_like(x) * self.domain.scale

        # Correlated component (GP)
        gp_samples = self._sample_gp(instrument, B)
        obs_sigma = torch.Tensor(self.domain.obs_noise[instrument])
        amplitude = self.amplitude_fraction * obs_sigma.median() * self.domain.scale
        correlated = amplitude * gp_samples

        return theta, x + independent + correlated

    def forward(self, batch_dict):

        noisy_batch_dict = {}
        for sim_name, (theta, x) in batch_dict.items():
            instrument = instrument_from_simname(sim_name)
            sigma = self.compute_sigma(theta, sim_name)
            theta, x_noisy = self._apply_noise(theta, x, sigma, instrument)
            noisy_batch_dict[sim_name] = (theta, x_noisy)

        return noisy_batch_dict
