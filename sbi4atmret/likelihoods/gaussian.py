import torch
from sbi4atmret.likelihoods.NoiseBase import BaseNoise
from sbi4atmret.utils.general import instrument_from_simname
from torch import Tensor
from typing import Dict, Optional


class GaussianNoise(BaseNoise):
    """
    Gaussian noise model with b-factor inflation.

    sigma_new = sqrt(obs_sigma^2 + 10^b)
    error = sigma_new * randn * scale

    The mapping from instrument → noise parameter name is provided
    via the `noise_params` kwarg in the config:

        noise:
          type: likelihoods.gaussian.GaussianNoise
          kwargs:
            noise_params:
              miri: bfactor_noise_miri
              gemini: bfactor_noise_gemini
              hst: bfactor_noise_hst
    """

    def __init__(self, domain, noise_params: Optional[Dict[str, str]] = None):
        super().__init__(domain)
        self.noise_params = noise_params or {}

    def _get_noise_param_name(self, sim_name: str) -> str:
        """
        Look up the noise parameter name for a given simulator.
        Uses the noise_params config mapping: instrument → param name.
        """
        instrument = instrument_from_simname(sim_name)
        if instrument in self.noise_params:
            return self.noise_params[instrument]
        # Fallback: try common conventions
        raise KeyError(
            f"No noise parameter configured for instrument '{instrument}'. "
            f"Add it to noise.kwargs.noise_params in your config YAML."
        )

    def gaussian_noise(self, sigma, x):
        error = sigma * torch.randn_like(x) * self.domain.scale
        return error

    def _apply_noise(self, theta, x, sigma):
        error = self.gaussian_noise(sigma, x)
        return theta, x + error

    def compute_sigma(self, theta, sim_name):
        """
        Compute per-sample sigma for a given simulator.
        
        If noise_params are configured for this instrument:
            sigma_new = sqrt(obs_sigma^2 + 10^b)  (flattened likelihood)
        Otherwise:
            sigma_new = obs_sigma  (no inflation, just observational noise)
        """
        instrument = instrument_from_simname(sim_name)
        
        if self.noise_params and instrument in self.noise_params:
            noise_name = self.noise_params[instrument]
            b_indx = self.domain.sim_param_index[sim_name][noise_name]
            b = torch.unsqueeze(theta[:, b_indx], 1)
            return self.flattening_likelihood(instrument, b)
        else:
            # No noise parameter — use raw observational sigma
            return torch.Tensor(self.domain.obs_noise[instrument]).unsqueeze(0)

    def forward(self, batch_dict):

        noisy_batch_dict = {}
        for sim_name, (theta, x) in batch_dict.items():
            sigma_new = self.compute_sigma(theta, sim_name)
            theta, x_noisy = self._apply_noise(theta, x, sigma_new)
            noisy_batch_dict[sim_name] = (theta, x_noisy)

        return noisy_batch_dict
