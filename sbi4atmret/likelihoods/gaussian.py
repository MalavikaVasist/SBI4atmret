import torch
from sbi4atmret.likelihoods.NoiseBase import BaseNoise
from sbi4atmret.utils.general import instrument_from_simname
from torch import Tensor

class GaussianNoise(BaseNoise):
    
    def __init__(self, domain):
        super().__init__(domain)

    
    def gaussian_noise(self, sigma, x):
        error = sigma * torch.randn_like(x) * self.domain.scale  
        return error  

    def _apply_noise(self, theta, x, sigma):
        error = self.gaussian_noise(sigma, x)
        return theta, x + error


    def compute_sigma(self, theta, sim_name):
        """
        Compute per-sample sigma for a given simulator.
        Returns sigma_new: (B, 1) tensor.
        """
        instrument = instrument_from_simname(sim_name)
        noise_name = "bfactor_noise_" + instrument
        b_indx = self.domain.sim_param_index[sim_name][noise_name]
        b = torch.unsqueeze(theta[:, b_indx], 1)
        return self.flattening_likelihood(instrument, b)

    def forward(self, batch_dict):

        noisy_batch_dict = {}
        for sim_name, (theta, x) in batch_dict.items():
            sigma_new = self.compute_sigma(theta, sim_name)
            theta, x_noisy = self._apply_noise(theta, x, sigma_new)
            noisy_batch_dict[sim_name] = (theta, x_noisy)

        return noisy_batch_dict

    
    
