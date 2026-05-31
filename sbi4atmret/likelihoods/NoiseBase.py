import torch
from torch import Tensor


class BaseNoise:
    def __init__(self, domain):
        self.domain = domain

    def __call__(self, batch_dict):
        """
        batches = [(theta1, x1), (theta2, x2), ...]
        """
        return self.forward(batch_dict)


    def flattening_likelihood(self, instrument, b):
        sigma_new = torch.sqrt(torch.Tensor(self.domain.obs_noise[instrument])**2 + 10**b)
        return sigma_new
    
    
    def _apply_noise(self):
        return NotImplemented

    def forward(self, batch_dict):
        raise NotImplementedError

