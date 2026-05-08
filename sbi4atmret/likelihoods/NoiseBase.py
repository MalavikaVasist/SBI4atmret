import torch
from torch import Tensor

class BaseNoise:
    def __init__(self, config):
        self.config = config
        self.applynoise = self._apply_noise()

    def __call__(self, batch_dict):
        """
        batches = [(theta1, x1), (theta2, x2), ...]
        """
        return self.forward(batch_dict)
    

    def _apply_noise(self, x, theta, instrument):
        noise_name = "b_" + instrument 
        b_indx = self.param_index[noise_name]
        b = torch.unsqueeze(theta[:, b_indx], 1)

        sigma_new = torch.sqrt(torch.Tensor(self.noise_dict[instrument])**2 + 10**b)
        error_new = sigma_new * torch.randn_like(x) * self.scale    
        return x + error_new , sigma_new
    
    def _apply_other_noise(self):
        return NotImplemented

    def forward(self, batch_dict):
        raise NotImplementedError

