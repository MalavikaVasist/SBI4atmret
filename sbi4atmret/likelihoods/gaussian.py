import torch
from NoiseBase import BaseNoise
from torch import Tensor

class GaussianNoise(BaseNoise):
    
    def __init__(self, domain):
        super().__init__(domain)

    
    def gaussian_noise(self, sigma, x):
        error = sigma * torch.randn_like(x) * self.domain.scale  
        return error  


    def _apply_noise(self, theta, x, instrument, simname):
        noise_name = "b_" + instrument 
        b_indx = self.domain.param_index[simname][noise_name]
        b = torch.unsqueeze(theta[:, b_indx], 1)

        sigma_new = self.flattening_likelihood(instrument, b)
        error = self.gaussian_noise(sigma_new, x)
        
        return theta, x + error


    def forward(self, batch_dict):

        noisy_batch_dict = {}
        for atm in batch_dict.keys():
            for inst in batch_dict[atm].keys():
                theta, x = batch_dict[atm][inst]
                _, x_noisy= self._apply_noise(theta, x, inst, str(atm)+ '_' +str(inst))
                noisy_batch_dict[atm][inst] = theta, x_noisy

        return noisy_batch_dict

    
    
