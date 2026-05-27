import torch
import numpy as np

from petitRADTRANS.retrieval.rebin_give_width import rebin_give_width
from petitRADTRANS.retrieval.data import Data

class BasePipe:
    def __init__(self, config, domain):
        self.config = config
        self.domain = domain 
        self._mask = self._build_mask()
        
    def _build_mask(self):
        ## Masking gaps in observation
        wlen_geminisim = self.domain.sim_wlens["cloudfree_gemini"]
        obs_wlen_gemini = self.domain.obs_wlens["gemini"]
        obs_wlen_gemini = torch.from_numpy(obs_wlen_gemini)
        mask = torch.zeros(len(wlen_geminisim), dtype=torch.bool)
        for ind in range(len(obs_wlen_gemini)):
            mask = mask + torch.isin(torch.from_numpy(wlen_geminisim), obs_wlen_gemini[ind].item())
        
        return mask
    
    def _wlenbins(self, w1):
        wlen_bins = np.zeros_like(w1)
        wlen_bins[:-1] = np.diff(w1)
        wlen_bins[-1] = wlen_bins[-2]
        return wlen_bins

    def _rebinit(self, xh):
        wlen_hstsim = self.domain.sim_wlens["cloudfree_hst"]
        obs_wlen_hst = self.domain.obs_wlens["hst"]
        xh = xh[:,115:552]
        wlen_hstsim = wlen_hstsim[:,115:552]
        wlen_bins = self._wlenbins(obs_wlen_hst)
    #     xx = np.stack([Data.convolve(wlen_hstsim, x, 130) for x in xh])
    #     flux_rebinned = torch.stack([torch.from_numpy(rebin_give_width(wlen_hstsim, x, self.wlen, wlen_bins)) for x in xx])
        xx = Data.convolve(wlen_hstsim, xh, 130)
        flux_rebinned = torch.from_numpy(self._rebin_give_width(wlen_hstsim, xx, obs_wlen_hst, wlen_bins))
        return flux_rebinned
       

    @property
    def param_index(self):
        return self.domain.param_index

    def __call__(self, *batches):
        """
        batches = [(theta1, x1), (theta2, x2), ...]
        """
        return self.forward(*batches)
    
    def modify_spec(self, batch_dict):
        return NotImplementedError
    

    def modify_theta(self, batch_dict):
        return NotImplementedError


    def forward(self, batch_dict: dict):

        batch_dict = self.modify_spec(batch_dict)
        batch_dict = self.modify_theta(batch_dict)

        return batch_dict 

    def build_input(self, batch_dict):
        raise NotImplementedError
    
    def merge_spec(self, spectra):
        raise NotImplementedError
    
    def merge_theta(self, parameters):
        raise NotImplementedError

    def split_theta(self, theta):
        raise NotImplementedError


    
    



    def theta_to_dict(self, theta):
        names = self.config.simulator.names

        if theta.shape[-1] != len(names):
            raise ValueError(
                f"Theta dim {theta.shape[-1]} != number of names {len(names)}"
            )

        # Case 1: batched [B, D]
        if theta.ndim == 2:
            return {name: theta[:, i] for i, name in enumerate(names)}

        # Case 2: single sample [D]
        elif theta.ndim == 1:
            return {name: theta[i] for i, name in enumerate(names)}

        else:
            raise ValueError(f"Unsupported theta shape: {theta.shape}")
    

    def dict_to_theta(self, theta_dict):
        names = self.config.simulator.names
        values = [theta_dict[name] for name in names]

        # check if batched
        if values[0].ndim == 1:
            return torch.stack(values, dim=-1)   # [B, D]
        else:
            return torch.tensor(values)          # [D]
