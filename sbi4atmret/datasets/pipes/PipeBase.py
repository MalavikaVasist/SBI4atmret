import torch
import numpy as np

from petitRADTRANS.retrieval.rebin_give_width import rebin_give_width
from petitRADTRANS.retrieval.data import Data

from sbi4atmret.datasets.theta_mapper.thetamapperbase import BaseThetaMapper


class BasePipe:
    def __init__(self, 
                 domain= None, 
                 posterior_names=None):
        
        self.domain = domain
        self.posterior_names = posterior_names
        self.theta_mapper = BaseThetaMapper(domain=domain, 
                                            posterior_param_names=posterior_names)

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
        wlen_hstsim = wlen_hstsim[115:552]
        wlen_bins = self._wlenbins(obs_wlen_hst)
        xx = np.stack([Data.convolve(wlen_hstsim, x, 130) for x in xh])
        flux_rebinned = torch.stack([torch.from_numpy(rebin_give_width(wlen_hstsim, x, obs_wlen_hst, wlen_bins)) for x in xx])
        # xx = Data.convolve(wlen_hstsim, xh, 130)
        # flux_rebinned = torch.from_numpy(rebin_give_width(wlen_hstsim, xx, obs_wlen_hst, wlen_bins))
        return flux_rebinned
       
    @property
    def param_index(self):
        return self.domain.param_index

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def modify_spec(self, batch_dict):
        raise NotImplementedError
    

    def modify_theta(self, batch_dict):
        raise NotImplementedError


    def forward(self, batch_dict: dict, mode="train"):

        batch_dict = self.modify_spec(batch_dict)

        if mode == "train":
            batch_dict = self.modify_theta(batch_dict)

        return batch_dict 
    
    def merge_spec(self, spectra):
        raise NotImplementedError
    
    def merge_theta(self, batch_dict):
        ## theta
        theta_dict = {
                    inst: batch_dict[inst][0]
                    for inst in self.theta_mapper.instrument_names
                }
        theta = self.theta_mapper.merge_theta(theta_dict)

        # store full dict for evaluation consistency
        self._last_theta_dict = theta_dict

        return theta

    def build_input(self, batch_dict, mode="train"):

        x = self.merge_spec(batch_dict)
        if mode == "train":
            theta = self.merge_theta(batch_dict)
        elif mode == "eval":
            theta  = None
        
        return theta, x


    def split_theta(self, theta_post):
        return self.theta_mapper.split_theta(theta_post)

    def split_spec(self, x):
        return self.domain.observation._get_observation_dict(full_flux=x.cpu().numpy())


