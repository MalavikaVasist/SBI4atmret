
from typing import Callable, Optional, List

from sbi4atmret.datasets.pipes.PipeBase import BasePipe
from sbi4atmret.utils.general import transform_uniform, simname_from_instrument
import torch
import numpy as np

from petitRADTRANS.retrieval.rebin_give_width import rebin_give_width
from petitRADTRANS.retrieval.data import Data


class MiriGeminiHSTcloudfreePipe(BasePipe):
    def __init__(self,
                posterior_names: Optional[List[str]] = None, 
                domain: Optional[Callable] = None ):
        
        super().__init__(posterior_names= posterior_names, 
                         domain = domain)

        self.domain = domain
        self._last_theta_dict = None
        self._mask = self._build_mask()


    ##specific    
    def _build_mask(self):
        ## Masking gaps in observation
        wlen_geminisim = self.domain.sim_wlens["cloudfree_gemini"]
        obs_wlen_gemini = self.domain.obs_wlens["gemini"]
        obs_wlen_gemini = torch.from_numpy(obs_wlen_gemini)
        mask = torch.zeros(len(wlen_geminisim), dtype=torch.bool)
        for ind in range(len(obs_wlen_gemini)):
            mask = mask + torch.isin(torch.from_numpy(wlen_geminisim), obs_wlen_gemini[ind].item())
        
        return mask
    
    ##specific 
    def _wlenbins(self, w1):
        wlen_bins = np.zeros_like(w1)
        wlen_bins[:-1] = np.diff(w1)
        wlen_bins[-1] = wlen_bins[-2]
        return wlen_bins

    ##specific 
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
    

    def modify_spec(self, batch_dict):
        theta, x   = batch_dict["cloudfree_miri"]
        thetag, xg = batch_dict["cloudfree_gemini"]
        thetah, xh = batch_dict["cloudfree_hst"]

        ## modify spectrum
        x  = x[:, 1:1299]
        xg = xg[:, self._mask]
        xh = self._rebinit(xh)

        batch_dict["cloudfree_miri"]   = (theta, x)
        batch_dict["cloudfree_gemini"] = (thetag, xg)
        batch_dict["cloudfree_hst"] = (thetah, xh)

        return batch_dict
    
    def modify_theta(self, batch_dict):

        thetag, xg = batch_dict["cloudfree_gemini"]
        thetah, xh = batch_dict["cloudfree_hst"]

        idxg = self.domain.sim_param_index["cloudfree_gemini"]["bfactor_noise_gemini"]
        idxh = self.domain.sim_param_index["cloudfree_hst"]["bfactor_noise_hst"]

        thetag[:, idxg] = transform_uniform(
            thetag[:, idxg], -17, -11, -15, -7
        )
        thetah[:, idxh] = transform_uniform(
            thetah[:, idxh], -17, -11, -15, -7
        )
        
        ## flipping to get diff bfactor
        thetahf = torch.flip(thetah, dims=(0,))

        batch_dict["cloudfree_gemini"]   = (thetag, xg)
        batch_dict["cloudfree_hst"] = (thetahf, xh)

        return batch_dict
        

    

# experiment = Experiment(
#     theta_mapper=CloudfreeThetaMapper(),
#     simulator_graph=CloudfreeGraph(),
#     processor=MiriHSTGeminiProcessor(),
#     feature_builder=StandardFeatureBuilder(),
# )

# experiment = Experiment(
#     theta_mapper=PatchyThetaMapper(),
#     simulator_graph=PatchyCombinerGraph(),
#     processor=MiriHSTGeminiProcessor(),
#     feature_builder=StandardFeatureBuilder(),
# )