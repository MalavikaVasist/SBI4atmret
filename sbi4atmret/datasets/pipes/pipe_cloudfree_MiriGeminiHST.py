
from typing import Callable, Optional, List

from sbi4atmret.datasets.pipes.PipeBase import BasePipe
from sbi4atmret.utils.general import transform_uniform, simname_from_instrument
import torch
import numpy as np


class MiriGeminiHSTcloudfreePipe(BasePipe):
    def __init__(self,
                posterior_names: Optional[List[str]] = None, 
                domain: Optional[Callable] = None ):
        
        super().__init__(posterior_names= posterior_names, 
                         domain = domain)

        self.domain = domain
        self._last_theta_dict = None
    

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