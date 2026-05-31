
from PipeBase import BasePipe
from sbi4atmret.utils.general import transform_uniform
from sbi4atmret.datasets.theta_mapper.thetamapperbase import BaseThetaMapper
import torch
import numpy as np
from typing import Optional, Dict, Any, Callable


class MiriGeminiHSTcloudfreePipe(BasePipe):
    def __init__(self,
                config,
                theta_mapper: Optional[Callable] = None,
    ):
        super().__init__(config)

        self.theta_mapper = theta_mapper
        self._last_theta_dict = None
    

    def modify_spec(self, batch_dict):
        cf = batch_dict["cloudfree"]
        theta, x   = cf["miri"]
        thetag, xg = cf["gemini"]
        thetah, xh = cf["hst"]

        ## modify spectrum
        x  = x[:, 1:1299]
        xg = xg[:, self._mask]
        xh = self._rebinit(xh)

        cf["miri"]   = (theta, x)
        cf["gemini"] = (thetag, xg)
        cf["hst"] = (thetah, xh)

        return batch_dict
    
    def modify_theta(self, batch_dict):
        cf = batch_dict["cloudfree"]

        thetag, xg = cf["gemini"]
        thetah, xh = cf["hst"]


        idxg = self.domain.sim_param_index["cloudfree_gemini"]["bfactor_noise_g"]
        idxh = self.domain.sim_param_index["cloudfree_hst"]["bfactor_noise_h"]

        thetag[:, idxg] = transform_uniform(
            thetag[:, idxg], -17, -11, -15, -7
        )
        thetah[:, idxh] = transform_uniform(
            thetah[:, idxh], -17, -11, -15, -7
        )
        
        ## flipping to get diff bfactor
        thetahf = torch.flip(thetah, dims=(0,))

        cf["gemini"]   = (thetag, xg)
        cf["hst"] = (thetahf, xh)

        return batch_dict
        
    def forward(self, batch_dict: dict):

        batch_dict = self.modify_spec(batch_dict)
        batch_dict = self.modify_theta(batch_dict)

        return batch_dict


    def merge_spec(self, x_dict):
         ## merge x
        xinst = torch.hstack((x_dict["cloudfree_hst"], x_dict["cloudfree_gemini"]))[:,self.domain.unsort_index]
        x = torch.hstack((xinst, x_dict["cloudfree_miri"]))
        x = torch.cat([x, x_dict["cloudfree_gemini"], x_dict["cloudfree_hst"]], dim=-1)

        return x
    
    def merge_theta(self, theta_dict):
        return self.theta_mapper.merge_theta(theta_dict)   

    def build_input(self, batch_dict):

        # cf = batch_dict["cloudfree"]

        ## spec
        x_dict = {
            inst: batch_dict[inst][1]
            for inst in self.theta_mapper.instrument_names
        }
        x = self.merge_spec(x_dict)

        ## theta
        theta_dict = {
                    inst: batch_dict[inst][0]
                    for inst in self.theta_mapper.instrument_names
                }
        theta = self.merge_theta(theta_dict)

        # store full dict for evaluation consistency
        self._last_theta_dict = theta_dict

        return theta, x

    def split_theta(self, theta_post):
        return self.theta_mapper.split_theta(theta_post)

    def split_spec(self, x):
        return self.domain.observation._get_observation_dict(full_flux=x.cpu().numpy())


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