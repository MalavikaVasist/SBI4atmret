
from PipeBase import BasePipe
from sbi4atmret.utils.general import transform_uniform
from theta_mapper import MiriGeminiHSTThetaMapper
import torch
import numpy as np

class MiriGeminiHSTcloudfreePipe(BasePipe):
    def __init__(self, config):
        super().__init__(config)
        self.theta_mapper = MiriGeminiHSTThetaMapper(self.domain)
        self._last_merge_metadata = None  # Cache metadata for evaluation


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


        idxg = self.domain.param_index["cloudfree_gemini"]["bfactor_noise_g"]
        idxh = self.domain.param_index["cloudfree_hst"]["bfactor_noise_h"]

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
        

    def build_input(self, batch_dict):

        cf = batch_dict["cloudfree"]

        thetam, x   = cf["miri"]
        thetag, xg = cf["gemini"]
        thetah, xh = cf["hst"]

        x = self.merge_spec(x, xg, xh)
        
        # Use mapper to merge theta and store metadata for later evaluation
        theta, self._last_merge_metadata = self.theta_mapper.merge(thetam, thetag, thetah)

        return theta, x

    def merge_spec(self, x, xg, xh):
         ## merge x
        xinst = torch.hstack((xh, xg))[:,self.domain.unsort_index]
        x = torch.hstack((xinst, x))
        x = torch.cat([x, xg, xh], dim=-1)

        return x
    
    def merge_theta(self, thetam, thetag, thetah):
        """
        DEPRECATED: Use theta_mapper.merge() instead.
        Kept for backward compatibility.
        """
        theta, metadata = self.theta_mapper.merge(thetam, thetag, thetah)
        self._last_merge_metadata = metadata
        return theta

    def split_theta(self, theta_posterior):
        """
        Split posterior theta samples back to simulator-specific thetas for evaluation.
        
        MUST call build_input() first during training to establish metadata.
        Or pass metadata explicitly if evaluating independently.
        
        Args:
            theta_posterior: [B, D_global] posterior samples from the network
            
        Returns:
            thetam, thetag, thetah: Simulator-specific parameters
        """
        if self._last_merge_metadata is None:
            raise RuntimeError(
                "Merge metadata not available. Call build_input() first during training, "
                "or use theta_mapper.split() directly with metadata."
            )
        return self.theta_mapper.split(theta_posterior, self._last_merge_metadata)
    



    



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