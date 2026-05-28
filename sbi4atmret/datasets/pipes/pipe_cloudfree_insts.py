
from PipeBase import BasePipe
from sbi4atmret.utils.general import transform_uniform
from sbi4atmret.datasets.theta_mapper.theta_mapping_metadata import MiriGeminiHSTThetaMapper
import torch
import numpy as np
from typing import Optional, Dict, Any, Callable


class MiriGeminiHSTcloudfreePipe(BasePipe):
    def __init__(self,
                config,
                theta_mapper: Optional[Callable] = None,
    ):
        super().__init__(config)

        self._last_merge_metadata = None

        self.theta_mapper = theta_mapper
    
    def _extract_simulator_names(self):
        simulator_names = {}
        for sim_name, simulator in self.domain.simulator_dict:
            simulator_names[sim_name] = simulator.names
        return simulator_names 


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

        ###modify spec and theta

        x = self.merge_spec(x, xg, xh)
        
        # Use mapper to merge theta and store metadata for later evaluation
        if self.theta_mapper is not None:
            theta_dict = {
                "miri": thetam,
                "gemini": thetag,
                "hst": thetah
            }
            theta, self._last_merge_metadata = self.theta_mapper.merge(theta_dict)
        else:
            # Fallback: old behavior or simple concatenation
            theta = torch.cat([thetam, thetag, thetah], dim=1)

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
        if self.theta_mapper is None:
            raise RuntimeError("theta_mapper not initialized. Check config.theta_mapper settings.")
        
        theta_dict = {"miri": thetam, "gemini": thetag, "hst": thetah}
        theta, metadata = self.theta_mapper.merge(theta_dict)
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
            Dict mapping instrument names to simulator-specific parameters
        """
        if self.theta_mapper is None:
            raise RuntimeError("theta_mapper not initialized. Check config.theta_mapper settings.")
        
        if self._last_merge_metadata is None:
            raise RuntimeError(
                "Merge metadata not available. Call build_input() first during training, "
                "or use theta_mapper.split() directly with metadata."
            )
        return self.theta_mapper.split(theta_posterior, self._last_merge_metadata)
    
    def evaluate_posterior_predictive(self, theta_posterior, simulators_dict):
        """
        Generate posterior predictive samples from posterior theta samples.
        
        Typical usage during evaluation:
            pipe = MiriGeminiHSTcloudfreePipe(config)
            # ... train network ...
            
            theta_posterior = posterior.sample((10000,))  # 10k posterior samples
            spec_predictive = pipe.evaluate_posterior_predictive(
                theta_posterior, 
                {"miri": miri_sim, "gemini": gemini_sim, "hst": hst_sim}
            )
        
        Args:
            theta_posterior: [B, D_global] posterior samples
            simulators_dict: Dict with keys "miri", "gemini", "hst" containing simulator functions
            
        Returns:
            Dict with keys "miri", "gemini", "hst" containing simulated spectra [B, D]
        """
        # Split theta back to simulator-specific parameters
        theta_dict = self.split_theta(theta_posterior)
        
        # Run simulators independently
        specs = {
            inst: simulators_dict[inst](theta_dict[inst])
            for inst in theta_dict.keys()
        }
        
        return specs


    def forward(self, batch_dict: dict):

        batch_dict = self.modify_spec(batch_dict)
        batch_dict = self.modify_theta(batch_dict)

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