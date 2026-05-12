
from PipeBase import BasePipe
from sbi4atmret.utils.general import transform_uniform
import torch
import numpy as np



class MiriGeminiHSTcloudfreePipe(BasePipe):
    def __init__(self, config):
        super().__init__(config)
       

    def modify_spec(self, batch_dict):
        cf = batch_dict["cloudfree"]
        theta, x   = cf["miri"]
        thetag, xg = cf["gemini"]
        
        ## modify spectrum
        x  = x[:, 1:1299]
        xg = xg[:, self._mask]

        cf["miri"]   = (theta, x)
        cf["gemini"] = (thetag, xg)

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

        xinst = torch.hstack((xh, xg))[:,self.domain.unsort_index]
        x = torch.hstack((xinst, x))
        x = torch.cat([x, xg, xh], dim=-1)

        ## miri
        idx_scalingm = self.domain.parameter_idx["cloudfree_miri"]["bfactor_noise_miri"]
        b = torch.unsqueeze(thetam[:, idx_scalingm], 1)

        ## gemini
        idx_c = self.domain.parameter_idx["cloudfree_gemini"]["mxture_fraction"]
        idx_scalingg =  self.domain.parameter_idx["cloudfree_gemini"]["Cushing_scale_factor_g"]
        thetag = np.delete(theta, [idx_c, idx_scalingg], axis=1)

        ## hst
        idx_scalingh =  self.domain.parameter_idx["cloudfree_hst"]["bfactor_noise_gemini"] ##inverted
        bCh = torch.unsqueeze(thetah[:,idx_scalingh],1)

        ## building together
        theta = torch.hstack((thetag, bCh, b))

        return theta, x


    

    




    