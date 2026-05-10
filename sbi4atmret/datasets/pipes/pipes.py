
from PipeBase import BasePipe
from sbi4atmret.utils.general import transform_uniform
import torch



class MiriGeminiHSTcloudfreePipe(BasePipe):
    def __init__(self, config):
        super().__init__(config)
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
    
    def modify_prior(self, batch_dict):
        cf = batch_dict["cloudfree"]

        thetag, xg = cf["gemini"]
        thetah, xh = cf["hst"]


        idxg = self.domain.param_index["cloudfree_gemini"]["Mike_Line_b_Cushing_g"]
        idxh = self.domain.param_index["cloudfree_hst"]["Mike_Line_b_Cushing_h"]

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

        theta, x   = cf["miri"]
        thetag, xg = cf["gemini"]
        thetah, xh = cf["hst"]

        xinst = torch.hstack((xh, xg))[:,self.domain.unsort_index]
        x = torch.hstack((xinst, x))

        idx_c = self.domain.parameter_idx["cloudfree_gemini"]["Cushing_scale_factor"]
        idx_scalingg =  self.domain.parameter_idx["cloudfree_gemini"]["scaling"]
        thetag = torch.hstack((thetag[:,:idx_c], thetag[:,idx_scalingg:]))  #removing the c and scaling 

        idx_scalingh =  self.domain.parameter_idx["cloudfree_hst"]["scaling"]
        b = torch.unsqueeze(b,1)
        bCh = torch.unsqueeze(thetah[:,idx_scalingh],1)

        theta = torch.hstack((thetag, bCh, b))
        x = torch.cat([x, xg, xh], dim=-1)

        return theta, x

    def forward(self, batch_dict: dict):

        batch_dict = self.modify_spec(batch_dict)
        batch_dict = self.modify_prior(batch_dict)

        return batch_dict 
    

    




    