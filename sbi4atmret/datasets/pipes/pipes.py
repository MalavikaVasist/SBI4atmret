
from PipeBase import BasePipe
from sbi4atmret.utils.general import transform_uniform
import torch


class MiriGeminiHSTPipe(BasePipe):
    def __init__(self, config):
        super().__init__(config)
        self._mask = self._build_mask()
        

    def _build_mask(self):
        ## Masking gaps in observation
        obs_wlen_gemini = torch.from_numpy(obs_wlen_gemini)
        mask = torch.zeros(len(wlen_geminisim), dtype=torch.bool)
        for ind in range(len(obs_wlen_gemini)):
            mask = mask + torch.isin(torch.from_numpy(wlen_geminisim), obs_wlen_gemini[ind].item())
        
        return mask

    def modify_spec(self, batch_dict):
        cf = batch_dict["cloudfree"]

        thetac, xc   = cf["miri"]
        thetagc, xgc = cf["gemini"]
        thetahc, xhc = cf["hst"]

        xc  = xc[:, 1:1299]
        xgc = xgc[:, self.mask]

        cf["miri"]   = (thetac, xc)
        cf["gemini"] = (thetagc, xgc)
        cf["hst"]    = (thetahc, xhc)

        return batch_dict

    def modify_prior(self, batch_dict):
        cf = batch_dict["cloudfree"]

        thetac, xc   = cf["miri"]
        thetagc, xgc = cf["gemini"]
        thetahc, xhc = cf["hst"]

        # vectorized transform 
        thetahc[:, -1] = transform_uniform(thetahc[:, -1], -17, -11, -15, -7)
        thetagc[:, -1] = transform_uniform(thetagc[:, -1], -17, -11, -15, -7)

        cf["miri"]   = (thetac, xc)
        cf["gemini"] = (thetagc, xgc)
        cf["hst"]    = (thetahc, xhc)

        return batch_dict

    def build_input(self, batch_dict):
        cf = batch_dict["cloudfree"]

        thetac, xc   = cf["miri"]
        _, xgc       = cf["gemini"]
        _, xhc       = cf["hst"]

        theta = thetac
        x = torch.cat([xc, xgc, xhc], dim=-1)

        return theta, x

    def forward(self, batch_dict: dict):

        batch_dict = self.modify_spec(batch_dict)
        batch_dict = self.modify_prior(batch_dict)

        return batch_dict 
    






    