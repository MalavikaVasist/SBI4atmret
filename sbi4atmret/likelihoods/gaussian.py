import torch
from NoiseBase import BaseNoise
from torch import Tensor
from ..datasets.pipes.PipeBase import BasePipe

class GaussianNoise(BaseNoise):
    
    def __init__(self, config):
        super().__init__(config)

    def forward(theta, x)-> Tensor:
        
        x, _ = self.noisybfactor(x, b, sigmaM)

        xg, _ = noisybfactor(xg, thetag[:,-1], sigmaG)
        
        thetahf = torch.flip(thetah, dims=(0,))
        xh, _ = noisybfactor(xh, thetahf[:,-1], sigmaH)
        
        xinst = torch.hstack((xh, xg))[:,index_argsort]
        x = torch.hstack((xinst, x))
        
        thetag = torch.hstack((thetag[:,:-3], thetag[:,-1:]))  #removing the c and scaling 

        b = torch.unsqueeze(b,1)
        bCh = torch.unsqueeze(thetahf[:,-1],1)
        theta = torch.hstack((thetag, bCh, b))
        
        return theta, x
    
