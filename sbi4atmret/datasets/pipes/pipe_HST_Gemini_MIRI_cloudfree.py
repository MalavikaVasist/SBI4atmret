

class Transforms:
    def __init__(self, x):
        self.x = x
    
    def transform_uniform(self, a, b, c, d):
        # Check if x is within the original range
        if a <= self.x <= b:
            # Apply the transformation formula
            y = c + ((self.x - a) * (d - c)) / (b - a)
            return y
        else:
            raise ValueError(f"x must be in the range [{a}, {b}]")


class BasePipe(ABC):
    def __init__(self, theta, x, return_loss: bool = True):
        self.x = x
        self.theta = theta


    def pipe():


    







def masking(wlen_geminisim) :
    obs_wlen_gemini = torch.from_numpy(obs_wlen_gemini)
    mask = torch.zeros(len(wlen_geminisim), dtype=torch.bool)
    for ind in range(len(obs_wlen_gemini)):
        mask = mask + torch.isin(torch.from_numpy(wlen_geminisim), obs_wlen_gemini[ind].item())
    return mask



def noisybfactor(x: Tensor, b: Tensor, sigma: Tensor, scale) -> Tensor:
        b = torch.unsqueeze(b, 1)
        sigma_new = torch.sqrt(torch.Tensor(sigma)**2 + 10**b)
        error_new = sigma_new * torch.randn_like(x) * scale    
        return x + error_new , sigma_new

def pipe( sim_models, simulator, return_loss = True ) -> Tensor: #, \

        theta, x = sim_models['cloudfree']['miri']
        thetag, xg = sim_models['cloudfree']['gemini']
        thetah, xh = sim_models['cloudfree']['hst']

        ##prior of wide noises
        # thetac[:,-1] = torch.hstack( [transform_uniform(thetac[i,-1], -15, -7, -13, -8) for i in range(len(thetac))] )
        thetah[:,-1] = torch.hstack( [transform_uniform(thetah[i,-1], -17, -11, -15, -7) for i in range(len(thetah))] )
        thetag[:,-1] = torch.hstack( [transform_uniform(thetag[i,-1], -17, -11, -15, -7) for i in range(len(thetag))] )
        
        thetahf = torch.flip(thetahc, dims=(0,))
        
        mask = masking(simulator.wavelength)
        scale = simulator.scale

        ## apply Noise
        x, _ = noisybfactor( x[:, 1:1299] , theta[:, -1], sigmaM, scale)
        xg, _ = noisybfactor( xg[:, mask], thetag[:, -1], sigmaG, scale)
        xh, _ = noisybfactor(xh, thetah[:, -1], sigmaH, scale)


        xinst = torch.hstack((xh, xg))[:,index_argsort]
        x = torch.hstack((xinst, x))
        thetag = torch.hstack((thetag[:,:-3], thetag[:,-1:]))  #removing the c and scaling 
        b = torch.unsqueeze(b,1)
        bCh = torch.unsqueeze(thetahf[:,-1],1)
        theta = torch.hstack((thetag, bCh, b))

        theta, x = theta.cuda(), x.cuda()

        if return_loss :
            return loss(theta, x)
        else:
            return theta, x


from abc import ABC, abstractmethod
import torch
from torch import Tensor





class DataPipeline:
    def __init__(self, noise_model, mask, sigmas):
        ...

    def __call__(self, batch):
        theta, x = ...
        return theta, x