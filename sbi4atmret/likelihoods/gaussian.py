import torch


class GaussianNoise():
    
    def __init__(self):
        pass

    def noise(x, theta, sigma):
        b = torch.unsqueeze(b, 1)
        sigma_new = torch.sqrt(torch.Tensor(sigma)**2 + 10**b)
        error_new = sigma_new * torch.randn_like(x) * simulator_miri_cloudfree.scale    
        return x + error_new , sigma_new

    def forward(theta, x)-> Tensor:
        
        x, _ = noisybfactor(x, b, sigmaM)

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
    
