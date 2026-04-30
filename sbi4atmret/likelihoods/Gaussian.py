import torch




def applynoise()-> Tensor:
    
    x, _ = noisybfactor(x, b, sigmaM)

    # xg = CushingScaleFactor(xg, thetag[:,-2])
    xg, _ = noisybfactor(xg, thetag[:,-1], sigmaG)
    
    thetahf = torch.flip(thetah, dims=(0,))
    # xh = CushingScaleFactor(xh, thetahf[:,-2])
    xh, _ = noisybfactor(xh, thetahf[:,-1], sigmaH)
    
    xinst = torch.hstack((xh, xg))[:,index_argsort]
    x = torch.hstack((xinst, x))
    
    thetag = torch.hstack((thetag[:,:-3], thetag[:,-1:]))  #removing the c and scaling 

    b = torch.unsqueeze(b,1)
    bCh = torch.unsqueeze(thetahf[:,-1],1)
    theta = torch.hstack((thetag, bCh, b))
    
    return theta, x