import torch


class BasePipe:
    def __init__(self, config, domain):
        self.config = config
        self.domain = domain 
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

    @property
    def param_index(self):
        return self.domain.param_index

    def __call__(self, *batches):
        """
        batches = [(theta1, x1), (theta2, x2), ...]
        """
        return self.forward(*batches)
    
    def modify_spec(self, batch_dict):
        return NotImplementedError
    

    def modify_theta(self, batch_dict):
        return NotImplementedError


    def forward(self, batch_dict: dict):

        batch_dict = self.modify_spec(batch_dict)
        batch_dict = self.modify_theta(batch_dict)

        return batch_dict 



    
    



    def theta_to_dict(self, theta):
        names = self.config.simulator.names

        if theta.shape[-1] != len(names):
            raise ValueError(
                f"Theta dim {theta.shape[-1]} != number of names {len(names)}"
            )

        # Case 1: batched [B, D]
        if theta.ndim == 2:
            return {name: theta[:, i] for i, name in enumerate(names)}

        # Case 2: single sample [D]
        elif theta.ndim == 1:
            return {name: theta[i] for i, name in enumerate(names)}

        else:
            raise ValueError(f"Unsupported theta shape: {theta.shape}")
    

    def dict_to_theta(self, theta_dict):
        names = self.config.simulator.names
        values = [theta_dict[name] for name in names]

        # check if batched
        if values[0].ndim == 1:
            return torch.stack(values, dim=-1)   # [B, D]
        else:
            return torch.tensor(values)          # [D]
