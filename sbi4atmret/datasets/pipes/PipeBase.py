import torch


class BasePipe:
    def __init__(self, config, domain):
        self.config = config
        self.domain = domain 

    def __call__(self, *batches):
        """
        batches = [(theta1, x1), (theta2, x2), ...]
        """
        return self.forward(*batches)

    def forward(self, *batches):
        raise NotImplementedError

    def _build_mask(self):
        return NotImplementedError

    @property
    def param_index(self):
        return self.domain.param_index

    
    



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
