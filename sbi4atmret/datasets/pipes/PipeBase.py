import torch
from sbi4atmret.observations.ObservationBase import Observation


class BasePipe:
    def __init__(self, config):
        self.config = config
        self.param_index = self._build_param_index()
        obs = Observation(observation_config=config.observation_config,
                                        dataset_config=config.dataset_config,
                                        simulator_config=config.simulator_config)
        
        self.noise_dict = {}
        for inst, dict in obs.load_noise().items():
            self.noise_dict[inst]= dict['sigma']
        self.scale = obs.scale

    def _build_param_index(self)-> dict:
        """
        Maps parameter name → column index in theta
        """
        names = self.config.simulator.names
        return {name: i for i, name in enumerate(names)}


    def _apply_noise(self, x, b, sigma):
        b = torch.unsqueeze(b, 1)
        sigma_new = torch.sqrt(torch.Tensor(sigma)**2 + 10**b)
        error_new = sigma_new * torch.randn_like(x) * self.scale    
        return x + error_new , sigma_new
    
    def _apply_noise_new(self, x, theta, sigma):
        b = theta[:, :-1]
        b = torch.unsqueeze(b, 1)
        sigma_new = torch.sqrt(torch.Tensor(sigma)**2 + 10**b)
        error_new = sigma_new * torch.randn_like(x) * self.scale    
        return x + error_new , sigma_new

    def __call__(self, *batches):
        """
        batches = [(theta1, x1), (theta2, x2), ...]
        """
        return self.forward(*batches)

    def forward(self, *batches):
        raise NotImplementedError

    def _build_mask(self):
        return NotImplementedError
    



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
