import torch
from sbi4atmret.observations.ObservationBase import Observation


class BasePipe:
    def __init__(self, config, simulators, observation):
        self.config = config
        self.simulators = simulators
        self.observation = observation
        
        self.noise_dict = self._build_obs_noise()
        self.wlen_obs_dict = self._build_obs_wlens()
        
        self.scale = observation.scale

        self.wlen_sim_dict = self._build_sim_wlens()

        self.parameter_index = self._build_param_index()


    def _build_param_index(self)-> dict:
        """
        Maps parameter name → column index in theta
        """
        param_index = {}

        for simname, simulator in self.simulators.items():
            names = simulator.names
            param_index[simname]= {name: i for i, name in enumerate(names)}    
        
        return param_index

    def _build_sim_wlens(self):
        '''
        returns:
        wlens = {
                "cloudfree_miri" : wlen, 
                "cloudfree_hst": wlen, 
                "cloudfre_gemini" : wlen, 

                "..": ....

                    }
        '''

        wlens = {}

        for name, sim in self.simulators.items():

            wlens[name] = sim.wavelength

        return wlens

    def _build_obs_wlens(self):
        '''
        returns:
        wlens = {
                "miri" : wlen, 
                "hst": wlen, 
                "gemini" : wlen, 
                    }
        '''

        wlens = {}

        for inst, d in self.observation.load_noise().items():
            wlens[inst] = d['wlen']

        return wlens
    
    def _build_obs_noise(self):
        '''
        returns:
        noise = {
                "miri" : sigmaM, 
                "hst": sigmaH, 
                "gemini" : sigmaG, 
                    }
        '''
        noise = {}
        for inst, d in self.observation.load_noise().items():
            noise[inst]= d['sigma']
        
        return noise


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
