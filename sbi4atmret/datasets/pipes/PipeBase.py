import torch
import numpy as np


from sbi4atmret.datasets.theta_mapper.thetamapperbase import BaseThetaMapper
from sbi4atmret.utils.general import simname_from_instrument


class BasePipe:
    def __init__(self, 
                 domain= None, 
                 posterior_names=None):
        
        self.domain = domain
        self.posterior_names = posterior_names
        self.theta_mapper = BaseThetaMapper(domain=domain, 
                                            posterior_param_names=posterior_names)

       
    @property
    def param_index(self):
        return self.domain.param_index

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def modify_spec(self, batch_dict):
        return NotImplementedError

    def modify_theta(self, batch_dict):
        raise NotImplementedError


    def forward(self, batch_dict: dict, mode="train"):

        batch_dict = self.modify_spec(batch_dict)

        if mode == "train":
            batch_dict = self.modify_theta(batch_dict)

        return batch_dict 
    
    def merge_spec(self, batch_dict):
        ## spec
        x_dict = {
            inst: batch_dict[inst][1]
            for inst in self.theta_mapper.simulator_names
        }

        ## merge x: concatenate all instruments in the same order as 
        ## Observation.sort_idx was built (instruments key order from config)
        ## then sort by wavelength using sort_idx
        obs_inst_order = list(self.domain.observation.instruments.keys())
        x_concat = torch.hstack([
            x_dict[simname_from_instrument(inst, self.theta_mapper.simulator_names)]
            for inst in obs_inst_order
        ])
        x = x_concat[:, self.domain.observation.sort_idx]

        return x
    
    
    def merge_theta(self, batch_dict):
        ## theta
        theta_dict = {
                    inst: batch_dict[inst][0]
                    for inst in self.theta_mapper.simulator_names
                }
        theta = self.theta_mapper.merge_theta(theta_dict)

        # store full dict for evaluation consistency
        self._last_theta_dict = theta_dict

        return theta

    def build_input(self, batch_dict, mode="train"):

        x = self.merge_spec(batch_dict)
        if mode == "train":
            theta = self.merge_theta(batch_dict)
        elif mode == "eval":
            theta  = None
        
        return theta, x


    def split_theta(self, theta_post):
        return self.theta_mapper.split_theta(theta_post)

    def split_spec(self, x):
        return self.domain.observation._get_observation_dict(full_flux=x.cpu().numpy())


