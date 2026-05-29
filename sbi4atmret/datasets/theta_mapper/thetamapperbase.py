from typing import List, Dict, Tuple
import torch


class ThetaMapper:
    '''
    posterior_names = [param names in posterior]  => [p]
    posterior_index = {param name: index in posterior} => {p: i}
    sim_param_names = {instrument: [param names in simulator]} => {inst: [p]}
    sim_param_index = {instrument: {param name: index in simulator}} => {inst: {p: j}}
    sim_to_global_idx = {instrument: [global index of sim params in posterior]} => {inst: [i]}
    n_total = total number of params in posterior
    self.instrument_names = sorted list of instruments => [inst]
    '''

    def __init__(self, domain):

        self.domain = domain

        self.instrument_names = sorted(domain.simulator_dict.keys())
        self.simulator_param_names = {
                                inst: domain.simulator_dict[inst].names
                                for inst in self.instrument_names
                            }
        self.posterior_names = self.domain.pipe.theta_mapper.posterior_names
        self.posterior_index = {p: i for i, p in enumerate(self.posterior_names)}
        self.n_total = len(self.posterior_names)     

        self.sim_to_global_idx = {
                            inst: [
                                self.posterior_index.get(p, None)
                                for p in self.simulator_param_names[inst]
                            ]
                            for inst in self.instrument_names
                        }
        
    def merge_theta(self, theta_dict: Dict[str, torch.Tensor]):
        B = next(iter(theta_dict.values())).shape[0]
        device = next(iter(theta_dict.values())).device
        dtype = next(iter(theta_dict.values())).dtype

        merged = torch.zeros((B, self.n_total), device=device, dtype=dtype)
        filled = torch.zeros(self.n_total, dtype=torch.bool)

        for inst in self.instrument_names:
            idx = self.sim_to_global_idx[inst]
            inst_theta = theta_dict[inst]

            for j, i in enumerate(idx):
                if i is None: ## if param not in posterior, skip
                    continue
                if filled[i]: ## if param repeated across instruments, skip (already filled by previous instrument)
                    continue
                merged[:, i] = inst_theta[:, j]
                filled[i] = True

        return merged


    def split_theta(self, merged_theta: torch.Tensor):
        B = merged_theta.shape[0]
        device = merged_theta.device
        dtype = merged_theta.dtype

        theta_dict = {}

        for inst in self.instrument_names:
            idx = self.sim_to_global_idx[inst]
            D_inst = len(idx)

            inst_theta = torch.zeros((B, D_inst), device=device, dtype=dtype)

            for j, i in enumerate(idx):
                if i is None: ## if param not in posterior, keeps values at 0 (since not in posterior)
                    continue
                inst_theta[:, j] = merged_theta[:, i]

            theta_dict[inst] = inst_theta

        return theta_dict