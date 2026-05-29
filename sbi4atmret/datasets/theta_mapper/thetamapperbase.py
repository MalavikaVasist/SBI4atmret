from typing import List, Dict, Tuple
import torch


class ThetaMapper:
    '''
    posterior_names = [param names in posterior]  => [p]
    posterior_index = {param name: index in posterior} => {p: i}

    simulator_param_names =
        {instrument: [param names in simulator]} => {inst: [p]}

    sim_valid_j =
        simulator-space column indices => {inst: tensor([j])}

    sim_valid_i =
        corresponding posterior-space column indices => {inst: tensor([i])}

    n_total = total number of params in posterior
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

        self.sim_valid_j = {}
        self.sim_valid_i = {}

        self.sim_valid_j_device = {}
        self.sim_valid_i_device = {}

        for inst in self.instrument_names:

            idx = [
                self.posterior_index.get(p, None)
                for p in self.simulator_param_names[inst]
            ]

            valid = [(j, i) for j, i in enumerate(idx) if i is not None]

            if valid:
                j_idx, i_idx = zip(*valid)

                self.sim_valid_j[inst] = torch.tensor(j_idx, dtype=torch.long)
                self.sim_valid_i[inst] = torch.tensor(i_idx, dtype=torch.long)
            else:
                self.sim_valid_j[inst] = torch.empty(0, dtype=torch.long)
                self.sim_valid_i[inst] = torch.empty(0, dtype=torch.long)

            self.sim_valid_j_device[inst] = {}
            self.sim_valid_i_device[inst] = {}

    def _get_device_indices(self, inst, device):

        # create cache dicts if first time using this device
        if device not in self.sim_valid_j_device[inst]:

            self.sim_valid_j_device[inst][device] = (
                self.sim_valid_j[inst].to(device)
            )

            self.sim_valid_i_device[inst][device] = (
                self.sim_valid_i[inst].to(device)
            )

        return (
            self.sim_valid_j_device[inst][device],
            self.sim_valid_i_device[inst][device]
        )
        
    def merge_theta(self, theta_dict):

        first = next(iter(theta_dict.values()))

        B = first.shape[0]
        device = first.device
        dtype = first.dtype

        merged = torch.zeros((B, self.n_total), device=device, dtype=dtype)

        filled = torch.zeros(self.n_total, dtype=torch.bool, device=device)

        for inst in self.instrument_names:

             # get cached device tensors
            j_idx, i_idx = self._get_device_indices(inst, device)

            keep = ~filled[i_idx]

            if keep.any():

                j = j_idx[keep]
                i = i_idx[keep]

                merged[:, i] = theta_dict[inst][:, j]

                filled[i] = True

        return merged
    

    def split_theta(self, merged_theta):

        B = merged_theta.shape[0]
        device = merged_theta.device
        dtype = merged_theta.dtype

        theta_dict = {}

        for inst in self.instrument_names:

            D_inst = len(self.simulator_param_names[inst])

            inst_theta = torch.zeros((B, D_inst), device=device, dtype=dtype)

            # get cached device tensors
            j, i = self._get_device_indices(inst, device)

            inst_theta[:, j] = merged_theta[:, i]

            theta_dict[inst] = inst_theta

        return theta_dict