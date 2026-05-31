from typing import List, Dict, Tuple
import torch


class ThetaMapper:
    '''
    posterior_names = [param names in posterior]  => [p]
    posterior_index = {param name: index in posterior} => {p: i}

    simulator_param_names =
        {instrument: [param names in simulator]} => {inst: [p]}

    sim_j =
        simulator-space column indices => {inst: tensor([j])}

    sim_i =
        corresponding posterior-space column indices => {inst: tensor([i])}

    merge_j =   sim_j columns that are in posterior => {inst: tensor([j])}

    merge_i =   corresponding sim_i columns => {inst: tensor([i])}

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

        self.indices = {
                        "sim_i": {},
                        "sim_j": {},
                        "merge_i": {},
                        "merge_j": {},
                    }

        self.indices_device = {
                        "sim_i": {},
                        "sim_j": {},
                        "merge_i": {},
                        "merge_j": {},
                    }

        claimed = torch.zeros(self.n_total, dtype=torch.bool)

        for inst in self.instrument_names:

            self.indices_device["sim_i"][inst] = {}
            self.indices_device["sim_j"][inst] = {}
            self.indices_device["merge_i"][inst] = {}
            self.indices_device["merge_j"][inst] = {}

            idx = [
                self.posterior_index.get(p, None)
                for p in self.simulator_param_names[inst]
            ]

            valid = [(j, i) for j, i in enumerate(idx) if i is not None]

            if valid:
                j_idx, i_idx = zip(*valid)
                j_idx = torch.tensor(j_idx, dtype=torch.long)
                i_idx = torch.tensor(i_idx, dtype=torch.long)

                self.indices["sim_i"][inst] = i_idx
                self.indices["sim_j"][inst] = j_idx

                keep = ~claimed[i_idx]
                self.indices["merge_i"][inst] = i_idx[keep]
                self.indices["merge_j"][inst] = j_idx[keep]
                claimed[i_idx[keep]] = True


            else:
                self.indices["merge_i"][inst] = torch.empty(0, dtype=torch.long)
                self.indices["merge_j"][inst] = torch.empty(0, dtype=torch.long)
                self.indices["sim_i"][inst] = torch.empty(0, dtype=torch.long)
                self.indices["sim_j"][inst] = torch.empty(0, dtype=torch.long)

    def _get_indices(self, kind, inst, device):

        if not isinstance(device, torch.device):
            device = torch.device(device)

        if device not in self.indices_device[kind]:
            self.indices_device[kind][device] = {}

        if inst not in self.indices_device[kind][device]:
            self.indices_device[kind][device][inst] = (
                self.indices[kind][inst].to(device)
            )

        return self.indices_device[kind][device][inst]

    def merge_theta(self, theta_dict):

        first = next(iter(theta_dict.values()))

        B = first.shape[0]
        device = first.device
        dtype = first.dtype

        merged = torch.zeros((B, self.n_total), device=device, dtype=dtype)

        for inst in self.instrument_names:
            j = self._get_indices("merge_j", inst, device)
            i = self._get_indices("merge_i", inst, device)

            merged[:, i] = theta_dict[inst][:, j]

        return merged


    def split_theta(self, merged_theta):

        B = merged_theta.shape[0]
        device = merged_theta.device

        theta_dict = {}

        for inst in self.instrument_names:

            i = self._get_indices("sim_i", inst, device)
            j = self._get_indices("sim_j", inst, device)

            D_inst = j.numel()

            inst_theta = merged_theta.new_zeros((B, D_inst))

            inst_theta[:, j] = merged_theta[:, i]

            theta_dict[inst] = inst_theta

        return theta_dict