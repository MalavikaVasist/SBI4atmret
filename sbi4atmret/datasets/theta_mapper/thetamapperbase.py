from typing import List, Dict, Tuple
import torch


class ThetaMapper:
    """
    CPU-only mapper.

    All indices are stored on CPU.
    No .to(device) caching.
    Device transfer happens outside this class.
    """

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

        claimed = torch.zeros(self.n_total, dtype=torch.bool)  

        for inst in self.instrument_names:

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
                empty = torch.empty(0, dtype=torch.long)

                self.indices["sim_i"][inst] = empty
                self.indices["sim_j"][inst] = empty
                self.indices["merge_i"][inst] = empty
                self.indices["merge_j"][inst] = empty

    # ------------------------------------------------------------
    # PURE CPU OPS
    # ------------------------------------------------------------

    def merge_theta(self, theta_dict):

        first = next(iter(theta_dict.values()))

        B = first.shape[0]

        merged = torch.zeros((B, self.n_total), dtype=first.dtype)

        for inst in self.instrument_names:

            j = self.indices["merge_j"][inst]
            i = self.indices["merge_i"][inst]

            merged[:, i] = theta_dict[inst][:, j]

        return merged

    def split_theta(self, merged_theta):

        B = merged_theta.shape[0]

        theta_dict = {}

        for inst in self.instrument_names:

            j = self.indices["sim_j"][inst]
            i = self.indices["sim_i"][inst]

            D_inst = j.numel()

            inst_theta = torch.zeros((B, D_inst), dtype=merged_theta.dtype)

            inst_theta[:, j] = merged_theta[:, i]

            theta_dict[inst] = inst_theta

        return theta_dict