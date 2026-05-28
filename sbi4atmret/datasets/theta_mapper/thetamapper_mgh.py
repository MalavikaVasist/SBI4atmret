from typing import List, Dict, Tuple
import torch
from sbi4atmret.datasets.theta_mapper.theta_mapping_metadata import ThetaMappingMetadata

# class MiriGeminiHSTThetaMapper:
#     """
#     Configuration-driven theta mapper using posterior_names and simulator.names.
    
#     Usage:
#         mapper = MiriGeminiHSTThetaMapper(
#             posterior_names=["R_pl", "mass", ..., "bfactor_noise_gemini", "bfactor_noise_hst", "bfactor_noise_miri"],
#             simulator_names={
#                 "miri": ["R_pl", "mass", ..., "bfactor_noise_miri"],
#                 "gemini": ["R_pl", "mass", ..., "mixture_fraction", "Cushing_scale_factor_g", "bfactor_noise_gemini"],
#                 "hst": ["R_pl", "mass", ..., "mixture_fraction", "Cushing_scale_factor_g", "bfactor_noise_hst"]
#             }
#         )
        
#         # Training
#         merged_theta, metadata = mapper.merge(thetam, thetag, thetah)
        
#         # Evaluation
#         thetam_post, thetag_post, thetah_post = mapper.split(posterior_theta)
#     """
    
#     def __init__(self, posterior_names: List[str], simulator_names: Dict[str, List[str]]):
#         """
#         Args:
#             posterior_names: Names of parameters in the merged posterior (e.g., shared + bg, bh, bm)
#             simulator_names: Dict mapping instrument name to its parameter names
#                             e.g., {"miri": [...], "gemini": [...], "hst": [...]}
#         """
#         self.posterior_names = posterior_names
#         self.simulator_names = simulator_names
#         self.instrument_names = list(simulator_names.keys())
        
#         # Build indices
#         self.posterior_param_indices = {name: idx for idx, name in enumerate(posterior_names)}
#         self.instrument_param_indices = {
#             inst: {name: idx for idx, name in enumerate(params)}
#             for inst, params in simulator_names.items()
#         }
        
#         # Identify shared parameters (present in all simulators' names)
#         all_simulator_sets = [set(names) for names in simulator_names.values()]
#         self.shared_param_names = sorted(list(set.intersection(*all_simulator_sets)))
        
#         # Identify instrument-specific parameters (in simulator but not shared across all)
#         self.instrument_specific_names = {}
#         for inst, names in simulator_names.items():
#             inst_specific = [n for n in names if n not in self.shared_param_names]
#             self.instrument_specific_names[inst] = inst_specific
    
#     def merge(self, theta_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ThetaMappingMetadata]:
#         """
#         Merge simulator-specific thetas into a posterior theta.
        
#         Concatenates: [shared_params | instrument_specific_params...]
        
#         Args:
#             theta_dict: {instrument_name: theta_tensor} where each tensor is [B, D_inst]
            
#         Returns:
#             merged_theta: [B, D_posterior] concatenated parameters
#             metadata: Information for reversing this operation
#         """
#         # Validate input
#         for inst in self.instrument_names:
#             if inst not in theta_dict:
#                 raise ValueError(f"Missing instrument {inst} in theta_dict")
        
#         batch_size = list(theta_dict.values())[0].shape[0]
#         device = list(theta_dict.values())[0].device
#         dtype = list(theta_dict.values())[0].dtype
        
#         # Extract shared parameters (use first simulator as source since they're identical)
#         first_inst = self.instrument_names[0]
#         shared_indices = [self.instrument_param_indices[first_inst][name] for name in self.shared_param_names]
#         shared_params = theta_dict[first_inst][:, shared_indices]  # [B, n_shared]
        
#         # Extract instrument-specific parameters in posterior order
#         instrument_params_list = [shared_params]
        
#         for inst in self.instrument_names:
#             param_names = self.instrument_specific_names[inst]
#             if param_names:
#                 indices = [self.instrument_param_indices[inst][name] for name in param_names]
#                 inst_params = theta_dict[inst][:, indices]
#                 instrument_params_list.append(inst_params)
        
#         # Concatenate to form posterior theta
#         merged_theta = torch.cat(instrument_params_list, dim=1)  # [B, D_posterior]
        
#         metadata = ThetaMappingMetadata(
#             posterior_names=self.posterior_names,
#             instrument_param_indices=self.instrument_param_indices,
#             posterior_param_indices=self.posterior_param_indices,
#             shared_param_names=self.shared_param_names,
#             instrument_specific_names=self.instrument_specific_names
#         )
        
#         return merged_theta, metadata
    
#     def split(self, merged_theta: torch.Tensor, metadata: ThetaMappingMetadata) -> Dict[str, torch.Tensor]:
#         """
#         Split posterior theta back to simulator-specific thetas.
        
#         Args:
#             merged_theta: [B, D_posterior] posterior samples
#             metadata: Information from merge operation
            
#         Returns:
#             Dict mapping instrument name to reconstructed theta tensors
#         """
#         batch_size = merged_theta.shape[0]
#         device = merged_theta.device
#         dtype = merged_theta.dtype
        
#         # Extract shared parameters (first n_shared columns)
#         n_shared = len(metadata.shared_param_names)
#         shared_params = merged_theta[:, :n_shared]  # [B, n_shared]
        
#         # Reconstruct each simulator's theta
#         theta_dict_split = {}
#         col_offset = n_shared
        
#         for inst in self.instrument_names:
#             # Create full theta for this instrument, starting with zeros
#             full_theta = torch.zeros(batch_size, len(self.simulator_names[inst]), 
#                                      device=device, dtype=dtype)
            
#             # Place shared parameters at their simulator-specific indices
#             for sim_idx, param_name in enumerate(metadata.shared_param_names):
#                 inst_idx = self.instrument_param_indices[inst][param_name]
#                 full_theta[:, inst_idx] = shared_params[:, sim_idx]
            
#             # Place instrument-specific parameters
#             inst_specific = metadata.instrument_specific_names[inst]
#             for i, param_name in enumerate(inst_specific):
#                 inst_idx = self.instrument_param_indices[inst][param_name]
#                 posterior_col = col_offset + i
#                 full_theta[:, inst_idx] = merged_theta[:, posterior_col]
            
#             theta_dict_split[inst] = full_theta
#             col_offset += len(inst_specific)
        
#         return theta_dict_split


from typing import Dict, List
import torch


class MiriGeminiHSTThetaMapper:

    def __init__(self, posterior_names: List[str], domain):

        self.posterior_names = sorted(posterior_names)
        self.posterior_set = set(posterior_names)

        self.domain = domain

        # ---- SORT INSTRUMENTS (IMPORTANT FIX #0) ----
        self.instrument_names = sorted(domain.simulator_dict.keys())

        # ---- simulator names ----
        self.simulator_names = {
            inst: domain.simulator_dict[inst].names
            for inst in self.instrument_names
        }

        # ---- use domain param_index (FIX #4) ----
        self.param_index = domain.param_index

        # ---- compute used params (restricted to posterior space) ----
        self.used_params = {
            inst: sorted([p for p in names if p in self.posterior_set])
            for inst, names in self.simulator_names.items()
        }

        # ---- shared params (sorted deterministic) ----
        self.shared_params = sorted(
            set.intersection(*[set(v) for v in self.used_params.values()])
        )

        # ---- instrument-specific params (sorted deterministic) ----
        self.inst_specific = {
            inst: sorted([
                p for p in self.used_params[inst]
                if p not in self.shared_params
            ])
            for inst in self.instrument_names
        }

    # =========================================================
    # MERGE
    # =========================================================

    def merge_theta(self, theta_dict: Dict[str, torch.Tensor]):
        '''
        input:
        theta_dict: {inst: theta_tensor} where each tensor is [B, D_inst] - order same as simulator.names
        returns:
        merged_theta: [B, D_posterior] concatenated parameters in order of sorted posterior names - sorted(shared_params) + sorted(instA_specific_params) + sorted(instB_specific_params) + ...
        where instA, instB is sorted too 
        '''


        first_inst = self.instrument_names[0]

        # ---- shared block ----
        shared_idx = [self.param_index[first_inst][p] for p in self.shared_params]
        shared = theta_dict[first_inst][:, shared_idx]

        parts = [shared]

        # ---- instrument-specific blocks ----
        for inst in self.instrument_names:

            spec = self.inst_specific[inst]
            if not spec:
                continue

            idx = [self.param_index[inst][p] for p in spec]
            parts.append(theta_dict[inst][:, idx])

        return torch.cat(parts, dim=1)

    # =========================================================
    # SPLIT
    # =========================================================

    def split_theta(self, posterior_theta: torch.Tensor):

        '''        
        input:
        merged_theta: [B, D_posterior] concatenated parameters in order of sorted posterior names - sorted(shared_params) + sorted(instA_specific_params) + sorted(instB_specific_params) + ...
        where instA, instB is sorted too 
        returns:
        theta_dict: {inst: theta_tensor} where each tensor is [B, D_inst] - order same as simulator.names
        '''


        batch_size = posterior_theta.shape[0]
        device = posterior_theta.device
        dtype = posterior_theta.dtype

        theta_dict = {}

        # ---- shared block ----
        n_shared = len(self.shared_params)
        shared = posterior_theta[:, :n_shared]
        offset = n_shared

        for inst in self.instrument_names:

            full_names = self.simulator_names[inst]

            inst_theta = torch.zeros(
                batch_size,
                len(full_names),
                device=device,
                dtype=dtype,
            )

            # ---- fill shared ----
            for i, name in enumerate(self.shared_params):
                if name in self.param_index[inst]:
                    inst_theta[:, self.param_index[inst][name]] = shared[:, i]

            # ---- fill instrument-specific ----
            spec = self.inst_specific[inst]

            for j, name in enumerate(spec):
                inst_theta[:, self.param_index[inst][name]] = posterior_theta[:, offset + j]

            offset += len(spec)

            theta_dict[inst] = inst_theta

        return theta_dict