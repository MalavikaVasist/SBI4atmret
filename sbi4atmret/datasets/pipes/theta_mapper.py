"""
Bidirectional theta mapping for multi-simulator parameter merging.
Handles forward merging (training) and backward splitting (evaluation).
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import numpy as np


@dataclass
class ThetaMappingMetadata:
    """Stores information needed to deterministically reverse theta merging"""
    # Indices that define parameter locations
    idx_c: int  # gemini mixture fraction
    idx_scalingg: int  # gemini Cushing scale
    idx_scalingh: int  # hst bfactor noise
    idx_scalingm: int  # miri bfactor noise
    
    # Original shapes before any operations
    orig_shapes: Dict[str, Tuple[int, ...]]
    
    # Extracted values (what we pulled out during merge)
    extracted_values: Dict[str, torch.Tensor]


class MiriGeminiHSTThetaMapper:
    """
    Maps theta parameters between simulator-specific and merged representations.
    
    Forward (merge): 3 separate simulator thetas → 1 global theta for training
    Backward (split): 1 global theta (posterior samples) → 3 simulator thetas for evaluation
    
    Usage:
        mapper = MiriGeminiHSTThetaMapper(domain)
        
        # Training
        merged_theta, metadata = mapper.merge(thetam, thetag, thetah)
        
        # Evaluation
        thetam_post, thetag_post, thetah_post = mapper.split(posterior_theta, metadata)
    """
    
    def __init__(self, domain):
        """
        Args:
            domain: Object with parameter_idx dict containing indices for each simulator
        """
        self.domain = domain
        self.param_idx = domain.parameter_idx
    
    def merge(self, thetam: torch.Tensor, thetag: torch.Tensor, thetah: torch.Tensor) -> Tuple[torch.Tensor, ThetaMappingMetadata]:
        """
        Merge 3 simulator thetas into 1 global theta.
        
        Removes redundant parameters that are shared or transformed:
        - Removes gemini mxture_fraction and Cushing_scale_factor_g (not used in global fit)
        - Extracts hst bfactor_noise_gemini (converted to scaling)
        - Extracts miri bfactor_noise_miri (converted to scaling)
        
        Args:
            thetam: [B, D_miri] MIRI parameters
            thetag: [B, D_gemini] Gemini parameters
            thetah: [B, D_hst] HST parameters
            
        Returns:
            merged_theta: [B, D_global] merged parameters
            metadata: Information to reverse this operation
        """
        # Get parameter indices
        idx_c = self.param_idx["cloudfree_gemini"]["mxture_fraction"]
        idx_scalingg = self.param_idx["cloudfree_gemini"]["Cushing_scale_factor_g"]
        idx_scalingh = self.param_idx["cloudfree_hst"]["bfactor_noise_gemini"]
        idx_scalingm = self.param_idx["cloudfree_miri"]["bfactor_noise_miri"]
        
        # Store original shapes
        orig_shapes = {
            "miri": thetam.shape,
            "gemini": thetag.shape,
            "hst": thetah.shape
        }
        
        # Extract values we'll need for reversal
        extracted_values = {
            "gemini_c": thetag[:, idx_c:idx_c+1].clone(),  # [B, 1]
            "gemini_scalingg": thetag[:, idx_scalingg:idx_scalingg+1].clone(),  # [B, 1]
            "hst_scalingh": thetah[:, idx_scalingh:idx_scalingh+1].clone(),  # [B, 1]
            "miri_scalingm": thetam[:, idx_scalingm:idx_scalingm+1].clone(),  # [B, 1]
        }
        
        # Build core gemini theta (remove redundant columns)
        thetag_core = torch.cat([
            thetag[:, :idx_c],  # All params before mxture_fraction
            thetag[:, idx_c+1:idx_scalingg],  # Params between c and scalingg
            thetag[:, idx_scalingg+1:]  # Params after scalingg
        ], dim=1)
        
        # Extract scaling factors as separate columns
        b_h = extracted_values["hst_scalingh"]  # [B, 1]
        b_m = extracted_values["miri_scalingm"]  # [B, 1]
        
        # Concatenate: [core_gemini | scaling_hst | scaling_miri]
        merged_theta = torch.cat([thetag_core, b_h, b_m], dim=1)
        
        metadata = ThetaMappingMetadata(
            idx_c=idx_c,
            idx_scalingg=idx_scalingg,
            idx_scalingh=idx_scalingh,
            idx_scalingm=idx_scalingm,
            orig_shapes=orig_shapes,
            extracted_values=extracted_values
        )
        
        return merged_theta, metadata
    
    def split(self, merged_theta: torch.Tensor, metadata: ThetaMappingMetadata) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split global theta (posterior samples) back to 3 simulator thetas.
        
        Uses metadata to deterministically reverse the merge operation.
        
        Args:
            merged_theta: [B, D_global] posterior samples (merged representation)
            metadata: Information from the merge operation
            
        Returns:
            thetam_split: [B, D_miri] MIRI parameters
            thetag_split: [B, D_gemini] Gemini parameters
            thetah_split: [B, D_hst] HST parameters
        """
        # Recover the indices from metadata
        idx_c = metadata.idx_c
        idx_scalingg = metadata.idx_scalingg
        idx_scalingh = metadata.idx_scalingh
        idx_scalingm = metadata.idx_scalingm
        
        # Recover original shapes
        n_gemini = metadata.orig_shapes["gemini"][1]
        n_hst = metadata.orig_shapes["hst"][1]
        n_miri = metadata.orig_shapes["miri"][1]
        
        # How many core gemini parameters? (after removing 2 columns)
        n_core_gemini = n_gemini - 2
        
        # Split merged_theta back
        thetag_core = merged_theta[:, :n_core_gemini]
        b_h = merged_theta[:, n_core_gemini:n_core_gemini+1]
        b_m = merged_theta[:, n_core_gemini+1:n_core_gemini+2]
        
        # Reconstruct full gemini theta by inserting removed values
        # Insert back at correct positions
        thetag_split = torch.cat([
            thetag_core[:, :idx_c],  # Params before mxture_fraction
            metadata.extracted_values["gemini_c"],  # mxture_fraction
            thetag_core[:, idx_c:idx_scalingg-1],  # Params between (adjusted for 1 removed)
            metadata.extracted_values["gemini_scalingg"],  # Cushing_scale_factor_g
            thetag_core[:, idx_scalingg-1:]  # Params after (adjusted for 2 removed)
        ], dim=1)
        
        # Reconstruct HST theta
        # Just need to put back the scaling factor at correct position
        thetah_split = torch.cat([
            metadata.extracted_values["hst_scalingh"],
            torch.zeros(merged_theta.shape[0], n_hst - 1, device=merged_theta.device)
        ], dim=1)
        
        # Reconstruct MIRI theta
        thetam_split = torch.cat([
            metadata.extracted_values["miri_scalingm"],
            torch.zeros(merged_theta.shape[0], n_miri - 1, device=merged_theta.device)
        ], dim=1)
        
        return thetam_split, thetag_split, thetah_split
