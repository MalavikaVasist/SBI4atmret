"""
Bidirectional theta mapping for multi-simulator parameter merging.

Maps between simulator-specific parameters and a merged posterior representation.

Key insight:
- Each simulator has its own parameter names (simulator.names)
- The posterior has a specific order (posterior_names)
- Shared parameters are those present in both; instrument-specific ones are kept separate
- During merge: extract instrument-specific params and concatenate to shared params
- During split: reconstruct each simulator's params from the posterior
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch


@dataclass
class ThetaMappingMetadata:
    """Metadata needed to bidirectionally map theta between representations"""
    
    # Posterior parameter structure
    posterior_names: List[str]
    
    # Mapping: which simulator param is at which index
    instrument_param_indices: Dict[str, Dict[str, int]]  # {instrument: {param_name: idx}}
    
    # Mapping: which posterior param is at which index  
    posterior_param_indices: Dict[str, int]  # {param_name: idx}
    
    # For each instrument, which posterior params are relevant
    shared_param_names: List[str]  # Params shared across all simulators
    instrument_specific_names: Dict[str, List[str]]  # {instrument: [param_names used only by this instrument]}
