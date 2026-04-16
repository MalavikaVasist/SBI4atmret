from importlib import import_module
from typing import Iterable, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import torch
from ..config.configs import OptimizerConfig
import torch.optim as optim


def get_optimizer_from_config(model_parameters: Iterable, optimizer_cfg: Dict[str, Any], index: int = 0) -> torch.optim.Optimizer:
    """
    Convenience function to create optimizer from config dict.

    Args:
        model_parameters: Model parameters to optimize.
        optimizer_cfg: Dictionary containing optimizer configuration.
        index: Index for selecting from list-valued parameters.

    Returns:
        Initialized PyTorch optimizer instance.
    """
    config = OptimizerConfig(**optimizer_cfg)
    return config.get_optimizer(model_parameters, index)
