from importlib import import_module
from typing import Iterable, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import torch
from ..config import OptimizerConfig as OptimizerConfigModel
import torch.optim as optim


class OptimizerConfig(BaseModel):
    """Pydantic model for optimizer configuration with validation."""
    model_config = ConfigDict(extra='allow')

    type: str
    lr: Optional[float] = None
    init_lr: Optional[float] = None
    weight_decay: Optional[float] = None
    kwargs: Optional[Dict[str, Any]] = {}

    @field_validator('type')
    @classmethod
    def validate_optimizer_type(cls, v):
        """Validate that the optimizer type exists in torch.optim."""
        try:
            getattr(import_module("torch.optim"), v)
        except AttributeError:
            raise ValueError(f"Invalid optimizer type '{v}'. Must be a valid torch.optim optimizer class.")
        return v

    @model_validator(mode='after')
    def validate_lr_fields(self):
        """Ensure either lr or init_lr is provided."""
        if self.lr is None and self.init_lr is None:
            raise ValueError("Either 'lr' or 'init_lr' must be provided")
        return self


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
    config = OptimizerConfig(optimizer_cfg)
    return config.get_optimizer(model_parameters, index)
