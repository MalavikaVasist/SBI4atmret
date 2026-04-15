from importlib import import_module
from typing import Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

def _select_by_index(value, index):
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


class SchedulerConfig(BaseModel):
    """Pydantic model for scheduler configuration with validation."""
    model_config = ConfigDict(extra='allow')

    type: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = {}

    @field_validator('type')
    @classmethod
    def validate_scheduler_type(cls, v):
        """Validate that the scheduler type exists in torch.optim.lr_scheduler."""
        if v is None:
            return v
        try:
            getattr(import_module("torch.optim.lr_scheduler"), v)
        except AttributeError:
            raise ValueError(f"Invalid scheduler type '{v}'. Must be a valid torch.optim.lr_scheduler class.")
        return v


def get_scheduler_from_config(optimizer, scheduler_cfg: Optional[Dict[str, Any]], index: int = 0):
    """
    Convenience function to create scheduler from config dict.

    Args:
        optimizer: PyTorch optimizer instance.
        scheduler_cfg: Dictionary containing scheduler configuration or None.
        index: Index for selecting from list-valued parameters.

    Returns:
        Initialized PyTorch scheduler instance or None if no scheduler configured.
    """
    config = SchedulerConfig(scheduler_cfg)
    return config.get_scheduler(optimizer, index)
