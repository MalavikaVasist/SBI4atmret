from importlib import import_module
from typing import Dict, Any, Optional
from ..config.configs import SchedulerConfig


def get_scheduler_from_config(optimizer, scheduler_cfg: Optional[Dict[str, Any]]):
    """
    Convenience function to create scheduler from config dict.

    Args:
        optimizer: PyTorch optimizer instance.
        scheduler_cfg: Dictionary containing scheduler configuration or None.

    Returns:
        Initialized PyTorch scheduler instance or None if no scheduler configured.
    """
    config = SchedulerConfig(**(scheduler_cfg or {}))
    return config.get_scheduler(optimizer)
