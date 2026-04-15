from importlib import import_module
from typing import Dict, Any, Optional


def _select_by_index(value, index):
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


class SchedulerConfig:
    """Configuration class for PyTorch learning rate schedulers with built-in validation using Pydantic."""

    def __init__(self, config: Optional[Dict[str, Any]]):
        """
        Initialize SchedulerConfig with Pydantic validation.

        Args:
            config: Dictionary containing scheduler configuration or None

        Raises:
            ValidationError: If configuration is invalid
        """
        if config is None:
            self.config = None
        else:
            from ..config import SchedulerConfig as SchedulerConfigModel
            self.config = SchedulerConfigModel(**config)

    def _normalize_kwargs(self, index: int = 0) -> Dict[str, Any]:
        """
        Normalize configuration dictionary to scheduler kwargs.

        Args:
            index: Index for selecting from list-valued parameters.

        Returns:
            Dictionary of kwargs suitable for scheduler instantiation.
        """
        if self.config is None:
            return {}

        kwargs = {}
        if self.config.kwargs:
            kwargs.update(self.config.kwargs)

        # Add other scheduler parameters
        for field_name, field_info in self.config.model_fields.items():
            if field_name in {"type", "kwargs"}:
                continue
            value = getattr(self.config, field_name)
            if value is not None:
                kwargs[field_name] = _select_by_index(value, index)

        return kwargs

    def get_scheduler(self, optimizer, index: int = 0):
        """
        Create a scheduler instance from this configuration.

        Args:
            optimizer: PyTorch optimizer instance.
            index: Index for selecting from list-valued parameters.

        Returns:
            Initialized PyTorch scheduler instance or None if no scheduler configured.
        """
        if self.config is None or self.config.type is None:
            return None

        scheduler_kwargs = self._normalize_kwargs(index)
        SchedulerClass = getattr(import_module("torch.optim.lr_scheduler"), self.config.type)
        return SchedulerClass(optimizer, **scheduler_kwargs)


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
