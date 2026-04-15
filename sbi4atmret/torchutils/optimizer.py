from importlib import import_module
from typing import Iterable, Dict, Any, Optional

import torch
from ..config import OptimizerConfig as OptimizerConfigModel


def _select_by_index(value, index):
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


class OptimizerConfig:
    """Configuration class for PyTorch optimizers with built-in validation using Pydantic."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OptimizerConfig with Pydantic validation.

        Args:
            config: Dictionary containing optimizer configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = OptimizerConfigModel(**config)

    def _normalize_kwargs(self, index: int = 0) -> Dict[str, Any]:
        """
        Normalize configuration dictionary to optimizer kwargs.

        Args:
            index: Index for selecting from list-valued parameters.

        Returns:
            Dictionary of kwargs suitable for optimizer instantiation.
        """
        kwargs = {}
        if self.config.kwargs:
            kwargs.update(self.config.kwargs)

        # Add lr with priority handling
        if self.config.lr is not None:
            kwargs["lr"] = _select_by_index(self.config.lr, index)
        elif self.config.init_lr is not None:
            kwargs["lr"] = _select_by_index(self.config.init_lr, index)

        # Add other optimizer parameters
        for field_name, field_info in self.config.model_fields.items():
            if field_name in {"type", "lr", "init_lr", "kwargs"}:
                continue
            value = getattr(self.config, field_name)
            if value is not None:
                kwargs[field_name] = _select_by_index(value, index)

        return kwargs

    def get_optimizer(self, model_parameters: Iterable, index: int = 0) -> torch.optim.Optimizer:
        """
        Create an optimizer instance from this configuration.

        Args:
            model_parameters: Model parameters to optimize.
            index: Index for selecting from list-valued parameters.

        Returns:
            Initialized PyTorch optimizer instance.
        """
        optimizer_kwargs = self._normalize_kwargs(index)
        OptimizerClass = getattr(import_module("torch.optim"), self.config.type)
        return OptimizerClass(model_parameters, **optimizer_kwargs)


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
