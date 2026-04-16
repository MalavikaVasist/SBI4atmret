from typing import Any, Dict, List, Optional, Union
from importlib import import_module

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _select_by_index(value, index):
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


class ParameterConfig(BaseModel):
    """Configuration for a single parameter with bounds."""
    name: str
    lower: float
    upper: float
    default: Optional[float] = None


class PriorConfig(BaseModel):
    """Configuration for prior distribution."""
    parameters: List[ParameterConfig] = Field(alias="PARAMETERS")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding layers."""
    miri: Union[List[int], int]
    gemini: Union[List[int], int]
    miri_output: Union[List[int], int]
    gemini_output: Union[List[int], int]


class ModelConfig(BaseModel):
    """Configuration for model architecture."""
    model_config = ConfigDict(extra='allow')

    embedding: EmbeddingConfig
    hidden_features: Union[List[int], int]
    no_of_params: Union[List[int], int]
    transforms: Union[List[int], int]
    signal: Union[List[int], int]
    batch_size: Optional[Union[List[int], int]] = None


class FlowConfig(BaseModel):
    """Configuration for a flow-based estimator."""
    model_config = ConfigDict(extra='allow')

    type: str
    hidden_features_no: Optional[Union[List[int], int]] = None
    hidden_features: Optional[Union[List[List[int]], List[int], int]] = None
    transforms: Optional[Union[List[int], int]] = None
    signal: Optional[Union[List[int], int]] = None
    kwargs: Optional[Dict[str, Any]] = None


class EstimatorConfig(BaseModel):
    """Configuration for estimator settings."""
    model_config = ConfigDict(extra='allow')

    flow: FlowConfig


class OptimizerConfig(BaseModel):
    """Configuration for optimizer settings."""
    model_config = ConfigDict(extra='allow')

    type: str
    lr: Optional[Union[List[float], float]] = None
    init_lr: Optional[Union[List[float], float]] = None
    weight_decay: Optional[Union[List[float], float]] = None
    kwargs: Optional[Dict[str, Any]] = None

    @field_validator('type')
    @classmethod
    def validate_optimizer_type(cls, v):
        try:
            getattr(import_module("torch.optim"), v)
        except AttributeError:
            raise ValueError(f"Invalid optimizer type '{v}'. Must be a valid torch.optim optimizer class.")
        return v

    @model_validator(mode='after')
    def validate_lr_fields(self):
        if self.lr is None and self.init_lr is None:
            raise ValueError("Either 'lr' or 'init_lr' must be provided")
        return self

    def get_optimizer(self, model_parameters, index: int = 0):
        """
        Create and return a PyTorch optimizer instance based on the configuration.

        Args:
            model_parameters: Model parameters to optimize.
            index: Index for selecting from list-valued parameters.

        Returns:
            Initialized PyTorch optimizer instance.
        """
        opt_class = getattr(torch.optim, self.type)
        kwargs = self.kwargs.copy() if self.kwargs else {}

        # Select lr from lr or init_lr, handling lists
        lr_value = self.lr if self.lr is not None else self.init_lr
        if lr_value is not None:
            kwargs['lr'] = _select_by_index(lr_value, index)

        # Handle weight_decay if provided
        if self.weight_decay is not None:
            kwargs['weight_decay'] = _select_by_index(self.weight_decay, index)

        return opt_class(model_parameters, **kwargs)


class SchedulerConfig(BaseModel):
    """Configuration for scheduler settings."""
    model_config = ConfigDict(extra='allow')

    type: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = None

    @field_validator('type')
    @classmethod
    def validate_scheduler_type(cls, v):
        if v is None:
            return v
        try:
            getattr(import_module("torch.optim.lr_scheduler"), v)
        except AttributeError:
            raise ValueError(f"Invalid scheduler type '{v}'. Must be a valid torch.optim.lr_scheduler class.")
        return v

    def get_scheduler(self, optimizer, index: int = 0):
        """
        Create and return a PyTorch scheduler instance based on the configuration.

        Args:
            optimizer: PyTorch optimizer instance.
            index: Index for selecting from list-valued parameters.

        Returns:
            Initialized PyTorch scheduler instance or None if no scheduler configured.
        """
        if self.type is None:
            return None

        scheduler_class = getattr(torch.optim.lr_scheduler, self.type)
        kwargs = self.kwargs.copy() if self.kwargs else {}

        # Apply index selection to kwargs if they are lists
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                kwargs[key] = value[index]

        return scheduler_class(optimizer, **kwargs)


class LossConfig(BaseModel):
    """Configuration for loss and optimizer/scheduler settings."""
    model_config = ConfigDict(extra='allow')

    loss_type: Union[List[str], str]
    optimizer: Optional[OptimizerConfig] = None
    scheduler: Optional[SchedulerConfig] = None


class TrainingConfig(BaseModel):
    """Configuration for training loops and learning schedule."""
    model_config = ConfigDict(extra='allow')

    epochs: Union[List[int], int]
    epoch_fin: Union[List[int], int]
    batch_size: Optional[Union[List[int], int]] = None
    gradient_steps_train: Optional[int] = None
    gradient_steps_valid: Optional[int] = None
    clip_grad_norm: float = 1.0
    stop_criterion: Optional[str] = None
    optimizer: Optional[OptimizerConfig] = None
    scheduler: Optional[SchedulerConfig] = None
    checkpoint_interval: Optional[int] = None


class PipeConfig(BaseModel):
    """Configuration for training pipeline."""
    module: str
    function: str


class BaseConfig(BaseModel):
    """Top-level configuration for training."""
    model_config = ConfigDict(extra='allow')

    ML_model_config: ModelConfig = None
    Loss: Optional[LossConfig] = None
    training: Optional[TrainingConfig] = None
    pipe: Optional[PipeConfig] = None
    Prior: Optional[List[ParameterConfig]] = None
    wandb: Optional[Dict[str, Any]] = None
    paths: Optional[Dict[str, Any]] = None

    @model_validator(mode='after')
    def validate_parameters(self):
        """Ensure parameters are provided in the expected location."""
        if not self.Prior:
            raise ValueError("Parameters must be provided in Prior")
        return self

    def get_parameters(self) -> List[ParameterConfig]:
        """Get parameters from the first available location."""
        return self.Prior

    def get_optimizer_config(self) -> Optional[OptimizerConfig]:
        """Get optimizer config from training or Loss section."""
        return self.training.optimizer if self.training and self.training.optimizer else (
            self.Loss.optimizer if self.Loss and self.Loss.optimizer else None
        )

    def get_scheduler_config(self) -> Optional[SchedulerConfig]:
        """Get scheduler config from training or Loss section."""
        return self.training.scheduler if self.training and self.training.scheduler else (
            self.Loss.scheduler if self.Loss and self.Loss.scheduler else None
        )

    def get_loss_type(self, index: int = 0) -> str:
        """Get loss type, handling list indexing."""
        if not self.Loss or not self.Loss.loss_type:
            raise ValueError("Loss configuration not found")
        loss_type = self.Loss.loss_type
        return loss_type[index] if isinstance(loss_type, list) else loss_type

    def get_clip_grad_norm(self) -> float:
        """Get gradient clipping value."""
        return self.training.clip_grad_norm if self.training else 1.0


class MetricsConfig(BaseModel):
    """Configuration for metrics logging."""
    enabled: bool = True
    log_interval: int = 100
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None