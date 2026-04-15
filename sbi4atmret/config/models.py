from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import torch
from importlib import import_module


class ParameterConfig(BaseModel):
    """Configuration for a single parameter with bounds."""
    name: str
    lower: float
    upper: float
    default: Optional[float] = None


class PriorConfig(BaseModel):
    """Configuration for prior distribution."""
    parameters: List[ParameterConfig] = Field(alias="PARAMETERS")


class OptimizerConfig(BaseModel):
    """Pydantic model for optimizer configuration with validation."""
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


class SchedulerConfig(BaseModel):
    """Pydantic model for scheduler configuration with validation."""
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


class EmbeddingConfig(BaseModel):
    """Configuration for embedding layers."""
    miri: Union[List[int], int]
    gemini: Union[List[int], int]
    miri_output: Union[List[int], int]
    gemini_output: Union[List[int], int]


class ModelConfig(BaseModel):
    """Configuration for model architecture."""
    embedding: EmbeddingConfig
    hidden_features: Union[List[int], int]
    no_of_params: Union[List[int], int]
    transforms: Union[List[int], int]
    signal: Union[List[int], int]


class EstimatorConfig(BaseModel):
    """Configuration for estimator model."""
    module: str
    class_name: str = Field(alias="class")
    instrument: str


class LossConfig(BaseModel):
    """Configuration for loss function."""
    loss_type: str = Field(alias="loss")
    optimizer: Optional[OptimizerConfig] = None
    scheduler: Optional[SchedulerConfig] = None


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""
    optimizer: Optional[OptimizerConfig] = None
    scheduler: Optional[SchedulerConfig] = None
    clip_grad_norm: Optional[float] = 1.0


class PipeConfig(BaseModel):
    """Configuration for training pipeline."""
    module: str
    function: str


class MLModelConfig(BaseModel):
    """Top-level configuration for ML models."""
    estimator: EstimatorConfig
    ML_model_configs: ModelConfig
    Loss: Optional[LossConfig] = None
    training: Optional[TrainingConfig] = None
    pipe: Optional[PipeConfig] = None
    PARAMETERS: Optional[List[ParameterConfig]] = None
    Prior: Optional[List[ParameterConfig]] = None
    prior: Optional[List[ParameterConfig]] = None

    @model_validator(mode='after')
    def validate_parameters(self):
        """Ensure parameters are provided in one of the expected locations."""
        if not any([self.PARAMETERS, self.Prior, self.prior]):
            raise ValueError("Parameters must be provided in one of: PARAMETERS, Prior, or prior")
        return self

    def get_parameters(self) -> List[ParameterConfig]:
        """Get parameters from the first available location."""
        return self.PARAMETERS or self.Prior or self.prior

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