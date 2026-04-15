from typing import Any, Dict, List, Optional, Union
from importlib import import_module

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class EstimatorConfig(BaseModel):
    """Configuration for estimator model."""
    module: str
    class_name: str = Field(alias="class")
    instrument: str


class LossConfig(BaseModel):
    """Configuration for loss function."""
    model_config = ConfigDict(extra='allow')

    loss_type: Union[List[str], str] = Field(alias="loss_type")
    optimizer: Optional[OptimizerConfig] = None
    scheduler: Optional[SchedulerConfig] = None

    @field_validator('loss_type', mode='before')
    @classmethod
    def validate_loss_alias(cls, v, info):
        if v is None and isinstance(info.data, dict) and 'loss' in info.data:
            return info.data['loss']
        return v


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""
    model_config = ConfigDict(extra='allow')

    optimizer: Optional[OptimizerConfig] = None
    scheduler: Optional[SchedulerConfig] = None
    clip_grad_norm: Optional[float] = 1.0
    epochs: Optional[Union[List[int], int]] = None
    epoch_fin: Optional[Union[List[int], int]] = None
    batch_size: Optional[Union[List[int], int]] = None
    gradient_steps_train: Optional[int] = None
    gradient_steps_valid: Optional[int] = None
    stop_criterion: Optional[str] = None
    checkpoint_interval: Optional[int] = None


class PipeConfig(BaseModel):
    """Configuration for training pipeline."""
    module: str
    function: str


class MLModelConfig(BaseModel):
    """Top-level configuration for ML models."""
    model_config = ConfigDict(extra='allow')

    ML_model_configs: ModelConfig = Field(alias="MLmodel_config")
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