from typing import Any, Dict, List, Optional, Union
from importlib import import_module

from sbi4atmret.utils import config
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import torch.optim as optim


class ParameterConfig(BaseModel):
    """Configuration for a single parameter with bounds."""
    name: str
    lower: float
    upper: float
    default: Optional[float] = None


class PriorConfig(BaseModel):
    """Configuration for prior distribution."""
    Prior: List[ParameterConfig] = Field(alias="Prior")

    def get_parameter_bounds(self) -> (List[float], List[float]):
        """Extract lower and upper bounds from Prior config."""
        if not self.Prior:
            raise KeyError('No Prior section found in config')
        lower = [p.lower for p in self.Prior]
        upper = [p.upper for p in self.Prior]
        return lower, upper

    def get_no_of_params(self) -> int:
        """Get the number of parameters from the Prior config."""
        if not self.Prior:
            raise KeyError('No Prior section found in config')
        return len(self.Prior)


class InstrumentEmbeddingConfig(BaseModel):
    hidden_features: List[int]
    output_dim: List[int]
    input_dim: List[int]


class EmbeddingKwargs(BaseModel):
    bound: float
    instruments: Dict[str, InstrumentEmbeddingConfig]


class EmbeddingConfig(BaseModel):
    type: str
    kwargs: EmbeddingKwargs

class FlowKwargs(BaseModel):
    hidden_features_no: int
    hidden_features: List[int]
    transforms: int
    signal: int

class FlowConfig(BaseModel):
    """Configuration for a flow-based estimator."""
    model_config = ConfigDict(extra='allow')

    type: str
    kwargs: Optional[Dict[str, Any]] = None


class EstimatorConfig(BaseModel):
    """Configuration for estimator settings."""
    model_config = ConfigDict(extra='allow')

    embedding: Optional[EmbeddingConfig]
    flow: Optional[FlowConfig] 


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

    def get_optimizer(self, model_parameters):
        """
        Create and return a PyTorch optimizer instance based on the configuration.

        Args:
            model_parameters: Model parameters to optimize.

        Returns:
            Initialized PyTorch optimizer instance.
        """
        opt_class = getattr(torch.optim, self.type)
        kwargs = self.kwargs.copy() if self.kwargs else {}

        # Select lr from lr or init_lr, handling lists
        lr_value = self.lr if self.lr is not None else self.init_lr
        if lr_value is not None:
            kwargs['lr'] = lr_value

        # Handle weight_decay if provided
        if self.weight_decay is not None:
            kwargs['weight_decay'] = self.weight_decay

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

    def get_scheduler(self, optimizer):
        """
        Create and return a PyTorch scheduler instance based on the configuration.

        Args:
            optimizer: PyTorch optimizer instance.

        Returns:
            Initialized PyTorch scheduler instance or None if no scheduler configured.
        """
        if self.type is None:
            return None

        scheduler_class = getattr(torch.optim.lr_scheduler, self.type)
        kwargs = self.kwargs.copy() if self.kwargs else {}

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

class InstrumentConfig(BaseModel):
    wavelength: list[float]
    path: str

class ObservationConfig(BaseModel):
    source: str
    simulation: bool
    instruments: dict[str, InstrumentConfig]

class DatasetConfig(BaseModel):
    D: float
    dataset_path: dict[str, dict[str, str]]  # condition → instrument → path
    savepath: list[str]


class BaseConfig(BaseModel):
    """Top-level configuration for training."""
    model_config = ConfigDict(extra='allow')
    
    estimator: Optional[EstimatorConfig] = None
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

    def select_at_index(self, i: int) -> 'BaseConfig':
        """
        Create a new config with all list-valued fields reduced to their i-th element.
        
        Args:
            i: Index to select from list-valued fields.
            
        Returns:
            A new BaseConfig instance with scalar values.
        """
        # Helper function to select by index
        def _select_value(value, index):
            if isinstance(value, (list, tuple)):
                return value[index]
            return value
        
        # Select model config
        model_config_dict = {}
        if self.model_config:
            mc = self.model_config
            model_config_dict = {
                'embedding': {
                    'miri': _select_value(mc.embedding.miri, i),
                    'gemini': _select_value(mc.embedding.gemini, i),
                    'miri_output': _select_value(mc.embedding.miri_output, i),
                    'gemini_output': _select_value(mc.embedding.gemini_output, i),
                },
                'hidden_features': _select_value(mc.hidden_features, i),
                'no_of_params': _select_value(mc.no_of_params, i),
                'transforms': _select_value(mc.transforms, i),
                'signal': _select_value(mc.signal, i),
            }
            if mc.batch_size is not None:
                model_config_dict['batch_size'] = _select_value(mc.batch_size, i)
        
        # Select training config
        training_dict = {}
        if self.training:
            tc = self.training
            training_dict = {
                'epochs': _select_value(tc.epochs, i),
                'epoch_fin': _select_value(tc.epoch_fin, i),
                'clip_grad_norm': tc.clip_grad_norm,
                'gradient_steps_train': tc.gradient_steps_train,
                'gradient_steps_valid': tc.gradient_steps_valid,
                'stop_criterion': tc.stop_criterion,
                'checkpoint_interval': tc.checkpoint_interval,
            }
            if tc.batch_size is not None:
                training_dict['batch_size'] = _select_value(tc.batch_size, i)
            if tc.optimizer:
                training_dict['optimizer'] = tc.optimizer
            if tc.scheduler:
                training_dict['scheduler'] = tc.scheduler
        
        # Select loss config
        loss_dict = {}
        if self.Loss:
            lc = self.Loss
            loss_dict = {
                'loss_type': _select_value(lc.loss_type, i),
            }
            if lc.optimizer:
                loss_dict['optimizer'] = lc.optimizer
            if lc.scheduler:
                loss_dict['scheduler'] = lc.scheduler
        
        # Build new config dict
        config_data = {
            'model_config': model_config_dict if model_config_dict else None,
            'Loss': loss_dict if loss_dict else None,
            'training': training_dict if training_dict else None,
            'pipe': self.pipe,
            'Prior': self.Prior,
            'wandb': self.wandb,
            'paths': self.paths,
        }
        
        return BaseConfig(**config_data)


class MetricsConfig(BaseModel):
    """Configuration for metrics logging."""
    enabled: bool = True
    log_interval: int = 100
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None