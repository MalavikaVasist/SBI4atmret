from typing import Any, Dict, List, Optional, Union
from importlib import import_module

from sbi4atmret.utils import config
from sbi4atmret.utils.load_utils import load_callable
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import torch.optim as optim



class InstrumentPath(BaseModel):
    """Configuration for dataset/observation paths for a specific instrument."""
    path: str

class ConditionPaths(BaseModel):
    """Configuration for dataset paths organized by condition and instrument."""

    __root__: Dict[str, InstrumentPath]  # miri/gemini/hst

class ComponentConfig(BaseModel):
    """Configuration for a generic component with type and kwargs."""
    type: str
    kwargs: Dict[str, Any] = {}


class ObservationConfig(BaseModel):
    """Configuration for observation settings."""
    source: str
    instruments: dict[str, InstrumentPath]
    simulated: Optional[InstrumentPath] = None

class DatasetConfig(BaseModel):
    """Configuration for dataset loading."""
    D: float
    shuffle: bool
    order: List[str]
    dataset_path: dict[str, dict[str, InstrumentPath]]  # condition → instrument → path
    pipe: ComponentConfig
    noise: ComponentConfig

class ParameterConfig(BaseModel):
    """Configuration for a single parameter with bounds."""
    name: str
    lower: float
    upper: float
    default: Optional[float] = None

class PriorConfig(BaseModel):
    """Configuration for prior distribution."""
    distribution: ComponentConfig
    parameters: List[ParameterConfig] 


class EstimatorConfig(BaseModel):
    flow: ComponentConfig
    embedding: ComponentConfig


class TrainingConfig(BaseModel):
    optimizer: ComponentConfig
    scheduler: ComponentConfig | None = None
    loss: ComponentConfig
    epoch_start: int
    batch_size: int
    epoch_final: int
    gradient_steps_train: int
    gradient_steps_valid: int
    clip_grad_norm: float
    stop_criterion: Optional[str] = None
    checkpoint_interval: Optional[int] = None
    name: str
    device: str
    output_dir: str


class WandbConfig(BaseModel):
    project: str
    array: int
    cpus: int
    gpus: int
    ram: str
    time: str
    title: str

class SimulatorConfig(ComponentConfig):
    pass

class BaseConfig(BaseModel):
    """Top-level configuration for training."""
    model_config = ConfigDict(extra='allow')
    
    observation_config: ObservationConfig
    dataset_config: DatasetConfig
    simulator_config: Dict[str, SimulatorConfig]
    prior_config: PriorConfig
    estimator_config: EstimatorConfig
    training_config: TrainingConfig
    wandb_config: WandbConfig
    

    def get_parameter_bounds(self) -> Union[List[float], List[float]]:
        """Extract lower and upper bounds from Prior config."""
        if self.prior_config is None:
            raise KeyError('No Prior section found in config')
        lower = [p.lower for p in self.prior_config.parameters]
        upper = [p.upper for p in self.prior_config.parameters]
        return lower, upper

    def get_parameter_names(self) -> List[str]:
        if self.prior_config is None:
            raise KeyError('No Prior section found in config')
        names = [p.name for p in self.prior_config.parameters]
        return names

    def get_no_of_params(self) -> int:
        """Get the number of parameters from the Prior config."""
        if not self.prior_config:
            raise KeyError('No Prior section found in config')
        return len(self.prior_config.parameters)
    
    # ---------- GENERIC BUILDER ----------

    def _build_component(self, cfg: ComponentConfig, **extra_kwargs):
        cls = load_callable(cfg.type)

        kwargs = {}

        for k, v in (cfg.kwargs or {}).items():

            if isinstance(v, ComponentConfig):
                kwargs[k] = self._build_component(v)

            elif isinstance(v, dict) and "type" in v:
                kwargs[k] = self._build_component(ComponentConfig(**v))

            else:
                kwargs[k] = v

        return cls(**kwargs, **extra_kwargs)

    # ---------- SPECIFIC BUILDERS ----------
    def build_embedding(self):
        return self._build_component(self.estimator_config.embedding)

    def build_flow(self):
        return self._build_component(
            self.estimator_config.flow,
            FlowConfig=self.estimator_config.flow,
            PriorConfig=self,
            EmbeddingConfig=self.estimator_config.embedding,
        )

    def build_loss(self, estimator, prior=None):
        return self._build_component(
            self.training_config.loss,
            estimator=estimator,
            prior=prior
        )
    
    def build_prior(self):
        lower, upper = self.get_parameter_bounds()
        return self._build_component(
            self.prior_config.distribution,
            lower=lower,
            upper=upper
        )

    def build_optimizer(self, parameters):
        return self._build_component(
            self.training_config.optimizer,
            params=parameters
        )

    def build_scheduler(self, optimizer):
        if self.training_config.scheduler is None:
            return None
        return self._build_component(
            self.training_config.scheduler,
            optimizer=optimizer
        )
  
    def build_pipe(self, simulator_dict, obs):
        return self._build_component(self.dataset_config.pipe, 
                                     config=self, 
                                     simulators=simulator_dict,
                                    observation=obs)
    

    def build_noise(self):
        return self._build_component(self.dataset_config.noise, 
                                     config=self)
    
    def build_simulators(self):
        return {
            name: self._build_component(cfg)
            for name, cfg in self.simulator_config.items()
        }
   

    # def select_at_index(self, i: int) -> 'BaseConfig':
    #     """
    #     Create a new config with all list-valued fields reduced to their i-th element.
        
    #     Args:
    #         i: Index to select from list-valued fields.
            
    #     Returns:
    #         A new BaseConfig instance with scalar values.
    #     """
    #     # Helper function to select by index
    #     def _select_value(value, index):
    #         if isinstance(value, (list, tuple)):
    #             return value[index]
    #         return value
        
    #     # Select model config
    #     model_config_dict = {}
    #     if self.model_config:
    #         mc = self.model_config
    #         model_config_dict = {
    #             'embedding': {
    #                 'miri': _select_value(mc.embedding.miri, i),
    #                 'gemini': _select_value(mc.embedding.gemini, i),
    #                 'miri_output': _select_value(mc.embedding.miri_output, i),
    #                 'gemini_output': _select_value(mc.embedding.gemini_output, i),
    #             },
    #             'hidden_features': _select_value(mc.hidden_features, i),
    #             'no_of_params': _select_value(mc.no_of_params, i),
    #             'transforms': _select_value(mc.transforms, i),
    #             'signal': _select_value(mc.signal, i),
    #         }
    #         if mc.batch_size is not None:
    #             model_config_dict['batch_size'] = _select_value(mc.batch_size, i)
        
    #     # Select training config
    #     training_dict = {}
    #     if self.training:
    #         tc = self.training
    #         training_dict = {
    #             'epochs': _select_value(tc.epochs, i),
    #             'epoch_fin': _select_value(tc.epoch_fin, i),
    #             'clip_grad_norm': tc.clip_grad_norm,
    #             'gradient_steps_train': tc.gradient_steps_train,
    #             'gradient_steps_valid': tc.gradient_steps_valid,
    #             'stop_criterion': tc.stop_criterion,
    #             'checkpoint_interval': tc.checkpoint_interval,
    #         }
    #         if tc.batch_size is not None:
    #             training_dict['batch_size'] = _select_value(tc.batch_size, i)
    #         if tc.optimizer:
    #             training_dict['optimizer'] = tc.optimizer
    #         if tc.scheduler:
    #             training_dict['scheduler'] = tc.scheduler
        
    #     # Select loss config
    #     loss_dict = {}
    #     if self.Loss:
    #         lc = self.Loss
    #         loss_dict = {
    #             'loss_type': _select_value(lc.loss_type, i),
    #         }
    #         if lc.optimizer:
    #             loss_dict['optimizer'] = lc.optimizer
    #         if lc.scheduler:
    #             loss_dict['scheduler'] = lc.scheduler
        
    #     # Build new config dict
    #     config_data = {
    #         'model_config': model_config_dict if model_config_dict else None,
    #         'Loss': loss_dict if loss_dict else None,
    #         'training': training_dict if training_dict else None,
    #         'pipe': self.pipe,
    #         'Prior': self.Prior,
    #         'wandb': self.wandb,
    #         'paths': self.paths,
    #     }
        
    #     return BaseConfig(**config_data)

