from typing import Any, Dict, List, Optional, Union
from importlib import import_module

from sbi4atmret.utils.load_utils import load_callable
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import torch.optim as optim



class InstrumentPath(BaseModel):
    """Configuration for dataset/observation paths for a specific instrument."""
    path: str

class ConditionPaths(BaseModel):
    """Configuration for dataset paths organized by condition and instrument."""
    paths: Dict[str, InstrumentPath]

class ComponentConfig(BaseModel):
    """Configuration for a generic component with type and kwargs."""
    type: str
    kwargs: Dict[str, Any] = {}


class ObservationConfig(BaseModel):
    """Configuration for observation settings."""
    source: str
    instruments: Dict[str, InstrumentPath]
    simulated: Optional[InstrumentPath] = None

class DatasetConfig(BaseModel):
    """Configuration for dataset loading."""
    D: float
    shuffle: bool
    order: Optional[List[str]] = None
    dataset_path: Dict[str, InstrumentPath]  # instrument → path
    pipe: ComponentConfig
    noise: ComponentConfig
    theta_mapper: Optional[ComponentConfig] = None  # Optional theta mapper for parameter space transformations

    @model_validator(mode="after")
    def _derive_order(self):
        if self.order is None:
            self.order = sorted(self.dataset_path.keys())
        return self

class ParameterConfig(BaseModel):
    """Configuration for a single parameter with bounds."""
    name: str                          # code name (matches simulator.names)
    label: Optional[str] = None        # display name for plots (LaTeX)
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
    scheduler: Optional[ComponentConfig] = None
    loss: ComponentConfig
    epoch_start: int
    batch_size: int
    epoch_final: int
    gradient_steps_train: int
    gradient_steps_valid: int
    clip_grad_norm: float
    stop_criterion: Optional[str] = None
    checkpoint_interval: Optional[int] = None
    name: Any
    device: str
    output_dir: str


class WandbConfig(BaseModel):
    project: str
    title: str

class SimulatorConfig(ComponentConfig):
    pass

class BaseConfig(BaseModel):
    """Top-level configuration for training."""
    model_config = ConfigDict(extra='allow', populate_by_name=True)
    
    observation_config: ObservationConfig
    dataset_config: DatasetConfig
    simulator_config: Dict[str, SimulatorConfig]
    prior_config: PriorConfig
    estimator_config: EstimatorConfig
    training_config: TrainingConfig
    wandb_config: WandbConfig = Field(alias="wandb")
    

    def get_parameter_bounds(self) -> Union[List[float], List[float]]:
        """Extract lower and upper bounds from Prior config."""
        if self.prior_config is None:
            raise KeyError('No Prior section found in config')
        lower = [p.lower for p in self.prior_config.parameters]
        upper = [p.upper for p in self.prior_config.parameters]
        return lower, upper

    def get_parameter_names(self) -> List[str]:
        """Get code names (matching simulator.names)."""
        if self.prior_config is None:
            raise KeyError('No Prior section found in config')
        return [p.name for p in self.prior_config.parameters]

    def get_parameter_labels(self) -> List[str]:
        """Get display labels for plotting (LaTeX). Falls back to name if no label."""
        if self.prior_config is None:
            raise KeyError('No Prior section found in config')
        return [p.label if p.label else p.name for p in self.prior_config.parameters]

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

            if isinstance(v, dict) and "type" in v:
                resolved = load_callable(ComponentConfig(**v).type)
                nested_kwargs = v.get("kwargs", {})
                if not nested_kwargs and callable(resolved) and not isinstance(resolved, type):
                    kwargs[k] = resolved
                else:
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
    
 
    def build_pipe(self, domain):
        return self._build_component(self.dataset_config.pipe, 
                                     domain = domain)
    

    def build_noise(self, domain):
        return self._build_component(self.dataset_config.noise, 
                                     domain = domain)
    
    def build_simulators(self):
        return {
            name: self._build_component(cfg)
            for name, cfg in self.simulator_config.items()
        }
    
    

   