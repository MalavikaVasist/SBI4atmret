import torch
from typing import Union, Optional
from lampe.utils import GDStep
from zuko.distributions import BoxUniform
from lampe.inference import NPELoss


from ..Train.losses import BNPELoss
from .build_estimator import build_estimator as build_model_estimator
from ..utils.load_utils import load_callable
from ..torchutils.optimizer import get_optimizer_from_config
from ..torchutils.scheduler import get_scheduler_from_config
from ..config.configs import BaseConfig
from ..config.selection import select_config_index


class Base:
    """
    Base class for all models providing common setup and utility methods.

    This class uses Pydantic models for type-safe configuration management
    instead of plain dicts, providing automatic validation and attribute access.
    """

    def __init__(self, config: Union[dict, BaseConfig]):
        """Initialize with config dict or BaseConfig instance."""
        self.config = config if isinstance(config, BaseConfig) else BaseConfig(**config)
        self.selected_index: Optional[int] = None
        self.selected_config: Optional[BaseConfig] = None
        self.estimator = None
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.prior = None
        self.pipe = None

    def _get_active_config(self, i: int = 0) -> BaseConfig:
        """Get the active config, either selected or by index."""
        if self.selected_config is not None:
            return self.selected_config
        return self.select_index_config(i)

    def select_index_config(self, i: int) -> BaseConfig:
        """Select config at index i and cache it."""
        selected = select_config_index(self.config, i)
        self.selected_index = i
        self.selected_config = selected
        return selected

    def get_selected_config(self) -> BaseConfig:
        """Get the currently selected config."""
        if self.selected_config is None:
            raise ValueError('No model index selected. Call select_index_config(i) first.')
        return self.selected_config

    def _get_parameter_bounds(self, config: BaseConfig):
        """Extract lower and upper bounds from Prior config."""
        if not config.Prior:
            raise KeyError('No Prior section found in config')
        lower = [p.lower for p in config.Prior]
        upper = [p.upper for p in config.Prior]
        return lower, upper

    def _build_flow_estimator(self):
        """Build a flow-based estimator from the selected config."""
        # For backward compatibility, pass dict to build_model_estimator
        config_dict = self.selected_config.model_dump() if self.selected_config else self.config.model_dump()
        return build_model_estimator(config_dict)

    def _build_estimator(self, model_config):
        """Build estimator from ModelConfig."""
        if model_config is None:
            return build_model_estimator(self.selected_config.model_dump() if self.selected_config else self.config.model_dump())

        # For backward compatibility with build_model_estimator which expects dict
        config_dict = self.selected_config.model_dump() if self.selected_config else self.config.model_dump()
        return build_model_estimator(config_dict)

    def setup_estimator(self, i: int = 0):
        """Set up the estimator for config index i."""
        config = self._get_active_config(i)
        model_config = config.ML_model_config
        self.estimator = self._build_estimator(model_config)
        if hasattr(self.estimator, 'cuda'):
            self.estimator = self.estimator.cuda()
        return self.estimator

    def setup_loss_and_prior(self, i: int = 0):
        """Set up loss function and prior for config index i."""
        config = self._get_active_config(i)
        
        if config.Loss is None:
            raise KeyError('Loss configuration not found')
        
        loss_type = config.get_loss_type(0)
        lower, upper = self._get_parameter_bounds(config)

        self.prior = BoxUniform(torch.tensor(lower).cuda(), torch.tensor(upper).cuda())
        if loss_type == 'NPELoss':
            self.loss = NPELoss(self.estimator)
        elif loss_type == 'BNPELoss':
            self.loss = BNPELoss(self.estimator, self.prior)
        else:
            raise NotImplementedError(f"Unsupported loss function: {loss_type}")
        return self.loss, self.prior

    def network_to_device(self, device: str = 'cuda'):
        """Move estimator to device."""
        if self.estimator is not None:
            if device == 'cuda':
                self.estimator.cuda()
            else:
                self.estimator.cpu()

    def initialize_op_scheduler(self, i: int = 0):
        """Set up optimizer and scheduler for config index i."""
        if self.estimator is None:
            raise ValueError("Estimator must be set up before initializing optimizer and scheduler")

        config = self._get_active_config(i)
        
        # Get optimizer config from Loss or training section
        optimizer_cfg = config.get_optimizer_config()
        if optimizer_cfg is None:
            raise KeyError("Optimizer configuration not found")

        # Use the optimizer method from OptimizerConfig
        self.optimizer = optimizer_cfg.get_optimizer(self.estimator.parameters())
        
        clip = config.get_clip_grad_norm()
        step = GDStep(self.optimizer, clip=clip)
        
        # Get scheduler config from Loss or training section
        scheduler_cfg = config.get_scheduler_config()
        self.scheduler = scheduler_cfg.get_scheduler(self.optimizer) if scheduler_cfg else None
        
        return self.optimizer, step, self.scheduler

    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics."""
        print(f"Step {step}: {metrics}")

    def save_model(self, path: str):
        """Save model state."""
        if self.estimator is not None:
            torch.save({
                'estimator_state_dict': self.estimator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            }, path)

    def load_estimator(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path)
        if self.estimator is not None:
            self.estimator.load_state_dict(checkpoint['estimator_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def setup_pipe(self):
        """Set up training pipeline."""
        if self.loss is None:
            raise ValueError("Loss must be set up before setting up pipe")

        config = self.selected_config if self.selected_config is not None else self.config
        pipe_config = config.pipe
        
        if pipe_config is None:
            raise KeyError('Pipe configuration not found')

        # Use attribute access from PipeConfig
        pipe_func = load_callable(pipe_config.module, pipe_config.function)
        self.pipe = pipe_func(self.loss)
        return self.pipe


# Standalone functions for backward compatibility
def setup_estimator(config: Union[dict, BaseConfig], i: int = 0):
    """Set up estimator (backward compatible)."""
    base = Base(config)
    return base.setup_estimator(i)


def setup_optimizer_and_scheduler(estimator, config: Union[dict, BaseConfig], i: int = 0):
    """Set up optimizer and scheduler (backward compatible)."""
    base = Base(config)
    base.estimator = estimator
    return base.initialize_op_scheduler(i)


def setup_loss_and_prior(estimator, config: Union[dict, BaseConfig], i: int = 0):
    """Set up loss and prior (backward compatible)."""
    base = Base(config)
    base.estimator = estimator
    return base.setup_loss_and_prior(i)


def setup_pipe(config: Union[dict, BaseConfig], loss):
    """Set up pipe (backward compatible)."""
    base = Base(config)
    base.loss = loss
    return base.setup_pipe()