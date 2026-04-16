import torch
from typing import Any, Dict, Optional
from lampe.utils import GDStep
from zuko.distributions import BoxUniform
from lampe.inference import NPELoss


from ..Train.losses import BNPELoss
from .build_estimator import build_estimator as build_model_estimator
from ..utils.load_utils import load_callable
from ..torchutils.optimizer import get_optimizer_from_config
from ..torchutils.scheduler import get_scheduler_from_config
from ..config.configs import BaseConfig
from ..config.selection import select_index_config


class Base:
    """
    Base class for all models providing common setup and utility methods.

    This class contains methods for setting up estimators, optimizers, schedulers,
    loss functions, and other common model operations using Pydantic for configuration validation.
    """

    def __init__(self, config: dict):
        self.config = BaseConfig(**config)
        self.selected_index: Optional[int] = None
        self.selected_config: Optional[Dict[str, Any]] = None
        self.estimator = None
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.prior = None
        self.pipe = None

    def _get_active_config(self, i: int = 0) -> Dict[str, Any]:
        if self.selected_config is not None:
            return self.selected_config
        return self.select_index_config(i)

    def select_index_config(self, i: int) -> dict:
        base_config = self.config.model_dump()
        selected = select_index_config(base_config, i)
        self.selected_index = i
        self.selected_config = selected
        return selected

    def get_selected_config(self) -> dict:
        if self.selected_config is None:
            raise ValueError('No model index selected. Call select_index_config(i) first.')
        return self.selected_config

    def _get_parameter_bounds(self, config: Dict[str, Any]):
        parameter_list = config.get('Prior') 
        if parameter_list is None:
            raise KeyError('No Prior section found in config')
        lower = [p['lower'] if isinstance(p, dict) else p[1] for p in parameter_list]
        upper = [p['upper'] if isinstance(p, dict) else p[2] for p in parameter_list]
        return lower, upper

    def _build_flow_estimator(self, flow_config: Dict[str, Any]):
        """
        Build a flow-based estimator from the provided flow configuration.

        This method is intentionally minimal and delegates the concrete model
        type selection to `build_model.py`. If your config provides a custom
        module/class pair, Base will use dynamic loading instead.
        """
        config = self.selected_config if self.selected_config is not None else self.config.model_dump()
        return build_model_estimator(config)

    def _build_estimator(self, model_configs: Dict[str, Any]):
        estimator_cfg = model_configs.get('estimator')
        if estimator_cfg is None:
            # Fall back to model builder helper for legacy or alternate schema.
            return build_model_estimator(self.selected_config if self.selected_config is not None else self.config.model_dump())

        if isinstance(estimator_cfg, dict) and 'module' in estimator_cfg and 'class' in estimator_cfg:
            return load_callable(estimator_cfg['module'], estimator_cfg['class'])(**estimator_cfg.get('kwargs', {}))

        if isinstance(estimator_cfg, dict) and 'flow' in estimator_cfg:
            return self._build_flow_estimator(estimator_cfg['flow'])

        return build_model_estimator(self.selected_config if self.selected_config is not None else self.config.model_dump())

    def setup_estimator(self, i: int = 0):
        config = self._get_active_config(i)
        model_configs = config.get('ML_model_config')
        self.estimator = self._build_estimator(model_configs)
        if hasattr(self.estimator, 'cuda'):
            self.estimator = self.estimator.cuda()
        return self.estimator

    def setup_loss_and_prior(self, i: int = 0):
        config = self._get_active_config(i)
        loss_config = config.get('Loss')
        if loss_config is None:
            raise KeyError('Loss configuration not found')
        loss_name = loss_config['loss_type']
        lower, upper = self._get_parameter_bounds(config)

        self.prior = BoxUniform(torch.tensor(lower).cuda(), torch.tensor(upper).cuda())
        if loss_name == 'NPELoss':
            self.loss = NPELoss(self.estimator)
        elif loss_name == 'BNPELoss':
            self.loss = BNPELoss(self.estimator, self.prior)
        else:
            raise NotImplementedError(f"Unsupported loss function: {loss_name}")
        return self.loss, self.prior

    def network_to_device(self, device: str = 'cuda'):
        if self.estimator is not None:
            if device == 'cuda':
                self.estimator.cuda()
            else:
                self.estimator.cpu()

    def initialize_op_scheduler(self, i: int = 0):
        if self.estimator is None:
            raise ValueError("Estimator must be set up before initializing optimizer and scheduler")

        config = self._get_active_config(i)
        optimizer_cfg = config.get('Loss', {}).get('optimizer') or config.get('training', {}).get('optimizer')
        if optimizer_cfg is None:
            raise KeyError("Optimizer configuration not found")

        self.optimizer = get_optimizer_from_config(self.estimator.parameters(), optimizer_cfg, 0)
        clip = config.get('training', {}).get('clip_grad_norm', 1.0)
        step = GDStep(self.optimizer, clip=clip)
        scheduler_cfg = config.get('Loss', {}).get('scheduler') or config.get('training', {}).get('scheduler')
        self.scheduler = get_scheduler_from_config(self.optimizer, scheduler_cfg, 0)
        return self.optimizer, step, self.scheduler

    def log_metrics(self, metrics: dict, step: int = None):
        print(f"Step {step}: {metrics}")

    def save_model(self, path: str):
        if self.estimator is not None:
            torch.save({
                'estimator_state_dict': self.estimator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            }, path)

    def load_estimator(self, path: str):
        checkpoint = torch.load(path)
        if self.estimator is not None:
            self.estimator.load_state_dict(checkpoint['estimator_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def setup_pipe(self):
        if self.loss is None:
            raise ValueError("Loss must be set up before setting up pipe")

        config = self.selected_config if self.selected_config is not None else self.config.model_dump()
        pipe_config = config.get('pipe') if isinstance(config, dict) else self.config.pipe
        if pipe_config is None:
            raise KeyError('Pipe configuration not found')

        if isinstance(pipe_config, dict):
            module = pipe_config['module']
            function = pipe_config['function']
        else:
            module = pipe_config.module
            function = pipe_config.function

        pipe_func = load_callable(module, function)
        self.pipe = pipe_func(self.loss)
        return self.pipe


# Standalone functions for backward compatibility
def setup_estimator(config, i: int = 0):
    base = Base(config)
    return base.setup_estimator(i)


def setup_optimizer_and_scheduler(estimator, config, i: int = 0):
    base = Base(config)
    base.estimator = estimator
    return base.initialize_op_scheduler(i)


def setup_loss_and_prior(estimator, config, i: int = 0):
    base = Base(config)
    base.estimator = estimator
    return base.setup_loss_and_prior(i)


def setup_pipe(config, loss):
    base = Base(config)
    base.loss = loss
    return base.setup_pipe()