import torch
from lampe.utils import GDStep
from zuko.distributions import BoxUniform

from ..Train.Loss import BNPELoss
from ..utils.load_utils import load_callable
from ..torchutils.optimizer import get_optimizer_from_config
from ..torchutils.scheduler import get_scheduler_from_config
from lampe.inference import NPELoss
from ..config import MLModelConfig


def _select_by_index(value, index):
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


class Base:
    """
    Base class for all models providing common setup and utility methods.

    This class contains methods for setting up estimators, optimizers, schedulers,
    loss functions, and other common model operations using Pydantic for configuration validation.
    """

    def __init__(self, config: dict):
        """
        Initialize the base model with Pydantic configuration validation.

        Args:
            config: Configuration dictionary that will be validated by Pydantic
        """
        self.config = MLModelConfig(**config)
        self.estimator = None
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.prior = None
        self.pipe = None

    def setup_estimator(self, i: int = 0):
        """
        Set up the estimator model from configuration.

        Args:
            i: Index for multi-model setups

        Returns:
            Initialized estimator model
        """
        estimator_cfg = self.config.estimator
        model_configs = self.config.ML_model_configs
        parameters = self.config.get_parameters()

        lower = [p.lower for p in parameters]
        upper = [p.upper for p in parameters]

        estimator_class = load_callable(estimator_cfg.module, estimator_cfg.class_name)

        self.estimator = estimator_class(
            hf_miri=_select_by_index(model_configs.embedding.miri, i),
            hf_inst=_select_by_index(model_configs.embedding.gemini, i),
            instrument=estimator_cfg.instrument,
            hidden_features=_select_by_index(model_configs.hidden_features, i),
            emb_miri_output=_select_by_index(model_configs.embedding.miri_output, i),
            emb_inst_output=_select_by_index(model_configs.embedding.gemini_output, i),
            no_of_params=_select_by_index(model_configs.no_of_params, i),
            transforms=_select_by_index(model_configs.transforms, i),
            signal=_select_by_index(model_configs.signal, i),
            LOWER=lower,
            UPPER=upper,
        )
        return self.estimator

    def setup_loss_and_prior(self, i: int = 0):
        """
        Set up loss function and prior distribution.

        Args:
            i: Index for multi-model setups

        Returns:
            Tuple of (loss, prior)
        """
        loss_name = self.config.get_loss_type(i)
        parameters = self.config.get_parameters()

        lower = [p.lower for p in parameters]
        upper = [p.upper for p in parameters]

        self.prior = BoxUniform(torch.tensor(lower).cuda(), torch.tensor(upper).cuda())
        if loss_name == 'NPELoss':
            self.loss = NPELoss(self.estimator)
        elif loss_name == 'BNPELoss':
            self.loss = BNPELoss(self.estimator, self.prior)
        else:
            raise NotImplementedError(f"Unsupported loss function: {loss_name}")
        return self.loss, self.prior

    def network_to_device(self, device: str = 'cuda'):
        """
        Move the network to the specified device.

        Args:
            device: Device to move to ('cuda' or 'cpu')
        """
        if self.estimator is not None:
            if device == 'cuda':
                self.estimator.cuda()
            else:
                self.estimator.cpu()

    def initialize_op_scheduler(self, i: int = 0):
        """
        Initialize optimizer and scheduler.

        Args:
            i: Index for multi-model setups

        Returns:
            Tuple of (optimizer, step, scheduler)
        """
        if self.estimator is None:
            raise ValueError("Estimator must be set up before initializing optimizer and scheduler")

        optimizer_cfg = self.config.get_optimizer_config()
        if optimizer_cfg is None:
            raise KeyError("Optimizer configuration not found")

        self.optimizer = get_optimizer_from_config(self.estimator.parameters(), optimizer_cfg.model_dump(), i)
        clip = self.config.get_clip_grad_norm()
        step = GDStep(self.optimizer, clip=clip)
        scheduler_cfg = self.config.get_scheduler_config()
        self.scheduler = get_scheduler_from_config(self.optimizer, scheduler_cfg.model_dump() if scheduler_cfg else None, i)
        return self.optimizer, step, self.scheduler

    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log training metrics. Override in subclasses for specific logging implementations.

        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
        """
        # Default implementation - subclasses should override
        print(f"Step {step}: {metrics}")

    def save_model(self, path: str):
        """
        Save the model state.

        Args:
            path: Path to save the model
        """
        if self.estimator is not None:
            torch.save({
                'estimator_state_dict': self.estimator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            }, path)

    def load_model(self, path: str):
        """
        Load the model state.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        if self.estimator is not None:
            self.estimator.load_state_dict(checkpoint['estimator_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def setup_pipe(self):
        """
        Set up the training pipeline.

        Returns:
            Initialized pipeline function
        """
        if self.loss is None:
            raise ValueError("Loss must be set up before setting up pipe")

        if not self.config.pipe:
            raise ValueError("Pipe configuration not found")

        pipe_func = load_callable(self.config.pipe.module, self.config.pipe.function)
        self.pipe = pipe_func(self.loss)
        return self.pipe