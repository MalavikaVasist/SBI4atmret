import torch
from typing import Union, Optional
from lampe.utils import GDStep
from zuko.distributions import BoxUniform
from lampe.inference import NPELoss


from ..Train.losses import BNPELoss
from .build import build_estimator as build_model_estimator
from ..utils.load_utils import load_callable
from ..torchutils.optimizer import get_optimizer_from_config
from ..torchutils.scheduler import get_scheduler_from_config
from ..config.configs import BaseConfig
from ..config.selection import select_config_index
from estimator.base import EstimatorBase


class Base:
    """
    Base class for all models providing common setup and utility methods.
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

    def setup_estimator(self):
        estimator_config = self.config.estimator
        embedding_config = estimator_config.embedding
        flow_config = estimator_config.flow
        
        # --- Build embedding ---
        embedding_type = estimator_config.embedding.name
        embedding_cls = load_callable("sbi4atmret.estimator.embedding", embedding_type)
        embedding = embedding_cls(embedding_config)


        # --- Build flow ---
        flow_type = estimator_config.flow.name
        flow_cls = load_callable("sbi4atmret.estimator.flows", flow_type)
        flow = flow_cls(flow_config)

        # --- Combine ---
        self.estimator = EstimatorBase(flow, embedding)

        return self

    
    def network_to_device(self, device: str = 'cuda'):
        """Move estimator to device."""
        if self.estimator is not None:
            if device == 'cuda':
                self.estimator.cuda()
            else:
                self.estimator.cpu()

    def setup_optimizer_and_scheduler(self):
        """Set up optimizer and scheduler from config."""
        if self.estimator is None:
            raise ValueError("Estimator must be set up before initializing optimizer and scheduler")

        optimizer_cfg = self.config.training.optimizer
        scheduler_cfg = self.config.training.scheduler

        self.optimizer = get_optimizer_from_config(optimizer_cfg, self.estimator.parameters())
        self.scheduler = get_scheduler_from_config(scheduler_cfg, self.optimizer) if scheduler_cfg else None
        
        return self.optimizer, self.scheduler

    def setup_loss(self):       
        """Set up loss function and prior from config."""
        loss_cfg = self.config.training.loss
        if loss_cfg is None:
            raise KeyError('Loss configuration not found')
        
        loss_type = loss_cfg.loss_type
        lower, upper = self._get_parameter_bounds(self.config)

        self.prior = BoxUniform(torch.tensor(lower).cuda(), torch.tensor(upper).cuda())
        if loss_type == 'NPELoss':
            self.loss = NPELoss(self.estimator)
        elif loss_type == 'BNPELoss':
            self.loss = BNPELoss(self.estimator, self.prior)
        else:
            raise NotImplementedError(f"Unsupported loss function: {loss_type}")
        return self.loss

    
    def load_from_checkpoint(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path)
        if self.estimator is not None:
            self.estimator.load_state_dict(checkpoint['estimator_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    def _step_scheduler(scheduler, metric=None):
        if scheduler is None:
            return
        if isinstance(scheduler, sched.ReduceLROnPlateau):
            scheduler.step(metric)
        else:
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(metric)


    def train(config: Union[dict, BaseConfig], datasets, simulator, observation, checkpoint_fn=None, checkpoint_interval=50):
        """
        - run the setup 
        - execute a training loop until limit.


        Args:
            config: Configuration dict or BaseConfig instance
            datasets: Training/validation/test datasets
            simulator: Simulator for forward modeling
            observation: Observation data
            checkpoint_fn: Optional checkpoint saving function
            checkpoint_interval: Interval for checkpointing
            
        Returns:
            Tuple of (estimator, runpath)
        """
        # Convert to BaseConfig if dict
        if isinstance(config, dict):
            config = BaseConfig(**config)
        
        # Use attribute access on Pydantic model
        model_name = str(config.estimator.embedding.output_dim) if config.estimator else "default"
        
        run = wandb.init(
            project=config.wandb['project'] if config.wandb else 'default',
            config={},
            name=f"{model_name}"
        )

        estimator = setup_estimator(config, 0)
        optimizer, step, scheduler = setup_optimizer_and_scheduler(estimator, config, 0)
        loss, prior = setup_loss_and_prior(estimator, config, 0)
        pipe = setup_pipe(config, loss)

        savepath = Path(config.paths['savepath'][0]) if config.paths else Path('./runs')
        runpath = savepath / run.name
        runpath.mkdir(parents=True, exist_ok=True)

        # Use attribute access for training config
        start_epoch = config.training.epoch_fin if config.training else 0
        end_epoch = config.training.epochs if config.training else 100
        
        for epoch in tqdm(range(start_epoch, end_epoch), unit='epoch'):
            # Build dataset lists (compatible with existing data loading)
            trainsets = [datasets[atm_type][instrument]['train'] 
                        for atm_type in config.wandb['simulator']['type'] if config.wandb
                        for instrument in config.wandb['instruments'] if config.wandb]
            losses_train, duration = train_epoch(estimator, step, trainsets, simulator, pipe, config.model_dump())

            validsets = [datasets[atm_type][instrument]['valid']
                        for atm_type in config.wandb['simulator']['type'] if config.wandb
                        for instrument in config.wandb['instruments'] if config.wandb]
            losses_val = validate_epoch(estimator, validsets, simulator, pipe, step, config.model_dump())

            log_to_wandb(run, optimizer.param_groups[0]['lr'], 
                        torch.nanmean(losses_train), torch.nanmean(losses_val),
                        torch.isnan(losses_train).mean(), torch.isnan(losses_val).mean(),
                        len(losses_train) / duration, len(losses_train), len(losses_val))

            _step_scheduler(scheduler, torch.nanmean(losses_val))

            if checkpoint_fn is not None and epoch > 100 and epoch % checkpoint_interval == 0:
                checkpoint_fn(runpath, estimator, optimizer, epoch)

            # Use attribute access for stop criterion
            if config.training and config.training.stop_criterion == 'early' and scheduler is not None and optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
                break

        # Post-training plotting
        testsets = [datasets[atm_type][instrument]['test']
                for atm_type in config.wandb['simulator']['type'] if config.wandb
                for instrument in config.wandb['instruments'] if config.wandb]
        plot_dict = plot_results(runpath, estimator, observation, testsets, pipe, simulator, config.model_dump())

        for key, fig in plot_dict.items():
            run.log({key: wandb.Image(fig)})

        run.finish()
        return estimator, runpath


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

