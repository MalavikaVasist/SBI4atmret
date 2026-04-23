import torch
from typing import Union

from ..config.configs import BaseConfig
from estimator.base import EstimatorBase
from datasets.base import Dataset
from torchutils.general import _resolve_device
from pathlib import Path
import wandb
from tqdm import tqdm


class Base:
    """
    Base class for all models providing common setup and utility methods.
    """

    def __init__(self, config: Union[dict, BaseConfig]):
        """Initialize with config dict or BaseConfig instance."""
        self.config = config if isinstance(config, BaseConfig) else BaseConfig(**config)
        # self.selected_index: Optional[int] = None
        # self.selected_config: Optional[BaseConfig] = None

        self.device = _resolve_device()
        self.estimator = None
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.prior = None
        self.pipe = None

    
    def build(self):
        # --- Build components ---
        embedding = self.config.build_embedding()
        flow = self.config.build_flow()

        # --- Compose estimator ---
        self.estimator = EstimatorBase(flow, embedding)
        self.estimator = self.to_device(self.estimator)

        # --- Compose prior ---
        self.prior = self.config.build_prior()
        self.prior = self.to_device(self.prior)

        # --- Training components ---
        self.loss = self.config.build_loss(self.estimator, self.prior)

        self.optimizer = self.config.build_optimizer(self.estimator.parameters())
        self.scheduler = self.config.build_scheduler(self.optimizer)

        self.pipe = self.config.build_pipe()

        return self

    
    def to_device(self, module):
        if module is None:
            return None

        return module.to(self.device)

    
    def load_from_checkpoint(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path)
        if self.estimator is not None:
            self.estimator.load_state_dict(checkpoint['estimator_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    def step_scheduler(scheduler, metric=None):
        if scheduler is None:
            return
        if isinstance(scheduler, sched.ReduceLROnPlateau):
            scheduler.step(metric)
        else:
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(metric)


    def train(
        self,
        datasets,
        simulator,
        observation,
        checkpoint_fn=None,
    ):
        """
        Train the model using already-built components.
        """

        # --- Build if not already done ---
        if self.estimator is None:
            self.build()


        # --- WandB ---
        wandb_cfg = self.config.wandb_config
        run = wandb.init(
            project=wandb_cfg["project"],
            name=wandb_cfg.get("title", "run"),
            config=self.config.model_dump()
        )

        # --- Paths ---
        self.savepath = Path(self.config.dataset_config.savepath)
        runpath = self.savepath / run.name
        runpath.mkdir(parents=True, exist_ok=True)

        # --- Epochs ---
        start_epoch = self.config.training_config.epoch_fin
        end_epoch = self.config.training_config.epochs

        dataset = Dataset(self.config.dataset_config) 
        dataloaders = dataset.return_dataloaders()


        # --- Loop ---
        for epoch in tqdm(range(start_epoch, end_epoch), unit="epoch"):

            for batches in zip(dataloaders):
                theta, x = pipe(*batches)

                losse = model(theta, x)


            trainsets = [
                datasets[cond][inst]["train"]
                for cond in self.config.dataset_config.dataset_path.keys()
                for inst in self.config.dataset_config.dataset_path[cond].keys()
            ]

            validsets = [
                datasets[cond][inst]["valid"]
                for cond in config.dataset_config.dataset_path.keys()
                for inst in config.dataset_config.dataset_path[cond].keys()
            ]

            losses_train, duration = train_epoch(
                estimator,
                optimizer,
                trainsets,
                simulator,
                loss_fn,
                config.model_dump()
            )

            losses_val = validate_epoch(
                estimator,
                validsets,
                simulator,
                loss_fn,
                optimizer,
                config.model_dump()
            )

            # --- Logging ---
            wandb.log({
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": torch.nanmean(losses_train),
                "val_loss": torch.nanmean(losses_val),
            })

            # --- Scheduler ---
            if scheduler is not None:
                try:
                    scheduler.step(torch.nanmean(losses_val))
                except TypeError:
                    scheduler.step()

            # --- Checkpoint ---
            interval = config.training_config.checkpoint_interval or 100
            if checkpoint_fn and epoch > 100 and epoch % interval == 0:
                checkpoint_fn(runpath, estimator, optimizer, epoch)

            # --- Early stopping ---
            if (
                config.training_config.stop_criterion == "early"
                and scheduler is not None
                and optimizer.param_groups[0]["lr"] <= getattr(scheduler, "min_lrs", [0])[0]
            ):
                break

        # --- Test ---
        testsets = [
            datasets[cond][inst]["test"]
            for cond in self.config.dataset_config.dataset_path.keys()
            for inst in self.config.dataset_config.dataset_path[cond].keys()
        ]

        plot_dict = self.plot_results(
            runpath,
            self.estimator,
            observation,
            testsets,
            self.loss,
            simulator,
            self.config.model_dump()
        )

        for key, fig in plot_dict.items():
            run.log({key: wandb.Image(fig)})

        run.finish()

        return self.estimator, runpath


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

