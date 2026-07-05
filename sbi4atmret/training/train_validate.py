import time

import yaml

from sbi4atmret.runtime.batch_processor import BatchProcessor
from sbi4atmret.utils.checkpoint import load_checkpoint, load_model_state, save_checkpoint
import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from itertools import islice


class Trainer:
    def __init__(self, model, context, config, dataset):
        self.model = model
        self.context = context
        self.config = config
        self.dataset = dataset

        ## domain components
        self.domain = context.runtime.domain

        self.simulator = self.domain.simulators
        self.observation = self.domain.observation
        self.pipe = self.domain.pipe
        self.noise = self.domain.noise
        
        self.checkpoint_path = context.runtime.checkpoint_path
        self.device = context.runtime.device

        # training components
        self.optimizer = context.optimizer
        self.scheduler = context.scheduler
        self.loss_fn = context.loss_fn

        # dataset components
        self.train_keys, self.train_loaders = context.train_lists 
        self.valid_keys, self.valid_loaders = context.valid_lists 

        # additional shortcuts
        self.net = self.model.estimator
        self.net.to(self.device)

        self.batch_processor = BatchProcessor(
                                    pipe=self.pipe,
                                    noise=self.noise,
                                    device=self.device,
                                )

    # ------------------------
    # PUBLIC API
    # ------------------------
    def train(self, resume =False):

        start_epoch = self.config.training_config.epoch_start

        # --- resume ---
        if resume:
            if self.checkpoint_path is None:
                raise ValueError(
                    "resume=True but no checkpoint provided"
                )

            start_epoch = (
                self.loading_checkpoint(self.checkpoint_path) + 1
            )

        # --- wandb ---
        wandb_cfg = self.config.wandb_config
        self.run = wandb.init(
            project=wandb_cfg.project,
            name=wandb_cfg.title, #model_name
            config=self.config.model_dump()
        )

        # --- save path ---
        output_dir = Path(self.config.trainer_config.output_dir) ##runs
        self.run_dir = output_dir / self.run.name  ##runs/model_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        ## save the config in runs/model_name/config.yaml
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(
                self.config.model_dump(mode= "json"),
                f,
                sort_keys=False,
            )
        
        self.checkpoint_dir = self.run_dir / "checkpoints" ##runs/model_name/checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        end_epoch = self.config.training_config.epoch_final

        for epoch in tqdm(range(start_epoch, end_epoch), unit="epoch"):
            
            start = time.time()
            train_loss = self.train_one_epoch()
            end = time.time()
            val_loss = self.validate_one_epoch()

            self.step_scheduler(train_loss, val_loss)

            self.log(epoch, train_loss, val_loss, end-start)

            self.maybe_checkpoint(epoch)        
            
            if self.config.training_config.stop_criterion == 'early_lr': 
                if self.optimizer.param_groups[0]['lr'] <= self.scheduler.min_lrs[0]:
                    break
        
        save_checkpoint(self.checkpoint_dir / "latest.pt", self.net, self.optimizer, self.scheduler, epoch)

        self.run.finish()

        return self.net, self.checkpoint_dir

    # ------------------------
    # CORE METHODS
    # ------------------------
    
    def train_one_epoch(self):
        self.net.train()

        losses = []

        for batches in islice(zip(*self.train_loaders), self.config.training_config.gradient_steps_train):
            batch_dict = self.dataset.reconstruct_batch(self.train_keys, batches)
            theta, x = self.batch_processor.prepare_batch(batch_dict)
            theta, x  = _to_device(theta), _to_device(x)

            self.optimizer.zero_grad()
            loss = self.loss_fn(theta, x)
            loss.backward()

            # gradient clipping
            clip = self.config.training_config.clip_grad_norm
            if clip:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                
            self.optimizer.step()

            losses.append(loss.detach().cpu())

        return torch.stack(losses)

    def validate_one_epoch(self):

        self.net.eval()

        losses = []

        with torch.no_grad():
            for batches in islice(zip(*self.valid_loaders), self.config.training_config.gradient_steps_valid):
                    theta, x = self.batch_processor.prepare_batch(batches, self.valid_keys)
                    theta, x  = _to_device(theta), _to_device(x)

                    loss = self.loss_fn(theta, x)
                    losses.append(loss.detach().cpu())

        return torch.stack(losses)

    # ------------------------
    # UTILITIES
    # ------------------------

    def step_scheduler(self, train_loss, val_loss):
        if self.scheduler is None:
            return 

        try:
            # e.g. ReduceLROnPlateau
            self.scheduler.step(val_loss if val_loss is not None else train_loss)
        except TypeError:
            self.scheduler.step()

    def log(self, epoch, train_loss, val_loss, epoch_time):
        log_dict = {
            "epoch": epoch,
            "train_loss": torch.nanmean(train_loss),
            "valid_loss": torch.nanmean(val_loss),  
            "lr": self.optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time, 
            'nans': torch.isnan(train_loss).float().mean(),  #percentage of NaNs
            'nans_val': torch.isnan(val_loss).float().mean(),
            'trainigset_len' :  len(train_loss),
            'validationset_len' : len(val_loss),
            "config": self.config.model_dump(),

        }

        self.run.log(log_dict)

    def maybe_checkpoint(self, epoch):
        interval = self.config.training_config.checkpoint_interval or 100

        if epoch > 0 and epoch % interval == 0:
            path = self.checkpoint_dir / f"checkpoint_{epoch}.pt"

            save_checkpoint(path, self.net, self.optimizer, self.scheduler, epoch)


    def loading_checkpoint(self, path: str):

        checkpoint = load_checkpoint(path, self.device)
        load_model_state(self.net, checkpoint)

        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint.get("epoch", 0)





