import time

import torch
import wandb
from tqdm import tqdm
from pathlib import Path
import islice


class Trainer:
    def __init__(self, model, context, config):
        self.model = model
        self.ctx = context
        self.config = config

        # shortcuts
        self.net = self.model.estimator
        self.optimizer = context.optimizer
        self.scheduler = context.scheduler
        self.loss_fn = context.loss_fn

        self.train_keys, self.train_loaders = context.train_lists 
        self.valid_keys, self.valid_loaders = context.valid_lists 
        self.pipe = context.pipe
        self.noise = context.noise

        self.device = context.device

        # move model + prior
        self.net.to(self.device)
        if self.ctx.prior is not None:
            self.ctx.prior = self.ctx.prior.to(self.device)


    # ------------------------
    # PUBLIC API
    # ------------------------
    def train(self, resume =False):

        start_epoch = self.config.training_config.epoch_start
        checkpoint_path = self.ctx.checkpoint_path

        # --- resume ---
        if resume:
            if checkpoint_path is None:
                raise ValueError(
                    "resume=True but no checkpoint provided"
                )

            start_epoch = (
                self.load_checkpoint(checkpoint_path) + 1
            )

        # --- wandb ---
        wandb_cfg = self.config.wandb_config
        self.run = wandb.init(
            project=wandb_cfg.project,
            name=wandb_cfg.title, #model_name
            config=self.config.model_dump()
        )

        # --- save path ---
        output_dir = Path(self.config.trainer_config.output_dir) ##alan
        self.run_dir = output_dir / self.run.name  ##alan/model_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        end_epoch = self.config.training_config.epoch_final

        for epoch in tqdm(range(start_epoch, end_epoch), unit="epoch"):
            
            start = time.time()
            train_loss = self.train_one_epoch()
            end = time.time()
            val_loss = self.validate_one_epoch()

            self.step_scheduler(train_loss, val_loss)

            self.log(epoch, train_loss, val_loss, end-start)

            self.maybe_checkpoint(epoch)

        self.run.finish()

        return self.net, self.run_dir

    # ------------------------
    # CORE METHODS
    # ------------------------
    def train_one_epoch(self):
        self.net.train()

        losses = []

        for batches in islice(zip(*self.train_loaders), self.config.training_config.gradient_steps_train):
                
                batch_dict = self.dataset.reconstruct_batch(self.train_keys, batches)
                processed_batch = self.pipe(batch_dict) #modifications- scaling and masking spec, prior expansion
                noisy_batch_dict = self.noise(processed_batch)
                theta, x = self.pipe.build_input(noisy_batch_dict)
                theta, x  = self._to_device(theta), self._to_device(x)

                self.optimizer.zero_grad()
                loss = self.loss_fn(theta, x)
                loss.backward()

                # gradient clipping
                clip = self.config.training_config.clip_grad_norm
                if clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                    
                self.optimizer.step()

                losses.append(loss.detach())

        return torch.stack(losses)

    def validate_one_epoch(self):

        self.net.eval()

        losses = []

        for batches in islice(zip(*self.valid_loaders), self.config.training_config.gradient_steps_valid):
                
                batch_dict = self.dataset.reconstruct_batch(self.valid_keys, batches)
                processed_batch = self.pipe(batch_dict) #modifications- scaling and masking spec, prior expansion
                noisy_batch_dict = self.noise(processed_batch)
                theta, x = self.pipe.build_input(noisy_batch_dict)
                theta, x  = self._to_device(theta), self._to_device(x)

                loss = self.loss_fn(theta, x)
                losses.append(loss.detach())

        return torch.stack(losses)

    # ------------------------
    # UTILITIES
    # ------------------------
    def compute_loss(self, batch):
        """
        Adapt this depending on your loss signature.
        """
        if self.ctx.simulator is not None:
            return self.loss_fn(batch, self.ctx.simulator)
        return self.loss_fn(batch)

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
        }

        if val_loss is not None:
            log_dict["val_loss"] = val_loss.item()

        wandb.log(log_dict)

    def maybe_checkpoint(self, epoch):
        interval = self.config.training_config.checkpoint_interval or 100

        if epoch > 0 and epoch % interval == 0:
            path = self.run_dir / f"checkpoint_{epoch}.pt"
            self.save_checkpoint(path, epoch)

    def _to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) for b in batch]
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        return batch.to(self.device)
    

    def load_checkpoint(self, path: str):

        checkpoint = torch.load(path, 
                                map_location = self.device)

        self.net.load_state_dict(
            checkpoint['estimator_state_dict']
        )

        if (
            self.optimizer
            and checkpoint.get('optimizer_state_dict')
        ):
            self.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )

        if (
            self.scheduler
            and checkpoint.get('scheduler_state_dict')
        ):
            self.scheduler.load_state_dict(
                checkpoint['scheduler_state_dict']
            )

        return checkpoint.get("epoch", 0)


    def save_checkpoint(self, path: str, epoch: int):
        checkpoint = {
        'epoch': epoch,

        'estimator_state_dict':
            self.net.state_dict(),

        'optimizer_state_dict':
            self.optimizer.state_dict()
            if self.optimizer else None,

        'scheduler_state_dict':
            self.scheduler.state_dict()
            if self.scheduler else None,
        }

        torch.save(checkpoint, path)

