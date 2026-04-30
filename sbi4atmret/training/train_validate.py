import time
from sbi4atmret.Train import args
from sbi4atmret.Train.train import load_model_dataset_resume
import torch
import torch.optim.lr_scheduler as sched
import wandb
from pathlib import Path
from tqdm import tqdm
from itertools import islice
from typing import Any, Mapping

from lampe.utils import GDStep
from zuko.distributions import BoxUniform

from ..models.base import (
    setup_estimator,
    setup_optimizer_and_scheduler,
    setup_loss_and_prior,
    setup_pipe,
)
from scripts.plotting import plot_results


'''
here you call the loaded estimator/model inside one training epoch. 
the training loop goes like this :
    - call the laoded estimator/model from the load_estimator
    - compute the dataset via the pipe 
    - compute the loss and backpropagate
    - return the loss and the time taken for the epoch.
'''

def execute_training(model:"Base", dataset):
    # Check if checkpoint file exists
    training_config = model.config.training
    model.run_training()
    print()


def train_epoch(estimator, step, trainsets, simulator, pipe, config: Mapping[str, Any]):
    estimator.train()
    start = time.time()

    for batches in islice(zip(*train_loaders), self.config.training_config.gradient_steps_train):
        batch_dict = dataset.reconstruct_batch(train_keys, batches)
        batches = dataset.modify_batch(train_keys, batches)  
        loss_batch = self.pipe(batch_dict, self.loss)

    loss_train_list = []
    for data_tuple in islice(zip(*trainsets), config["training"]["gradient_steps_train"]):
        sim_models = {}
        idx = 0
        for atm_type in config['simulator']["type"]:
            sim_models[atm_type] = {}
            for instrument in config['instruments']:
                sim_models[atm_type][instrument] = data_tuple[idx]
                idx += 1
        # Assuming the last type and instrument for pipe
        atm_type = config['simulator']["type"][0]
        instrument = config['instruments'][0]
        output = pipe(sim_models, simulator[atm_type][instrument])
        loss_train = step(output)
        loss_train_list.append(loss_train)
    losses_train = torch.stack(loss_train_list).cpu().numpy()
    end = time.time()
    return losses_train, end - start


def validate_epoch(estimator, validsets, simulator, pipe, step, config: Mapping[str, Any]):
    estimator.eval()
    loss_valid_list = []
    with torch.no_grad():
        for data_tuple in islice(zip(*validsets), config["training"]["gradient_steps_valid"]):
            sim_models = {}
            idx = 0
            for atm_type in config['simulator']["type"]:
                sim_models[atm_type] = {}
                for instrument in config['instruments']:
                    sim_models[atm_type][instrument] = data_tuple[idx]
                    idx += 1
            atm_type = config['simulator']["type"][0]
            instrument = config['instruments'][0]
            output = pipe(sim_models, simulator[atm_type][instrument])
            loss_valid = step(output)
            loss_valid_list.append(loss_valid)
    losses_val = torch.stack(loss_valid_list).cpu().numpy()
    return losses_val


def log_to_wandb(run, lr, loss_train, loss_val, nans, nans_val, speed, train_len, val_len):
    run.log({
        'lr': lr,
        'loss': loss_train,
        'loss_val': loss_val,
        'nans': nans,
        'nans_val': nans_val,
        'speed': speed,
        'trainigset_len': train_len,
        'validationset_len': val_len,
    })




import torch
import wandb
from tqdm import tqdm
from pathlib import Path


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
        self.val_keys, self.val_loaders = context.val_lists 
        self.pipe = context.pipe

        self.device = context.device

        # move model + prior
        self.net.to(self.device)
        if self.ctx.prior is not None:
            self.ctx.prior = self.ctx.prior.to(self.device)

    # ------------------------
    # PUBLIC API
    # ------------------------
    def train(self, resume_from=None):

        checkpoint_path = resume_from or self.ctx.checkpoint_path

        # --- resume ---
        if checkpoint_path:
            self.model.load_from_checkpoint(checkpoint_path)

        # --- wandb ---
        wandb_cfg = self.config.wandb_config
        self.run = wandb.init(
            project=wandb_cfg.project,
            name=wandb_cfg.title,
            config=self.config.model_dump()
        )

        # --- save path ---
        savepath = Path(self.config.dataset_config.savepath)
        self.runpath = savepath / self.run.name
        self.runpath.mkdir(parents=True, exist_ok=True)

        epochs = self.config.training_config.epochs

        for epoch in range(epochs):

            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)

            self.step_scheduler(train_loss, val_loss)

            self.log(epoch, train_loss, val_loss)

            self.maybe_checkpoint(epoch)

        self.run.finish()

        return self.net, self.runpath

    # ------------------------
    # CORE METHODS
    # ------------------------
    def train_one_epoch(self, epoch):
        self.net.train()

        losses = []

        for loader in self.train_loaders:
            for batch in loader:

                batch = self._to_device(batch)

                self.optimizer.zero_grad()

                loss = self.compute_loss(batch)

                loss.backward()

                # gradient clipping
                clip = self.config.training_config.clip_grad_norm
                if clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                self.optimizer.step()

                losses.append(loss.detach())

        return torch.nanmean(torch.stack(losses))

    def validate_one_epoch(self, epoch):
        if not self.val_loaders:
            return None

        self.net.eval()
        losses = []

        with torch.no_grad():
            for loader in self.val_loaders:
                for batch in loader:

                    batch = self._to_device(batch)

                    loss = self.compute_loss(batch)
                    losses.append(loss)

        return torch.nanmean(torch.stack(losses))

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

    def log(self, epoch, train_loss, val_loss):
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        if val_loss is not None:
            log_dict["val_loss"] = val_loss.item()

        wandb.log(log_dict)

    def maybe_checkpoint(self, epoch):
        interval = self.config.training_config.checkpoint_interval or 100

        if epoch > 0 and epoch % interval == 0:
            path = self.runpath / f"checkpoint_{epoch}.pt"
            self.model.save_model(path)

    def _to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) for b in batch]
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        return batch.to(self.device)