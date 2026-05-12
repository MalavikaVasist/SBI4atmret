import torch
from typing import Union

from ..config.configs import BaseConfig
from estimator.base import EstimatorBase

class BaseModel:
    """
    Base class for all models providing common setup and utility methods.
    """

    def __init__(self, config):
        self.config = config 

    
    def build(self):
        # --- Build components ---
        self.embedding = self.config.build_embedding()
        self.flow = self.config.build_flow()

        # --- Compose estimator ---
        self.estimator = EstimatorBase(self.flow, self.embedding)

        return self
    

    def save_weights(self, path):

        torch.save(
            self.estimator.state_dict(),
            path
        )

    def load_weights(self, path):

        state_dict = torch.load(path)

        self.estimator.load_state_dict(state_dict)      

    



    # def train(
    #     self, ctx: TrainingContext):

    #     """
    #     Train the model using already-built components.
    #     """

    #     optimizer = ctx.optimizer
    #     loss_fn = ctx.loss_fn
    #     .
    #     .
    #     .
    #     .




    #     # --- Build if not already done ---
    #     if self.estimator is None:
    #         self.build()

    #     loss_fn.estimator = self.estimator



    #     ##to device all 
    #     prior = to_device(prior, device)




    #     # --- WandB ---
    #     wandb_cfg = self.config.wandb_config
    #     run = wandb.init(
    #         project=wandb_cfg["project"],
    #         name=wandb_cfg.get("title", "run"),
    #         config=self.config.model_dump()
    #     )

    #     # --- Paths ---
    #     self.savepath = Path(self.config.dataset_config.savepath)
    #     runpath = self.savepath / run.name
    #     runpath.mkdir(parents=True, exist_ok=True)

    #     # --- Epochs ---
    #     start_epoch = self.config.training_config.epoch_fin
    #     end_epoch = self.config.training_config.epochs

    #     dataset = Dataset(self.config.dataset_config) 
    #     dataloaders_dict = dataset.return_dataloaders_dict()

    #     train_keys, train_loaders = dataset.flatten_loaders(dataloaders_dict["train"])
    #     valid_keys, valid_loaders = dataset.flatten_loaders(dataloaders_dict["valid"])

    #     # --- Loop ---
    #     for epoch in tqdm(range(start_epoch, end_epoch), unit="epoch"):

    #         losses_train, duration = train_one_epoch(
    #                                             self.estimator,
    #                                             self.optimizer,
    #                                             train_keys, 
    #                                             train_loaders,
    #                                             self.simulator,
    #                                             self.loss,
    #                                             self.config.model_dump()
    #                                         )

    #         losses_val = validate_one_epoch(
    #                                         self.estimator,
    #                                         self.optimizer,
    #                                         valid_keys, 
    #                                         valid_loaders,
    #                                         self.simulator,
    #                                         self.loss,
    #                                         self.config.model_dump()
    #         )

    #         # --- Logging ---
    #         wandb.log({
    #             "lr": optimizer.param_groups[0]["lr"],
    #             "train_loss": torch.nanmean(losses_train),
    #             "val_loss": torch.nanmean(losses_val),
    #         })

    #         # --- Scheduler ---
    #         if scheduler is not None:
    #             try:
    #                 scheduler.step(torch.nanmean(losses_val))
    #             except TypeError:
    #                 scheduler.step()

    #         # --- Checkpoint ---
    #         interval = config.training_config.checkpoint_interval or 100
    #         if checkpoint_fn and epoch > 100 and epoch % interval == 0:
    #             checkpoint_fn(runpath, estimator, optimizer, epoch)

    #         # --- Early stopping ---
    #         if (
    #             config.training_config.stop_criterion == "early"
    #             and scheduler is not None
    #             and optimizer.param_groups[0]["lr"] <= getattr(scheduler, "min_lrs", [0])[0]
    #         ):
    #             break

        
    #     run.finish()

    #     return self.estimator, runpath


    # def test(eslf):
    #     # --- Test ---
    #     testsets = [
    #         datasets[cond][inst]["test"]
    #         for cond in self.config.dataset_config.dataset_path.keys()
    #         for inst in self.config.dataset_config.dataset_path[cond].keys()
    #     ]

    #     plot_dict = self.plot_results(
    #         runpath,
    #         self.estimator,
    #         observation,
    #         testsets,
    #         self.loss,
    #         simulator,
    #         self.config.model_dump()
    #     )

    #     for key, fig in plot_dict.items():
    #         run.log({key: wandb.Image(fig)})
