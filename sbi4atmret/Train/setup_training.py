from sbi4atmret.MLmodel.estimator import estimator
from sbi4atmret.models.build import build_dataset, build_model
from sbi4atmret.utils import config
import torch
import torch.optim.lr_scheduler as sched
import wandb
from pathlib import Path
from tqdm import tqdm
from typing import Union

from .train_validate import train_epoch, validate_epoch, log_to_wandb
from ..models.Base import (
    setup_estimator,
    setup_optimizer_and_scheduler,
    setup_loss_and_prior,
    setup_pipe,
)
from ..config.configs import BaseConfig, DatasetConfig, EstimatorConfig
from scripts.plotting import plot_results


'''
TO DO:
here i setup training. 
load dataset and model.
update model depending on loading from checkpoint or start from scratch. 

and then call the main training loop.????


'''


def load_model(config: BaseConfig):
    full_config = BaseConfig(**config)  # Load full config
    model = build_model(full_config.ML_model_config)
    return model

def load_dataset_batchwise(config: BaseConfig, name: str):
    full_config = BaseConfig(**config)  # Load full config
    dataset_config = full_config.dataset
    training_config = full_config.training

    path = dataset_config.dataset_path
    batch_size = training_config.batch_size
    shuffle = dataset_config.shuffle

    dataset = H5Dataset(path / name, batch_size=batch_size, shuffle=shuffle)

    return dataset

def setup_training(config: BaseConfig, checkpoint_file_path: Union[Path, None] = None):
    # Load model and dataset
    model = load_model()

    savepath = Path(config.dataset.savepath)
    model_name = config.training.name
    runpath = savepath / model_name
    epoch_fin = config.training.epochs
    map_location = config.training.map_location

    if checkpoint_file_path.exists():
        print("Checkpoint found, resuming training!", flush=True)
        states = torch.load(savepath / model_name / ('states_' + str(epoch_fin) + '.pth'), map_location=map_location)
        estimator.load_state_dict(states['estimator'])


        print("CUDA information:")
        if not get_cuda_info():
            print("No CUDA devices found!\n")
        else:
            for key, value in get_cuda_info().items():
                print(f"  {key}: {value}")
            print()

    
        model, dataset = load_model_dataset_resume(
                                                    experiment_dir=args.config_dir,
                                                    checkpoint_name=args.checkpoint_name,
                                                    config=config,
        )

    else:
        print("No checkpoint found, starting training from scratch!", flush=True)
        model, dataset = load_model_dataset_new()

    dataset = load_dataset_batchwise()


    return model, dataset



