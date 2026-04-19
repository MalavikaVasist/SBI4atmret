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
from ..config.configs import BaseConfig
from scripts.plotting import plot_results


'''
TO DO:
here i setup training. 
load dataset and model.
update model depending on loading from checkpoint or start from scratch. 

and then call the main training loop.????
'''

def load_model():
    '''
    model =     
    '''
    # Placeholder for loading model
    model = build_model(config)  # Replace with actual model initialization
    return model

def load_dataset():
    # Placeholder for loading dataset
    dataset = None  # Replace with actual dataset loading
    return dataset


def load_model_dataset_new():
    # Placeholder for loading model and dataset from scratch
    model = None  # Replace with actual model initialization
    dataset = None  # Replace with actual dataset loading
    return model, dataset


def load_model_dataset_resume():
    # Placeholder for loading model and dataset from checkpoint
    model = None  # Replace with actual model loading from checkpoint
    dataset = None  # Replace with actual dataset loading from checkpoint
    return model, dataset



def setup_training(config: BaseConfig):
    # Load model and dataset
    model = load_model()
    dataset = load_dataset()

    if checkpoint_file_path.exists():
        print("Checkpoint found, resuming training!", flush=True)
        model, dataset = load_model_dataset_resume(
            experiment_dir=args.config_dir,
            checkpoint_name=args.checkpoint_name,
            config=config,
        )

    else:
        print("No checkpoint found, starting training from scratch!", flush=True)
        model, dataset = load_model_dataset_new()


    return model, dataset



