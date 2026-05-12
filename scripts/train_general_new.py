#!/usr/bin/env python

import os
from scripts.plotting import plot_results
import torch
from pathlib import Path

import yaml
from pydantic import ValidationError
from sbi4atmret.config.configs import BaseConfig
from sbi4atmret.training.args import parse_args, get_config_path
from sbi4atmret.simulators.simulator import build_simulator
from sbi4atmret.datasets.DatasetBase import Dataset
from sbi4atmret.training import setup_training
from sbi4atmret.training.train_validate import Trainer
from sbi4atmret.models.ModelBase import BaseModel

from dawgz import job, schedule

'''
load args
find the experiment directory
'''

# Parse command line arguments
args = parse_args()
config_path = get_config_path(args.config_dir)

# Load configuration from YAML and validate with Pydantic
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)

'''
load configs from Base
'''

try:
    # Validate and keep as Pydantic model throughout
    config = BaseConfig(**config_dict)
except ValidationError as exc:
    raise RuntimeError(f"Configuration validation failed: {exc}") from exc


'''
from args see what to do plot or train 

'''

if args.action == 'train':

    '''
    if train see if i'm resuming or starting from scratch
    load model and dataset depending on whether I'm resuming or starting from scratch
    '''
    
    model = BaseModel(config)
    model.build()

    dataset = Dataset(config)

    checkpoint_file_path = None
    if args.checkpoint_path: 
        print("Checkpoint found, resuming training!", flush=True)
        checkpoint_file_path = Path(args.checkpoint_path)

    ctx = setup_training(config = config, 
                        model = model, 
                        dataset = dataset, 
                        checkpoint_path = checkpoint_file_path, 
                        device = "cuda")

    trainer = Trainer(
        model=model,
        context=ctx,
        config=config,
    )

    trainer.train(resume = args.resume)

    print()


    '''
    
    train model until time limit is reached or early stopping or full
    if completed, end the job
    '''


if args.action == 'plot':
    print("Plotting results!", flush=True)
    plot_results(args.experiment_dir, config)

scratch = os.environ.get(config.paths['scratch_env'] if config.paths else '')
home = os.environ.get(config.paths['home_env'] if config.paths else '')

# Load observations
observation = load_observations_data(config.model_dump())

# Build simulator
simulator = build_simulator(config.model_dump())


def save_checkpoint(runpath, estimator, optimizer, epoch):
    torch.save({
        'estimator': estimator.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, runpath / f'states_{epoch}.pth')

array = config.wandb['array'] if config.wandb else 1
cpus = config.wandb['cpus'] if config.wandb else 1
gpus = config.wandb['gpus'] if config.wandb else 0
ram = config.wandb['ram'] if config.wandb else '16GB'
time = config.wandb['time'] if config.wandb else '01:00:00'

@job(array=array, cpus=cpus, gpus=gpus, ram=ram, time=time)
def train(i: int):
    # Select the config for this model index using Pydantic
    selected_config = base.select_index_config(i)
    # Load datasets
    datasets = load_datasets(selected_config.model_dump(), scratch)
    estimator, runpath = run_training(
        selected_config,
        datasets,
        simulator,
        observation,
        checkpoint_fn=save_checkpoint,
        checkpoint_interval=selected_config.training.checkpoint_interval if selected_config.training else 50,
    )

if __name__ == '__main__':
    schedule(
        train, 
        name='Training',
        backend='slurm',
        env=[
            'source ~/.bashrc',
            'conda activate WISEJ1828',
            'export WANDB_SILENT=true',
        ]
    )