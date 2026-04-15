#!/usr/bin/env python

import os
import torch
from pathlib import Path

from ..sbi4atmret.utils.config import Config
from ..sbi4atmret.simulators.simulator import build_simulator
from .data import load_observations_data, load_datasets
from ..sbi4atmret.Train.trainer import run_training

from dawgz import job, schedule

# Load configuration
config = Config("config.yaml")

scratch = os.environ.get(config['scratch_env'])
home = os.environ.get(config['home_env'])

# Load observations
observation = load_observations_data(config)

# Build simulator
simulator = build_simulator(config)


def save_checkpoint(runpath, estimator, optimizer, epoch):
    torch.save({
        'estimator': estimator.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, runpath / f'states_{epoch}.pth')

array, cpus, gpus, ram, time = config["wandb"]["array"], config["wandb"]["cpus"], config["wandb"]["gpus"], config["wandb"]["ram"], config["wandb"]["time"]

# Run training
@job(array=array, cpus=cpus, gpus=gpus, ram=ram, time=time)
def train(i: int):
    # Load datasets
    datasets = load_datasets(config, scratch, i)
    i = 9  # Hardcoded for now
    estimator, runpath = run_training(
        config,
        i,
        datasets,
        simulator,
        observation,
        checkpoint_fn=save_checkpoint,
        checkpoint_interval=config.get("training", {}).get("checkpoint_interval", 50),
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