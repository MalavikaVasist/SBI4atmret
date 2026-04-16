#!/usr/bin/env python

import os
import torch
from pathlib import Path

import yaml
from pydantic import ValidationError
from sbi4atmret.config.models import MLModelConfig
from sbi4atmret.models.Base import Base
from sbi4atmret.Train.args import parse_args, get_config_path
from sbi4atmret.simulators.simulator import build_simulator
from scripts.data import load_observations_data, load_datasets
from sbi4atmret.Train.train import run_training

from dawgz import job, schedule

# Parse command line arguments
args = parse_args()
config_path = get_config_path(args.config_dir)

# Load configuration from YAML and validate with Pydantic
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)

try:
    validated_config = MLModelConfig(**config_dict)
except ValidationError as exc:
    raise RuntimeError(f"Configuration validation failed: {exc}") from exc

# Use the validated config as a plain dict for backward compatibility with existing code
config = validated_config.model_dump()

# Build a Base helper
base = Base(config)

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

array, cpus, gpus, ram, time = config['wandb']['array'], config['wandb']['cpus'], config['wandb']['gpus'], config['wandb']['ram'], config['wandb']['time']

@job(array=array, cpus=cpus, gpus=gpus, ram=ram, time=time)
def train(i: int):
    # Select the config for this model index
    selected_config = base.select_index_config(i)
    # Load datasets
    datasets = load_datasets(selected_config, scratch, i)
    estimator, runpath = run_training(
        selected_config,
        datasets,
        simulator,
        observation,
        checkpoint_fn=save_checkpoint,
        checkpoint_interval=selected_config.get("training", {}).get("checkpoint_interval", 50),
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