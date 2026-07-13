#!/usr/bin/env python

import os
os.environ['pRT_input_data_path'] = '/media/mvasist/Elements/PhDprojects/scratch/input_data_v2.4.9/input_data'
os.environ['WANDB_API_KEY'] = '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'

from pathlib import Path
import yaml
from pydantic import ValidationError

import torch

from sbi4atmret.config.configs import BaseConfig
from sbi4atmret.training.args import parse_args, get_config_path
from sbi4atmret.datasets.DatasetBase import Dataset
from sbi4atmret.training.setup_training import setup_training
from sbi4atmret.evaluation.setup_evaluation import setup_evaluation
from sbi4atmret.training.train_validate import Trainer
from sbi4atmret.models.ModelBase import BaseModel
from sbi4atmret.evaluation.EvaluateBase import BaseEvaluator

# Parse command line arguments
args = parse_args()
config_path = get_config_path(args.config_dir)

# Load configuration from YAML and validate with Pydantic
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)

try:
    # Validate and keep as Pydantic model throughout
    config = BaseConfig(**config_dict)
except ValidationError as exc:
    raise RuntimeError(f"Configuration validation failed: {exc}") from exc

# Build model and dataset
model = BaseModel(config)
model.build()

dataset = Dataset(config)

# Handle checkpoint path
checkpoint_file_path = None
if args.checkpoint_path:
    checkpoint_file_path = Path(args.checkpoint_path)

# If --name is given and no checkpoint path, auto-resolve for evaluation
if args.action == "evaluate" and checkpoint_file_path is None and args.name:
    auto_ckpt = Path(config.training_config.output_dir) / args.name / "checkpoints" / "latest.pt"
    if auto_ckpt.exists():
        checkpoint_file_path = auto_ckpt
        print(f"Auto-resolved checkpoint: {checkpoint_file_path}", flush=True)
    else:
        raise FileNotFoundError(
            f"No checkpoint found at {auto_ckpt}. "
            f"Pass --checkpoint-path explicitly or ensure training completed for run '{args.name}'."
        )

if args.checkpoint_path:
    print(f"Using checkpoint: {checkpoint_file_path}", flush=True)

if args.action == 'train':
    print("Starting training!", flush=True)

    ## setup the training context 
    ctx = setup_training(
        config=config,
        model=model,
        dataset=dataset,
        checkpoint_path=checkpoint_file_path,
        device="cuda",
    )

    trainer = Trainer(
        config=config,
        model=model,
        dataset=dataset,
        context=ctx,
        run_name=args.name,
    )

    trainer.train(resume=args.resume)
    print("Training complete!", flush=True)


elif args.action == "evaluate":
    print("Plotting and saving results!", flush=True)

    ctx = setup_evaluation(
        config=config,
        dataset=dataset,
        checkpoint_path=checkpoint_file_path,
        device="cuda",
    )

    evaluator = BaseEvaluator(
        model=model,
        context=ctx,
        config=config,
    )

    evaluator.run_all()
    print("Evaluation complete!", flush=True)















# scratch = os.environ.get(config.paths['scratch_env'] if config.paths else '')
# home = os.environ.get(config.paths['home_env'] if config.paths else '')

# # Load observations
# observation = load_observations_data(config.model_dump())

# # Build simulator
# simulator = build_simulator(config.model_dump())


# def save_checkpoint(runpath, estimator, optimizer, epoch):
#     torch.save({
#         'estimator': estimator.state_dict(),
#         'optimizer': optimizer.state_dict(),
#     }, runpath / f'states_{epoch}.pth')

# array = config.wandb['array'] if config.wandb else 1
# cpus = config.wandb['cpus'] if config.wandb else 1
# gpus = config.wandb['gpus'] if config.wandb else 0
# ram = config.wandb['ram'] if config.wandb else '16GB'
# time = config.wandb['time'] if config.wandb else '01:00:00'

# @job(array=array, cpus=cpus, gpus=gpus, ram=ram, time=time)
# def train(i: int):
#     # Select the config for this model index using Pydantic
#     selected_config = base.select_index_config(i)
#     # Load datasets
#     datasets = load_datasets(selected_config.model_dump(), scratch)
#     estimator, runpath = run_training(
#         selected_config,
#         datasets,
#         simulator,
#         observation,
#         checkpoint_fn=save_checkpoint,
#         checkpoint_interval=selected_config.training.checkpoint_interval if selected_config.training else 50,
#     )

# if __name__ == '__main__':
#     schedule(
#         train, 
#         name='Training',
#         backend='slurm',
#         env=[
#             'source ~/.bashrc',
#             'conda activate WISEJ1828',
#             'export WANDB_SILENT=true',
#         ]
#     )