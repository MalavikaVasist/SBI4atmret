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

from ..models.Base import (
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




