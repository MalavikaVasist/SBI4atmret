import time
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
from ...scripts.plotting import plot_results


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


def _step_scheduler(scheduler, metric=None):
    if scheduler is None:
        return
    if isinstance(scheduler, sched.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        try:
            scheduler.step()
        except TypeError:
            scheduler.step(metric)


def run_training(config: Mapping[str, Any], i: int, datasets, simulator, observation, checkpoint_fn=None, checkpoint_interval=50):
    model_configs = config.get("ML_model_configs") or config.get("model_configs")
    training_configs = config["training"]

    model_name_list = model_configs.get("model_name") or model_configs.get("name")
    name = str(model_name_list[i])
    run = wandb.init(project=config["wandb"]["project"], config={}, name=f"{model_configs['model']}_{name}")

    estimator = setup_estimator(config, i)
    optimizer, step, scheduler = setup_optimizer_and_scheduler(estimator, config, i)
    loss, prior = setup_loss_and_prior(estimator, config, i)
    pipe = setup_pipe(config, loss)

    savepath = Path(config["sim_paths"]["savepath"][0])
    runpath = savepath / run.name
    runpath.mkdir(parents=True, exist_ok=True)

    start_epoch = training_configs["epoch_fin"][i]
    end_epoch = training_configs["epochs"][i]
    for epoch in tqdm(range(start_epoch, end_epoch), unit='epoch'):
        trainsets = [datasets[atm_type][instrument]['train'] for atm_type in config['simulator']["type"] for instrument in config['instruments']]
        losses_train, duration = train_epoch(estimator, step, trainsets, simulator, pipe, config)

        validsets = [datasets[atm_type][instrument]['valid'] for atm_type in config['simulator']["type"] for instrument in config['instruments']]
        losses_val = validate_epoch(estimator, validsets, simulator, pipe, step, config)

        log_to_wandb(run, optimizer.param_groups[0]['lr'], 
                     torch.nanmean(losses_train), torch.nanmean(losses_val),
                     torch.isnan(losses_train).mean(), torch.isnan(losses_val).mean(),
                     len(losses_train) / duration, len(losses_train), len(losses_val))

        _step_scheduler(scheduler, torch.nanmean(losses_val))

        if checkpoint_fn is not None and epoch > 100 and epoch % checkpoint_interval == 0:
            checkpoint_fn(runpath, estimator, optimizer, epoch)

        if training_configs["stop_criterion"] == 'early' and scheduler is not None and optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
            break

    # Post-training plotting
    testsets = [datasets[atm_type][instrument]['test'] for atm_type in config['simulator']["type"] for instrument in config['instruments']]
    plot_dict = plot_results(runpath, estimator, observation, testsets, pipe, simulator, config)

    # Log plots to wandb
    import wandb
    for key, fig in plot_dict.items():
        run.log({key: wandb.Image(fig)})

    run.finish()
    return estimator, runpath