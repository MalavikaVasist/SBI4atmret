import torch
import torch.optim.lr_scheduler as sched
import wandb
from pathlib import Path
from tqdm import tqdm
from typing import Any, Mapping

from .train_validate import train_epoch, validate_epoch, log_to_wandb
from ..models.Base import (
    setup_estimator,
    setup_optimizer_and_scheduler,
    setup_loss_and_prior,
    setup_pipe,
)
from scripts.plotting import plot_results


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


def run_training(config: Mapping[str, Any], datasets, simulator, observation, checkpoint_fn=None, checkpoint_interval=50):
    model_configs = config.get("ML_model_configs") or config.get("model_configs")
    training_configs = config["training"]

    name = str(model_configs["model_name"])
    run = wandb.init(project=config["wandb"]["project"], config={}, name=f"{model_configs['model']}_{name}")

    estimator = setup_estimator(config, 0)
    optimizer, step, scheduler = setup_optimizer_and_scheduler(estimator, config, 0)
    loss, prior = setup_loss_and_prior(estimator, config, 0)
    pipe = setup_pipe(config, loss)

    savepath = Path(config["sim_paths"]["savepath"][0])
    runpath = savepath / run.name
    runpath.mkdir(parents=True, exist_ok=True)

    start_epoch = training_configs["epoch_fin"]
    end_epoch = training_configs["epochs"]
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

        if training_configs.get("stop_criterion") == 'early' and scheduler is not None and optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
            break

    # Post-training plotting
    testsets = [datasets[atm_type][instrument]['test'] for atm_type in config['simulator']["type"] for instrument in config['instruments']]
    plot_dict = plot_results(runpath, estimator, observation, testsets, pipe, simulator, config)

    for key, fig in plot_dict.items():
        run.log({key: wandb.Image(fig)})

    run.finish()
    return estimator, runpath