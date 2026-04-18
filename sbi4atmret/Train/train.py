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


def run_training(config: Union[dict, BaseConfig], datasets, simulator, observation, checkpoint_fn=None, checkpoint_interval=50):
    """
    Run training loop with Pydantic config.
    
    Args:
        config: Configuration dict or BaseConfig instance
        datasets: Training/validation/test datasets
        simulator: Simulator for forward modeling
        observation: Observation data
        checkpoint_fn: Optional checkpoint saving function
        checkpoint_interval: Interval for checkpointing
        
    Returns:
        Tuple of (estimator, runpath)
    """
    # Convert to BaseConfig if dict
    if isinstance(config, dict):
        config = BaseConfig(**config)
    
    # Use attribute access on Pydantic model
    model_name = str(config.ML_model_config.embedding.miri) if config.ML_model_config else "default"
    
    run = wandb.init(
        project=config.wandb['project'] if config.wandb else 'default',
        config={},
        name=f"{model_name}"
    )

    estimator = setup_estimator(config, 0)
    optimizer, step, scheduler = setup_optimizer_and_scheduler(estimator, config, 0)
    loss, prior = setup_loss_and_prior(estimator, config, 0)
    pipe = setup_pipe(config, loss)

    savepath = Path(config.paths['savepath'][0]) if config.paths else Path('./runs')
    runpath = savepath / run.name
    runpath.mkdir(parents=True, exist_ok=True)

    # Use attribute access for training config
    start_epoch = config.training.epoch_fin if config.training else 0
    end_epoch = config.training.epochs if config.training else 100
    
    for epoch in tqdm(range(start_epoch, end_epoch), unit='epoch'):
        # Build dataset lists (compatible with existing data loading)
        trainsets = [datasets[atm_type][instrument]['train'] 
                    for atm_type in config.wandb['simulator']['type'] if config.wandb
                    for instrument in config.wandb['instruments'] if config.wandb]
        losses_train, duration = train_epoch(estimator, step, trainsets, simulator, pipe, config.model_dump())

        validsets = [datasets[atm_type][instrument]['valid']
                    for atm_type in config.wandb['simulator']['type'] if config.wandb
                    for instrument in config.wandb['instruments'] if config.wandb]
        losses_val = validate_epoch(estimator, validsets, simulator, pipe, step, config.model_dump())

        log_to_wandb(run, optimizer.param_groups[0]['lr'], 
                     torch.nanmean(losses_train), torch.nanmean(losses_val),
                     torch.isnan(losses_train).mean(), torch.isnan(losses_val).mean(),
                     len(losses_train) / duration, len(losses_train), len(losses_val))

        _step_scheduler(scheduler, torch.nanmean(losses_val))

        if checkpoint_fn is not None and epoch > 100 and epoch % checkpoint_interval == 0:
            checkpoint_fn(runpath, estimator, optimizer, epoch)

        # Use attribute access for stop criterion
        if config.training and config.training.stop_criterion == 'early' and scheduler is not None and optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
            break

    # Post-training plotting
    testsets = [datasets[atm_type][instrument]['test']
               for atm_type in config.wandb['simulator']['type'] if config.wandb
               for instrument in config.wandb['instruments'] if config.wandb]
    plot_dict = plot_results(runpath, estimator, observation, testsets, pipe, simulator, config.model_dump())

    for key, fig in plot_dict.items():
        run.log({key: wandb.Image(fig)})

    run.finish()
    return estimator, runpath