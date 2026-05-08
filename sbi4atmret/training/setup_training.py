
from pathlib import Path
from typing import Union, Optional
import logging

import torch
import torch.optim.lr_scheduler as sched

from ..config.configs import BaseConfig
from ..torchutils.general import get_cuda_info, to_device
from ..datasets.DatasetBase import Dataset
from ..observations.ObservationBase import Observation

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class TrainingContext:

    # training components
    optimizer: torch.optim.Optimizer
    scheduler: Optional[Any]
    loss_fn: Any

    # dataset components
    train_lists = list(Any, Any), 
    val_lists = list(Any, Any), 
    pipe = Any

    # domain-specific
    prior: Any

    # bookkeeping
    checkpoint_path: Optional[str]
    device: str



def setup_training(
    config: Union[dict, BaseConfig],
    model,
    dataset: Dataset,
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda",
) -> TrainingContext:
    """
    Build the full training context (everything except the Trainer).
    """

    # --- config ---
    if isinstance(config, dict):
        config = BaseConfig(**config)
    elif not isinstance(config, BaseConfig):
        raise TypeError(f"Expected dict or BaseConfig, got {type(config)}")

    logger.info("=" * 60)
    logger.info("BUILDING TRAINING CONTEXT")
    logger.info("=" * 60)

    # --- device ---
    cuda_info = get_cuda_info(config)
    if device == "cuda" and not cuda_info:
        logger.warning("CUDA not available, switching to CPU")
        device = "cpu"

    
    # --- dataset / dataloaders ---
    logger.info("Loading dataset...")
    dataloaders_dict = dataset.return_dataloaders_dict()

    train_keys, train_loaders = dataset.flatten_loaders(dataloaders_dict["train"])
    val_keys, val_loaders = dataset.flatten_loaders(dataloaders_dict["valid"])

    # --- simulator ---
    simulator_dict = config.build_simulators()

    # --- observation ---

    obs = Observation(
        observation_config=config.observation_config,
        dataset_config=config.dataset_config,
        simulator_config=config.simulator_config,
            )
    
    # --- pipeline

    pipe = config.build_pipe(
        simulators=simulator_dict,
        observation=obs,
    )

    # --- prior ---
    logger.info("Building prior...")
    prior = config.build_prior()

    # --- optimizer ---
    logger.info("Building optimizer...")
    optimizer = config.build_optimizer(model.estimator.parameters())

    # --- scheduler ---
    logger.info("Building scheduler...")
    scheduler = config.build_scheduler(optimizer)

    # --- loss ---
    logger.info("Building loss...")
    loss_fn = config.build_loss(model.estimator, prior)

    # --- validating checkpoint path ---
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            logger.info(f"Checkpoint found: {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Training setup ready.")

    return TrainingContext(
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        prior=prior,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        device=device,
        train_lists = (train_keys, train_loaders), 
        val_lists = (val_keys, val_loaders), 
        pipe = pipe
    )



