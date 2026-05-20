
from logging import config
from pathlib import Path
from typing import Union, Optional
import logging

from sbi4atmret.runtime.setup_runtime import CoreRuntimeContext, setup_runtime
import torch
import torch.optim.lr_scheduler as sched

from ..config.configs import BaseConfig
from ..torchutils.general import get_cuda_info, to_device
from ..datasets.DatasetBase import Dataset
from ..observations.ObservationBase import Observation
from ..domain.builders import build_domain_context

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class TrainingContext:

    runtime: CoreRuntimeContext

    # training components
    optimizer: Any
    scheduler: Any
    loss_fn: Any

    # dataset components
    train_lists: Any
    valid_lists: Any

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

    # --- setup runtime context ---
    runtime = setup_runtime(
        config=config,
        dataset=dataset,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # --- config ---
    if isinstance(config, dict):
        config = BaseConfig(**config)
    elif not isinstance(config, BaseConfig):
        raise TypeError(f"Expected dict or BaseConfig, got {type(config)}")

    logger.info("=" * 60)
    logger.info("BUILDING TRAINING CONTEXT")
    logger.info("=" * 60)

    # --- dataset ---
    train_keys, train_loaders = dataset.flatten_loaders(runtime.dataloaders_dict["train"])
    valid_keys, valid_loaders = dataset.flatten_loaders(runtime.dataloaders_dict["valid"])

    # --- prior ---
    logger.info("Building prior...")
    prior = config.build_prior().to(device)

    # --- optimizer ---
    logger.info("Building optimizer...")
    optimizer = config.build_optimizer(model.estimator.parameters())

    # --- scheduler ---
    logger.info("Building scheduler...")
    scheduler = config.build_scheduler(optimizer)

    # --- loss ---
    logger.info("Building loss...")
    loss_fn = config.build_loss(model.estimator.to(device), prior)

    logger.info("Training setup ready.")

    return TrainingContext(
        runtime=runtime,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_lists=(train_keys, train_loaders),
        valid_lists=(valid_keys, valid_loaders),
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        device=device,
    )


