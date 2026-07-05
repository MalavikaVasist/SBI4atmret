from dataclasses import dataclass
from typing import Any, Optional, Union
import logging
from pathlib import Path

import torch

from sbi4atmret.config.configs import BaseConfig
from sbi4atmret.datasets.DatasetBase import Dataset
from sbi4atmret.domain.builders import build_domain_context
from sbi4atmret.domain.context import DomainContext
from sbi4atmret.observations.ObservationBase import Observation

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class CoreRuntimeContext:
    domain: DomainContext
    checkpoint_path: Optional[str]
    device: str


def setup_runtime(
    config: Union[dict, BaseConfig],
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda",
) -> CoreRuntimeContext:
    """Build the core runtime context shared by training and evaluation."""

    # --- config ---
    if isinstance(config, dict):
        config = BaseConfig(**config)
    elif not isinstance(config, BaseConfig):
        raise TypeError(f"Expected dict or BaseConfig, got {type(config)}")

    logger.info("=" * 60)
    logger.info("BUILDING CORE RUNTIME CONTEXT")
    logger.info("=" * 60)

    # --- device ---
    device = device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- simulator ---
    logger.info("Building simulators...")
    simulator_dict = config.build_simulators()

    # --- observation ---
    logger.info("Building observation...")
    obs = Observation(
        observation_config=config.observation_config,
        dataset_config=config.dataset_config,
        simulator_config=config.simulator_config,
    )

    # --- domain context ---
    logger.info("Building domain context...")
    domain = build_domain_context(
        simulator_dict=simulator_dict,
        observation=obs,
        config=config,
    )

    # --- validating checkpoint path ---
    if checkpoint_path:
        if checkpoint_path.exists():
            logger.info(f"Checkpoint found: {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Runtime setup ready.")

    return CoreRuntimeContext(
        domain=domain,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        device=device,
    )

