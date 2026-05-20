
from dataclasses import dataclass
from typing import Any, Optional, Union
import logging
from pathlib import Path

from sbi4atmret.datasets.DatasetBase import Dataset
from sbi4atmret.runtime.setup_runtime import CoreRuntimeContext, setup_runtime
from sbi4atmret.config.configs import BaseConfig
from sbi4atmret.torchutils.general import get_cuda_info

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class EvaluationContext:

    runtime: CoreRuntimeContext

    # dataset components
    test_lists: Any

def setup_evaluation(
    config: Union[dict, BaseConfig],
    dataset: Dataset,
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda",
) -> EvaluationContext:
    """
    Build the full evaluation context.
    """

    # --- setup runtime context ---
    runtime = setup_runtime(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # --- config ---
    if isinstance(config, dict):
        config = BaseConfig(**config)
    elif not isinstance(config, BaseConfig):
        raise TypeError(f"Expected dict or BaseConfig, got {type(config)}")

    logger.info("=" * 60)
    logger.info("BUILDING EVALUATION CONTEXT")
    logger.info("=" * 60)

    # --- dataset ---
    logger.info("Loading dataset...")
    dataloaders_dict = dataset.return_dataloaders_dict()
    test_keys, test_loaders = dataset.flatten_loaders(dataloaders_dict["test"])

    logger.info("Evaluation setup ready.")

    return EvaluationContext(
        runtime=runtime,
        test_lists=(test_keys, test_loaders),
    )

