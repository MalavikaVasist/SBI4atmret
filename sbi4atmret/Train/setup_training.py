import torch
import torch.optim.lr_scheduler as sched
import wandb
from pathlib import Path
from tqdm import tqdm
from typing import Union, Optional, Tuple, Dict, Any
import logging

from lampe.data import H5Dataset

from .train_validate import train_epoch, validate_epoch, log_to_wandb
from ..torchutils.general import get_cuda_info
from ..models.Base import Base
from ..models.build import build_model
from ..config.configs import BaseConfig, EstimatorConfig, TrainingConfig
from ..utils.load_utils import load_callable

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

'''
load daata and model and setup the loss, optimizer and scheduler from scratch or checkpoint. 
'''

def setup_training(
    config: Union[dict, BaseConfig],
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Consistent setup aligned with Base class implementation.
    """

    # --- config ---
    if isinstance(config, dict):
        config = BaseConfig(**config)
    elif not isinstance(config, BaseConfig):
        raise TypeError(f"Expected dict or BaseConfig, got {type(config)}")

    logger.info("=" * 60)
    logger.info("SETUP TRAINING")
    logger.info("=" * 60)

    # --- device ---
    cuda_info = get_cuda_info()
    if device == "cuda" and not cuda_info:
        logger.warning("CUDA not available, switching to CPU")
        device = "cpu"

    # --- build base + estimator ---
    logger.info("Building estimator...")
    base = Base(config)
    base.setup_estimator()
    base.network_to_device(device=device)

    if base.estimator is None:
        raise RuntimeError("Estimator not initialized")

    # --- optimizer + scheduler ---
    logger.info("Setting up optimizer and scheduler...")
    optimizer, scheduler = base.setup_optimizer_and_scheduler()

    # --- loss ---
    logger.info("Setting up loss...")
    loss_fn = base.setup_loss()

    # --- dataset ---
    logger.info("Loading dataset...")
    dataset_name = config.observation.source if config.observation else "default"

    train_loader = load_dataset_batchwise(config, dataset_name, split="train")

    val_loader = None
    try:
        val_loader = load_dataset_batchwise(config, dataset_name, split="valid")
    except Exception:
        logger.warning("Validation dataset not found, continuing without it")

    # --- checkpoint ---
    checkpoint_state = {"resumed": False}

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            base.load_from_checkpoint(str(checkpoint_path))
            checkpoint_state["resumed"] = True
            checkpoint_state["path"] = str(checkpoint_path)
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Setup complete.")

    return {
        "model": base.estimator,   # torch model
        "base": base,              # full wrapper (keeps optimizer, scheduler, etc.)
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_fn": loss_fn,
        "checkpoint": checkpoint_state,
        "device": device,
    }

# Example usage:
if __name__ == "__main__":
    import yaml
    
    # Load YAML config
    with open("config_one.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Setup training with no checkpoint
    model, dataset, _ = setup_training(config_dict, device='cuda')
    
    # OR resume from checkpoint
    # model, dataset, checkpoint_state = setup_training(
    #     config_dict, 
    #     checkpoint_path=Path("./runs/model_checkpoint.pth"),
    #     device='cuda'
    # )
    
    logger.info(f"Model ready for training")
    logger.info(f"Dataset: {type(dataset).__name__}")
    logger.info(f"Model estimator: {type(model.estimator).__name__}")



