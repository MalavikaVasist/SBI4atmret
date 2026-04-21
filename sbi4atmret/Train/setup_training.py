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



def load_model(config: BaseConfig) -> Base:
    """
    Load and initialize the estimator model from configuration.
    
    Args:
        config: BaseConfig instance with model configuration
        
    Returns:
        Base instance with initialized estimator
        
    Raises:
        ValueError: If config is invalid or missing required fields
    """
    if not isinstance(config, BaseConfig):
        raise TypeError(f"Expected BaseConfig, got {type(config)}")
    
    if config.estimator is None:
        raise ValueError("Estimator configuration not found in config")
    
    logger.info("Loading model from configuration...")
    
    # Create Base instance and setup estimator
    model = Base(config).build()
    
    if model.estimator is None:
        raise RuntimeError("Failed to build estimator model")
    
    logger.info(f"Model loaded successfully: {type(model.estimator).__name__}")
    return model


def load_dataset_batchwise(
    config: BaseConfig, 
    dataset_name: str,
    split: str = 'train'
) -> H5Dataset:
    """
    Load dataset in batches using H5Dataset.
    
    Args:
        config: BaseConfig instance with dataset configuration
        dataset_name: Name of the dataset to load
        split: Dataset split ('train', 'valid', or 'test')
        
    Returns:
        H5Dataset instance
        
    Raises:
        ValueError: If config is invalid or dataset path doesn't exist
    """
    if not isinstance(config, BaseConfig):
        raise TypeError(f"Expected BaseConfig, got {type(config)}")
    
    if config.observation is None:
        raise ValueError("Observation configuration not found in config")
    
    # Construct dataset path
    dataset_config = config.observation
    if not hasattr(dataset_config, 'dataset_path') or dataset_config.dataset_path is None:
        raise ValueError("Dataset path not configured in observation config")
    
    # Get batch size from training config
    batch_size = 64  # Default batch size
    if config.training and hasattr(config.training, 'batch_size'):
        batch_size = config.training.batch_size
        if isinstance(batch_size, list):
            batch_size = batch_size[0]
    
    # Construct full path
    dataset_path_dict = dataset_config.dataset_path
    if isinstance(dataset_path_dict, dict):
        # Try to get path for the dataset
        path = dataset_path_dict.get(split, {}).get(dataset_name, None)
        if path is None:
            # Try alternate structure
            for key, val in dataset_path_dict.items():
                if isinstance(val, dict) and dataset_name in val:
                    path = val[dataset_name]
                    break
    else:
        path = str(dataset_path_dict)
    
    if path is None:
        raise ValueError(f"Dataset path for '{dataset_name}' not found in config")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    
    logger.info(f"Loading dataset from {path} with batch_size={batch_size}...")
    
    try:
        dataset = H5Dataset(path, batch_size=batch_size)
        logger.info(f"Dataset loaded successfully: {len(dataset) if hasattr(dataset, '__len__') else 'unknown'} samples")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def setup_training(
    config: Union[dict, BaseConfig], 
    checkpoint_path: Optional[Path] = None,
    device: str = 'cuda'
) -> Tuple[Base, H5Dataset, Optional[Dict[str, Any]]]:
    """
    Setup complete training environment: model, dataset, optimizer, scheduler, and loss.
    
    Args:
        config: Configuration dict or BaseConfig instance
        checkpoint_path: Path to checkpoint file (optional, for resuming training)
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model_base, dataset, checkpoint_state)
        where checkpoint_state contains resume information if checkpoint was loaded
        
    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If checkpoint file not found (when specified)
    """
    # Validate and convert config
    if isinstance(config, dict):
        try:
            config = BaseConfig(**config)
            logger.info("Configuration loaded from dict and validated with Pydantic")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}") from e
    elif not isinstance(config, BaseConfig):
        raise TypeError(f"Expected dict or BaseConfig, got {type(config)}")
    
    logger.info("=" * 60)
    logger.info("SETUP TRAINING")
    logger.info("=" * 60)
    
    # Check CUDA availability
    logger.info("Checking CUDA availability...")
    cuda_info = get_cuda_info()
    if cuda_info and device == 'cuda':
        logger.info("CUDA Information:")
        for key, value in cuda_info.items():
            logger.info(f"  {key}: {value}")
    else:
        if device == 'cuda':
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        logger.info("Using CPU for training")
    
    # Load and setup model
    logger.info("\n" + "-" * 60)
    logger.info("Loading Model...")
    logger.info("-" * 60)
    try:
        model = load_model(config)
        model.network_to_device(device=device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Setup optimizer, scheduler, and loss
    logger.info("\n" + "-" * 60)
    logger.info("Setting up Optimizer and Scheduler...")
    logger.info("-" * 60)
    try:
        optimizer, step, scheduler = model.initialize_op_scheduler(i=0)
        logger.info(f"Optimizer: {config.get_optimizer_config().type if config.get_optimizer_config() else 'None'}")
        logger.info(f"Scheduler: {config.get_scheduler_config().type if config.get_scheduler_config() else 'None'}")
    except Exception as e:
        logger.error(f"Failed to setup optimizer/scheduler: {e}")
        raise
    
    logger.info("\n" + "-" * 60)
    logger.info("Setting up Loss and Prior...")
    logger.info("-" * 60)
    try:
        loss, prior = model.setup_loss_and_prior(i=0)
        logger.info(f"Loss function: {config.get_loss_type(0)}")
    except Exception as e:
        logger.error(f"Failed to setup loss and prior: {e}")
        raise
    
    # Load dataset
    logger.info("\n" + "-" * 60)
    logger.info("Loading Dataset...")
    logger.info("-" * 60)
    try:
        dataset_name = config.observation.source if config.observation else "default"
        dataset = load_dataset_batchwise(config, dataset_name=dataset_name, split='train')
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Handle checkpoint loading
    checkpoint_state = None
    if checkpoint_path is not None:
        logger.info("\n" + "-" * 60)
        logger.info("Checkpoint Handling...")
        logger.info("-" * 60)
        
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            logger.info(f"Checkpoint found: {checkpoint_path}")
            try:
                model.load_estimator(str(checkpoint_path))
                checkpoint_state = {
                    'resumed': True,
                    'checkpoint_path': str(checkpoint_path)
                }
                logger.info("Checkpoint loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                raise
        else:
            logger.warning(f"Checkpoint path specified but not found: {checkpoint_path}")
            logger.info("Starting training from scratch")
    else:
        logger.info("No checkpoint specified, starting training from scratch")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SETUP COMPLETE")
    logger.info("=" * 60 + "\n")
    
    return model, dataset, checkpoint_state


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



