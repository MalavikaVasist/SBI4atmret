
from ..config.configs import BaseConfig

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def setup_evaluations(
    config: Union[dict, BaseConfig],
    estimator,
    dataset: Dataset,
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda",
) -> EvaluationContext:
    """
    Build the full evaluation context.
    """


     # --- config ---
    if isinstance(config, dict):
        config = BaseConfig(**config)
    elif not isinstance(config, BaseConfig):
        raise TypeError(f"Expected dict or BaseConfig, got {type(config)}")

    logger.info("=" * 60)
    logger.info("BUILDING EVALUATION CONTEXT")
    logger.info("=" * 60)

    # --- device ---
    cuda_info = get_cuda_info(config)
    if device == "cuda" and not cuda_info:
        logger.warning("CUDA not available, switching to CPU")
        device = "cpu"

    
    # --- dataset / dataloaders ---
    logger.info("Loading dataset...")
    dataloaders_dict = dataset.return_dataloaders_dict()

    test_keys, test_loaders = dataset.flatten_loaders(dataloaders_dict["test"])

    

