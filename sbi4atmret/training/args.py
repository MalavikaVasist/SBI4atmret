import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments for training script.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - config_dir: Path to directory containing config.yaml
    """
    parser = argparse.ArgumentParser(
        description="Train SBI model with Pydantic config validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_general_new.py --config-dir experiments/
  python train_general_new.py -c /path/to/config/dir/
        """
    )
    
    parser.add_argument(
        '--config-dir',
        '-c',
        type=str,
        default='experiments',
        help='Directory containing config.yaml file (default: experiments)',
    )

    parser.add_argument(
        '--action',
        '-c',
        type=str,
        default='experiments',
        help='',
    )
    
    parser.add_argument(
    '--checkpoint-path',
    '-c',
    type=str,
    default='experiments/model1',
    help='config_model1.yaml file (default: config_model1.yaml)',
    )

    parser.add_argument(
    '--resume',
    '-c',
    type=str,
    default='False',
    help='',
    )


    return parser.parse_args()


def get_config_path(config_dir: str) -> Path:
    """Get absolute path to config.yaml file.
    
    Args:
        config_dir: Directory containing config.yaml
        
    Returns:
        Path: Absolute path to config.yaml
        
    Raises:
        FileNotFoundError: If config.yaml does not exist
    """
    config_path = Path(config_dir) / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            f"Please provide a valid config directory with -c/--config-dir"
        )
    
    return config_path.resolve()
