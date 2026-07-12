import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments for training/evaluation script."""
    parser = argparse.ArgumentParser(
        description="Train or evaluate SBI model",
    )

    parser.add_argument(
        '--config-dir',
        type=str,
        default='experiments',
        help='Directory containing config YAML file (default: experiments)',
    )

    parser.add_argument(
        '--action',
        type=str,
        choices=['train', 'evaluate'],
        default='train',
        help='Action to perform: train or evaluate',
    )

    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='Path to checkpoint file for resuming or evaluation',
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Resume training from checkpoint',
    )

    return parser.parse_args()


def get_config_path(config_dir: str) -> Path:
    """Get absolute path to config YAML file."""
    config_path = Path(config_dir)

    # If it's a directory, look for config.yaml inside
    if config_path.is_dir():
        config_path = config_path / 'config.yaml'

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}"
        )

    return config_path.resolve()
