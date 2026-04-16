"""
Method to build (i.e., instantiate) a model from a checkpoint file or
from a configuration dictionary.
"""

from pathlib import Path
from typing import Any

import torch

from ..MLmodel.flows.fmpe import FMPEModel
from ..MLmodel.flows.npe import NPEModel


def build_estimator(
    experiment_dir: Path | None = None,
    file_path: Path | None = None,
    config: dict | None = None,
    **kwargs: Any,
) -> FMPEModel | NPEModel:
    """
    Build a model from a checkpoint file or from a `config` dictionary.

    Args:
        experiment_dir: Path to the experiment directory.
        file_path: Path to a checkpoint file (`*.pt`).
        config: Dictionary with full experiment configuration.
        **kwargs: Extra keyword arguments to pass to the model class.

    Returns:
        Model instance.
    """

    # Get the model type, either from the checkpoint file or from the settings
    if file_path is not None:
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        model_type = checkpoint["config"]["model"]["model_type"].lower()
        random_seed = checkpoint["config"]["model"]["random_seed"]
    elif config is not None:
        if "model" in config and isinstance(config["model"], dict) and "model_type" in config["model"]:
            model_type = config["model"]["model_type"].lower()
            random_seed = config["model"].get("random_seed", config.get("random_seed", 0))
        elif "ML_model_configs" in config:
            estimator_cfg = config["ML_model_configs"].get("estimator", {})
            flow_cfg = estimator_cfg.get("flow") if isinstance(estimator_cfg, dict) else None
            if flow_cfg and isinstance(flow_cfg, dict) and "type" in flow_cfg:
                model_type = flow_cfg["type"].lower()
                random_seed = config.get("random_seed", 0)
            else:
                raise ValueError("Config does not contain a recognizable model type for estimator building")
        else:
            raise ValueError("Either `file_path` or `config` must be provided!")

    # Select the model class
    match model_type:
        case "fmpe":
            return FMPEModel(
                experiment_dir=experiment_dir,
                file_path=file_path,
                config=config,
                random_seed=random_seed,
                **kwargs,
            )
        case "npe":
            return NPEModel(
                experiment_dir=experiment_dir,
                file_path=file_path,
                config=config,
                random_seed=random_seed,
                **kwargs,
            )
        case _:
            raise ValueError(f"{model_type} is not a valid model type!")