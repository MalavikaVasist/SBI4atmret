
from typing import Any
from config.configs import BaseConfig

import torch


def get_cuda_info(config: dict):
    return torch.device(
        config.training_config.device
        if hasattr(config.training_config, "device")
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def to_device(module, device):
    if module is None:
        return None

    return module.to(device)