
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


def _to_device(self, batch):
    if isinstance(batch, (list, tuple)):
        return [b.to(self.device) for b in batch]
    if isinstance(batch, dict):
        return {k: v.to(self.device) for k, v in batch.items()}
    return batch.to(self.device)
