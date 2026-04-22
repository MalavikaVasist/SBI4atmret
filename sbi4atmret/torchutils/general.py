
from typing import Any

import torch


def _resolve_device(self):
    import torch
    return torch.device(
        self.config.training_config.device
        if hasattr(self.config.training_config, "device")
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
