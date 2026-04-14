## architecture HST
import torch.nn as nn
from lampe.inference import NPE, NPELoss
from lampe.nn import ResMLP
from lampe.nn.flows import NAF

import torch
from torch import Tensor


class SoftClip(nn.Module):
    def __init__(self, bound: float = 1.0):
        super().__init__()

        self.bound = bound

    def forward(self, x: Tensor) -> Tensor:
        return x / (1 + abs(x / self.bound))