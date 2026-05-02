import torch
from torch import Tensor

class BaseNoise:
    def __init__(self, config):
        self.config = config
        self.applynoise = self._apply_noise()

    def __call__(self, batch_dict):
        """
        batches = [(theta1, x1), (theta2, x2), ...]
        """
        return self.forward(batch_dict)

    def forward(self, batch_dict):
        raise NotImplementedError

