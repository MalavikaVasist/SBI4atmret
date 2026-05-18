import torch
from typing import Union

from ..config.configs import BaseConfig
from estimator.base import EstimatorBase

class BaseModel:
    """
    Base class for all models providing common setup and utility methods.
    """

    def __init__(self, config):
        self.config = config 

    
    def build(self):
        # --- Build components ---
        self.embedding = self.config.build_embedding()
        self.flow = self.config.build_flow()

        # --- Compose estimator ---
        self.estimator = EstimatorBase(self.flow, self.embedding)

        return self
    




