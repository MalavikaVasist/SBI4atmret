'''
initialise the npe flow model with the config files
retruns the npe model initialized with the config files.
'''

from lampe.inference import NPE
import torch.nn as nn
from torch import Tensor


class NPEFlow(nn.Module):
    def __init__(self, FlowConfig, PriorConfig, EmbeddingConfig):
        super().__init__()

        l, u = PriorConfig.get_parameter_bounds()
        no_of_params = PriorConfig.get_no_of_params()

        flow_input_dim = sum(
            inst_cfg.output_dim
            for inst_cfg in EmbeddingConfig.kwargs.instruments.values()
        )

        self.flow = NPE(
            no_of_params,
            flow_input_dim,
            moments=((l + u) / 2, (u - l) / 2),
            transforms=FlowConfig.kwargs.transforms,
            build=FlowConfig.kwargs.build if hasattr(FlowConfig.kwargs, "build") else None,
            signal=FlowConfig.kwargs.signal,
            hidden_features=FlowConfig.kwargs.hidden_features * FlowConfig.kwargs.hidden_features_no,
            activation="ELU",
        )

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        return self.flow(theta, x)
    
    def flow(self, x: Tensor):  # -> Distribution):
        return self.flow(x)

