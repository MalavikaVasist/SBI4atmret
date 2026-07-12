'''
initialise the npe flow model with the config files
retruns the npe model initialized with the config files.
'''

from lampe.inference import NPE
from sbi4atmret.estimator.base import EstimatorBase
import torch
import torch.nn as nn
from torch import Tensor


class NPEFlow(nn.Module):
    def __init__(self, FlowConfig, PriorConfig, EmbeddingConfig):
        super().__init__()

        l, u = PriorConfig.get_parameter_bounds()
        l, u = torch.tensor(l, dtype=torch.float32), torch.tensor(u, dtype=torch.float32)
        no_of_params = PriorConfig.get_no_of_params()

        flow_input_dim = sum(
            inst_cfg["output_dim"]
            for inst_cfg in EmbeddingConfig.kwargs["instruments"].values()
        )

        npe_kwargs = dict(
            moments=((l + u) / 2, (u - l) / 2),
            transforms=FlowConfig.kwargs["transforms"],
            signal=FlowConfig.kwargs["signal"],
            hidden_features=FlowConfig.kwargs["hidden_features"] * FlowConfig.kwargs["hidden_features_no"],
            activation="ELU",
        )

        if "build" in FlowConfig.kwargs and FlowConfig.kwargs["build"] is not None:
            build_name = FlowConfig.kwargs["build"]
            if build_name == "NAF":
                from lampe.nn.flows import NAF
                npe_kwargs["build"] = NAF
            else:
                npe_kwargs["build"] = build_name  # assume callable

        self._flow = NPE(
            no_of_params,
            flow_input_dim,
            **npe_kwargs,
        )

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        return self._flow(theta, x)
    
    def flow(self, x: Tensor):
        """Return the posterior distribution conditioned on x."""
        return self._flow.flow(x)


