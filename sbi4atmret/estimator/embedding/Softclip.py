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


class SoftclipResMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        base_sizes = torch.tensor([512, 256, 128])
        self.embeddings = nn.ModuleDict()

        # use the passed config
        for inst, inst_cfg in config.kwargs.instruments.items():

            hidden_layers = torch.hstack([
                base_sizes[i].repeat(inst_cfg.hidden_features[i])
                for i in range(3)
            ])

            self.embeddings[inst] = nn.Sequential(
                SoftClip(config.kwargs.limit),
                ResMLP(
                    input_dim=inst_cfg.input_dim,
                    output_dim=inst_cfg.output_dim,
                    hidden_features=hidden_layers.tolist(),
                    activation=nn.ELU,
                ),
            )

    
    def forward(self, x_dict: dict) -> Tensor:
        return torch.cat(
            [self.embeddings[inst](x_dict[inst]) for inst in self.embeddings],
            dim=-1
        )
