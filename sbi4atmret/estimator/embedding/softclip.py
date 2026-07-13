import torch
import torch.nn as nn
from torch import Tensor
from lampe.nn import ResMLP


class SoftClip(nn.Module):
    def __init__(self, bound: float = 1.0):
        super().__init__()
        self.bound = bound

    def forward(self, x: Tensor) -> Tensor:
        return x / (1 + abs(x / self.bound))


class SoftclipResMLP(nn.Module):
    def __init__(self, config):
        """
        Multi-instrument embedding with SoftClip + ResMLP per instrument.

        Args:
            config: ComponentConfig with kwargs.bound and kwargs.instruments
        """
        super().__init__()

        bound = config.kwargs["bound"]
        instruments = config.kwargs["instruments"]

        base_sizes = torch.tensor([512, 256, 128])
        self.embeddings = nn.ModuleDict()

        for inst_name, inst_cfg in instruments.items():
            hf = inst_cfg["hidden_features"]
            input_dim = inst_cfg["input_dim"]
            output_dim = inst_cfg["output_dim"]

            hidden_layers = torch.hstack([
                base_sizes[i].repeat(hf[i])
                for i in range(len(hf))
            ])

            self.embeddings[inst_name] = nn.Sequential(
                SoftClip(bound),
                ResMLP(
                    input_dim,
                    output_dim,
                    hidden_features=hidden_layers.tolist(),
                    activation=nn.ELU,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass. Splits concatenated input by instrument dimensions.
        """
        if isinstance(x, dict):
            return torch.cat(
                [self.embeddings[inst](x[inst]) for inst in sorted(self.embeddings.keys())],
                dim=-1,
            )

        # Single tensor: split by instrument input dims
        parts = []
        offset = 0
        for inst in sorted(self.embeddings.keys()):
            resmlp = self.embeddings[inst][1]  # ResMLP is second in Sequential (after SoftClip)
            dim = resmlp.in_features
            parts.append(self.embeddings[inst](x[:, offset:offset + dim]))
            offset += dim

        return torch.cat(parts, dim=-1)
