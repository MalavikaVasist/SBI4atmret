"""
Transformer-based spectral embedding.

Treats the spectrum as a sequence of wavelength tokens and applies
self-attention to learn inter-wavelength relationships before
compressing to a fixed-size embedding vector.

Config example:
    embedding:
      type: estimator.embedding.transformer.TransformerEmbedding
      kwargs:
        instruments:
          miri:
            input_dim: 1298
            output_dim: 64
          gemini:
            input_dim: 434
            output_dim: 16
        d_model: 64
        nhead: 4
        num_layers: 3
        dim_feedforward: 256
        dropout: 0.1
        patch_size: 8
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict


class PatchEmbedding(nn.Module):
    """
    Split a 1-D spectrum into non-overlapping patches and project each
    to d_model dimensions. Acts as the tokenizer for the transformer.
    """

    def __init__(self, input_dim: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = math.ceil(input_dim / patch_size)
        # Pad input to be divisible by patch_size
        self.padded_dim = self.n_patches * patch_size
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, D) spectrum

        Returns:
            (B, n_patches, d_model) patch tokens
        """
        B = x.shape[0]
        # Pad if needed
        if x.shape[1] < self.padded_dim:
            x = nn.functional.pad(x, (0, self.padded_dim - x.shape[1]))
        # Reshape to patches: (B, n_patches, patch_size)
        x = x.view(B, self.n_patches, self.patch_size)
        # Project: (B, n_patches, d_model)
        return self.proj(x)


class TransformerEmbedding(nn.Module):
    """
    Multi-instrument transformer embedding.

    For each instrument:
    1. Split spectrum into patches (tokenize)
    2. Add learnable positional encoding
    3. Apply transformer encoder layers (self-attention)
    4. Pool to fixed output_dim via a learned [CLS] token or mean pooling
    5. Project to output_dim

    The outputs from all instruments are concatenated.

    Args:
        config: ComponentConfig with kwargs containing:
            - instruments: dict of {name: {input_dim, output_dim}}
            - d_model: transformer hidden dimension (default: 64)
            - nhead: number of attention heads (default: 4)
            - num_layers: number of encoder layers (default: 3)
            - dim_feedforward: feedforward dimension (default: 256)
            - dropout: dropout rate (default: 0.1)
            - patch_size: spectrum patch size for tokenization (default: 8)
            - pooling: "cls" or "mean" (default: "mean")
    """

    def __init__(self, config):
        super().__init__()

        instruments = config.kwargs["instruments"]
        d_model = config.kwargs.get("d_model", 64)
        nhead = config.kwargs.get("nhead", 4)
        num_layers = config.kwargs.get("num_layers", 3)
        dim_feedforward = config.kwargs.get("dim_feedforward", 256)
        dropout = config.kwargs.get("dropout", 0.1)
        patch_size = config.kwargs.get("patch_size", 8)
        pooling = config.kwargs.get("pooling", "mean")

        self.pooling = pooling
        self.embeddings = nn.ModuleDict()

        for inst_name, inst_cfg in instruments.items():
            input_dim = inst_cfg["input_dim"]
            output_dim = inst_cfg["output_dim"]

            self.embeddings[inst_name] = _InstrumentTransformer(
                input_dim=input_dim,
                output_dim=output_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                patch_size=patch_size,
                pooling=pooling,
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass. Splits concatenated input by instrument dimensions.

        Args:
            x: (B, D_total) concatenated spectrum across instruments
               OR dict {inst_name: (B, D_inst)} per-instrument tensors

        Returns:
            (B, sum_of_output_dims) concatenated embeddings
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
            dim = self.embeddings[inst].input_dim
            parts.append(self.embeddings[inst](x[:, offset:offset + dim]))
            offset += dim

        return torch.cat(parts, dim=-1)


class _InstrumentTransformer(nn.Module):
    """
    Transformer encoder for a single instrument's spectrum.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        patch_size: int,
        pooling: str,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.pooling = pooling

        # Patch tokenization
        self.patch_embed = PatchEmbedding(input_dim, patch_size, d_model)
        n_patches = self.patch_embed.n_patches

        # Positional encoding (learnable)
        n_tokens = n_patches + (1 if pooling == "cls" else 0)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        # CLS token (if using cls pooling)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, input_dim) single instrument spectrum

        Returns:
            (B, output_dim) embedding
        """
        # Tokenize: (B, n_patches, d_model)
        tokens = self.patch_embed(x)

        # Prepend CLS token if using cls pooling
        if self.pooling == "cls":
            B = tokens.shape[0]
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        # Add positional encoding
        tokens = tokens + self.pos_encoding

        # Self-attention: (B, n_tokens, d_model)
        tokens = self.transformer(tokens)

        # Pool to single vector
        if self.pooling == "cls":
            pooled = tokens[:, 0]  # CLS token output
        else:
            pooled = tokens.mean(dim=1)  # mean over patches

        # Project to output_dim: (B, output_dim)
        return self.output_proj(pooled)
