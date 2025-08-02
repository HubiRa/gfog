import torch
from torch import nn

from typing import List
from torch.nn.utils import spectral_norm
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: nn.Module = nn.GELU(),
        use_layernorm: bool = False,
        use_output_layernorm: bool = False,
        zero_one_normalization: bool = False,
        output_activation: nn.Module | None = None,
        unit_vec_normalize: bool = False,
        use_spectral_norm: bool = False,
        disable_bias: bool = False,
    ) -> None:
        super().__init__()

        self.activation = activation
        self.output_activation = output_activation
        self.zero_one_normalization = zero_one_normalization
        self.unit_vec_normalize = unit_vec_normalize

        # choose spectral norm wrapper
        sn = spectral_norm if use_spectral_norm else (lambda x: x)

        # build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList(
            [
                sn(nn.Linear(dims[i], dims[i + 1], bias=(not disable_bias)))
                for i in range(len(dims) - 1)
            ]
        )

        # add optional LayerNorm per hidden layer
        self.layernorms = (
            nn.ModuleList([nn.LayerNorm(h) for h in hidden_dims])
            if use_layernorm
            else None
        )

        # output layernorm
        self.out_layernorm = nn.LayerNorm(output_dim) if use_output_layernorm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # apply activation + layernorm except after final layer
            if i < len(self.layers) - 1:
                if self.layernorms is not None:
                    x = self.layernorms[i](x)
                x = self.activation(x)

        # output activation
        if self.output_activation is not None:
            x = self.output_activation(x)

        # optional output layernorm
        if self.out_layernorm is not None:
            x = self.out_layernorm(x)

        # optional normalization modes
        if self.zero_one_normalization:
            min_vals = x.min(dim=1, keepdim=True).values
            max_vals = x.max(dim=1, keepdim=True).values
            x = (x - min_vals) / (max_vals - min_vals + 1e-5)
        elif self.unit_vec_normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x
