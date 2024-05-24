from typing import Iterable

import torch
import torch.nn as nn


class AddPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def reset_parameters(self):
        nn.init.normal_(self.pe.weight, 0, 0.02)

    def forward(self, x):
        """
        Arguments:
                x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FourierFeatures(nn.Module):
    def __init__(self, output_size, learnable=True):
        super().__init__()

        self.output_size = output_size
        self.learnable = learnable

        if learnable:
            # we'll just assume this will always be used for denoising iteration k of size [B, 1]
            self.kernel = nn.Parameter(torch.randn(output_size // 2, 1))
        else:
            half_dim = output_size // 2
            f = torch.log(10000) / (half_dim - 1)
            f = torch.exp(torch.arange(half_dim) * -f)
            self.register_buffer("f", f)

    def forward(self, x):
        f = 2 * torch.pi * x @ self.kernel.t() if self.learnable else self.f * x
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims: Iterable[int]):
        super().__init__()
        layers = []
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, dim))
            if i + 1 < len(hidden_dims):
                layers.append(nn.SiLU())
            input_dim = dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPResNetBlock(nn.Module):
    def __init__(self, in_dim, dropout=0, use_layer_norm=True):
        super().__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if use_layer_norm:
            layers.append(nn.LayerNorm(in_dim))
        layers += [nn.Linear(in_dim, in_dim * 4), nn.SiLU(), nn.Linear(in_dim * 4, in_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class MLPResNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim)]
        for _ in range(num_layers):
            layers.append(MLPResNetBlock(hidden_dim))
        layers += [nn.SiLU(), nn.Linear(hidden_dim, out_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
