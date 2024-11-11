import torch
import torch.nn as nn


class IdentityProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.placeholder_param = nn.Parameter(torch.zeros(1))

    def forward(self, *args):
        return args[0]

    def configure_optimizers(self, weight_decay, lr, betas):
        return torch.optim.AdamW(
            self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
