import torch
import torch.nn as nn
from copy import deepcopy


class EMA(nn.Module):
    def __init__(self, src_model: nn.Module, beta: float, copy: bool = True):
        super().__init__()
        if copy:
            self.model = deepcopy(src_model)
        else:
            self.model = src_model
        self.model.eval()
        self.model.requires_grad_(False)
        self.beta = beta

    def step(self, src_model):
        one_minus_beta = 1.0 - self.beta
        for ema_param, src_param in zip(
                self.model.parameters(), src_model.parameters()
        ):
            # ema_param = ema_param * beta + src_param * (1 - beta)
            ema_param.data.mul_(self.beta).add_(src_param.data, alpha=one_minus_beta)
            ema_param.requires_grad_(False)

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.model(*args, **kwargs)
