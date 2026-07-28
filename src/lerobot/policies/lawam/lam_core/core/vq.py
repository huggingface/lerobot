import torch
import torch.nn as nn
from torch import Tensor


class VAEQuantizer(nn.Module):
    """Continuous VAE bottleneck used by the released LaWAM LAM checkpoint."""

    def __init__(
        self,
        code_dim: int = 64,
        beta: float = 5e-5,
        clamp_logvar: float | None = 10.0,
        layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.code_dim = int(code_dim)
        self.beta = float(beta)
        self.clamp_logvar = float(clamp_logvar) if clamp_logvar is not None else None
        self.pre_norm = nn.LayerNorm(self.code_dim) if layer_norm else nn.Identity()
        self.mu = nn.Linear(self.code_dim, self.code_dim)
        self.logvar = nn.Linear(self.code_dim, self.code_dim)

    def _encode(self, nodes: Tensor) -> tuple[Tensor, Tensor]:
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(1)
        pooled_nodes = nodes.mean(dim=1, keepdim=True)
        hidden = self.pre_norm(pooled_nodes)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        if self.clamp_logvar is not None:
            logvar = torch.clamp(logvar, min=-self.clamp_logvar, max=self.clamp_logvar)
        return mu, logvar

    def forward(self, nodes: Tensor) -> tuple[Tensor, Tensor, None, Tensor, Tensor]:
        mu, logvar = self._encode(nodes)
        std = (0.5 * logvar).exp()
        quantized = mu + torch.randn_like(std) * std
        kl_loss = (0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)).mean()
        zero = nodes.new_tensor(0.0)
        return quantized, zero, None, zero, kl_loss * self.beta

    @torch.no_grad()
    def inference(self, nodes: Tensor, user_specific=None) -> tuple[Tensor, None]:
        del user_specific
        mu, _ = self._encode(nodes)
        return mu, None
