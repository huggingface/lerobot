import torch
from torch import Tensor, nn

from .interaction_bottleneck import InteractionGeometryBottleneck


class FlowMatchingController(nn.Module):
    """Stage B boundary: accepts only explicit geometry, state, noisy action and time."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 8,
        bottleneck_mode: str = "interaction_tuple",
    ):
        super().__init__()
        self.bottleneck_mode = bottleneck_mode
        self.condition = nn.Linear(13 + state_dim, hidden_dim)
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.time_in = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.action_out = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        bottleneck: InteractionGeometryBottleneck,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        geometry = bottleneck.as_controller_tensor(self.bottleneck_mode)
        condition = self.condition(torch.cat([geometry, state], -1)).unsqueeze(1)
        action_tokens = self.action_in(noisy_actions) + self.time_in(timestep[:, None]).unsqueeze(1)
        return self.action_out(self.transformer(torch.cat([condition, action_tokens], 1))[:, 1:])
