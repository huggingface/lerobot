import torch
from torch import Tensor, nn

from .interaction_bottleneck import InteractionGeometryBottleneck


class InteractionGeometryHead(nn.Module):
    def __init__(self, memory_dim, state_dim, hidden_dim=512, num_heads=8, num_layers=2):
        super().__init__()
        self.memory_projection = nn.Linear(memory_dim, hidden_dim)
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(5, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.translation_head = nn.Linear(hidden_dim, 3)
        self.direction_head = nn.Linear(hidden_dim, 3)
        self.magnitude_head = nn.Linear(hidden_dim, 1)
        self.rotation_head = nn.Linear(hidden_dim, 3)
        self.gripper_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, state: Tensor):
        hidden_states = hidden_states.to(self.memory_projection.weight.dtype)
        state = state.to(self.state_projection.weight.dtype)
        memory = self.memory_projection(hidden_states)
        queries = self.queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        queries = queries + self.state_projection(state).unsqueeze(1)
        grounded = self.decoder(queries, memory, memory_key_padding_mask=~attention_mask.bool())
        translation_goal = self.translation_head(grounded[:, 0])
        translation_magnitude = torch.nn.functional.softplus(self.magnitude_head(grounded[:, 2]))
        return InteractionGeometryBottleneck(
            translation_goal=translation_goal,
            approach_direction=torch.nn.functional.normalize(self.direction_head(grounded[:, 1]), dim=-1),
            translation_magnitude=translation_magnitude,
            rotation_goal=self.rotation_head(grounded[:, 3]),
            gripper_transition=self.gripper_head(grounded[:, 4]),
            # No confidence target exists in LIBERO-Safety. Keep the field neutral rather
            # than feeding an untrained random head into Stage B.
            confidence_logit=torch.zeros_like(translation_magnitude),
            valid_mask=translation_magnitude > 1e-5,
        )
