from collections import deque
from typing import Unpack

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_mlp_bc import MLPBCConfig


class MLPBCPolicy(PreTrainedPolicy):
    """Feed-forward state -> action regressor (L1 loss)."""

    config_class = MLPBCConfig
    name = "mlp_bc"

    def __init__(
        self,
        config: MLPBCConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        dataset_meta=None,
    ):
        super().__init__(config)
        config.validate_features()

        state_dim = config.robot_state_feature.shape[0]
        action_dim = config.action_feature.shape[0]

        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            if config.use_layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

        self.reset()

    def get_optim_params(self) -> list[Tensor]:
        return [p for p in self.parameters() if p.requires_grad]

    def reset(self) -> None:
        # Single-action policy, but we keep a queue for API parity with chunking policies.
        self._action_queue: deque[Tensor] = deque([], maxlen=1)

    def _predict(self, state: Tensor) -> Tensor:
        return self.net(state)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        state = batch[OBS_STATE]
        target = batch[ACTION]
        # When the dataset slices a single time step the action arrives as [B, action_dim]
        # but other lerobot machinery sometimes pads to [B, 1, action_dim] — collapse it.
        if target.dim() == 3 and target.shape[1] == 1:
            target = target.squeeze(1)
        pred = self._predict(state)
        loss = F.l1_loss(pred, target)
        return loss, {}

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        action = self._predict(batch[OBS_STATE])
        return action.unsqueeze(1)  # [B, 1, action_dim] for API parity

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            for t in range(chunk.shape[1]):
                self._action_queue.append(chunk[:, t])
        return self._action_queue.popleft()
