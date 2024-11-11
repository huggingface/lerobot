import torch
import torch.nn as nn
from ..transformer_encoder import TransformerEncoderConfig, TransformerEncoder


class InverseDynamicsProjector(nn.Module):
    def __init__(
        self,
        window_size: int,
        input_dim: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.config = TransformerEncoderConfig(
            block_size=window_size,
            input_dim=input_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.model = TransformerEncoder(self.config)

    def forward(self, obs_enc: torch.Tensor):
        N, T, V = obs_enc.shape[:3]
        obs_proj = []
        # TODO: vectorize this for-statement
        for i in range(V):
            this_view_proj = self.model(obs_enc[:, :, i])  # (N, T, Z)
            obs_proj.append(this_view_proj)
        return torch.stack(obs_proj, dim=2)  # (N, T, V, Z)

    def configure_optimizers(self, weight_decay, lr, betas):
        return self.model.configure_optimizers(weight_decay, lr, betas)
