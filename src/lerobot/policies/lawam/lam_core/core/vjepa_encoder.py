from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import AutoModel


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 encoder used by the released LaWAM latent action model."""

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_latent_layers: int = 1,
        norm_layer_type: str = "l2",
        enable_norm: bool = False,
    ):
        super().__init__()
        self.model_id = model_id
        self.num_latent_layers = max(int(num_latent_layers), 1)
        self.norm_layer_type = norm_layer_type
        self.enable_norm = enable_norm
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.float32)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        hidden_size = getattr(self.model.config, "hidden_size", None)
        self.feature_dim = int(hidden_size) if hidden_size is not None else 1024
        if self.norm_layer_type == "bn":
            self.latent_norms = nn.ModuleList(
                [nn.SyncBatchNorm(self.feature_dim, affine=False) for _ in range(self.num_latent_layers)]
            )
        elif self.norm_layer_type == "ln":
            self.latent_norms = nn.ModuleList(
                [
                    nn.LayerNorm(self.feature_dim, elementwise_affine=False)
                    for _ in range(self.num_latent_layers)
                ]
            )
        else:
            self.latent_norms = None

    def train(self, mode: bool = True):
        del mode
        super().train(False)
        self.model.eval()
        return self

    def _normalize(self, tokens: torch.Tensor, layer_index: int) -> torch.Tensor:
        if not self.enable_norm:
            return tokens
        if self.norm_layer_type == "bn":
            if self.latent_norms is None:
                raise ValueError("DINOv3Encoder has no batch normalization layers.")
            tokens_2d = tokens.reshape(-1, self.feature_dim)
            return self.latent_norms[layer_index](tokens_2d).view_as(tokens)
        if self.norm_layer_type == "ln":
            if self.latent_norms is None:
                raise ValueError("DINOv3Encoder has no layer normalization layers.")
            return self.latent_norms[layer_index](tokens)
        if self.norm_layer_type == "l2":
            return functional.normalize(tokens, p=2, dim=-1)
        raise ValueError(f"Unsupported DINOv3 normalization mode: {self.norm_layer_type}")

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        remove_cls: bool = True,
        n: int | Sequence[int] = -1,
    ) -> torch.Tensor | list[torch.Tensor]:
        if images.dim() != 4:
            batch_size, num_frames, channels, height, width = images.shape
            images = images.reshape(-1, channels, height, width)
        else:
            batch_size, channels, height, width = images.shape
            num_frames = 1

        need_hidden_states = not (isinstance(n, int) and n == -1)
        outputs = self.model(pixel_values=images, output_hidden_states=need_hidden_states)
        if not need_hidden_states:
            tokens = outputs.last_hidden_state
            if remove_cls:
                tokens = tokens[:, 5:, :]
            tokens = self._normalize(tokens, layer_index=0)
            return tokens.reshape(batch_size, num_frames, -1, self.feature_dim).detach()

        layer_indices = [n] if isinstance(n, int) else list(n)
        if len(layer_indices) > self.num_latent_layers:
            raise ValueError(
                f"DINOv3Encoder has {self.num_latent_layers} normalization layers, "
                f"but {len(layer_indices)} feature layers were requested."
            )

        features = []
        for layer_index, hidden_state_index in enumerate(layer_indices):
            tokens = outputs.hidden_states[hidden_state_index]
            if remove_cls:
                tokens = tokens[:, 5:, :]
            tokens = self._normalize(tokens, layer_index=layer_index)
            features.append(tokens.reshape(batch_size, num_frames, -1, self.feature_dim).detach())
        return features[0] if isinstance(n, int) else features


def build_vision_encoder(
    model_id: str,
    num_latent_layers: int = 1,
    norm_layer_type: str = "l2",
    enable_norm: bool = False,
) -> tuple[DINOv3Encoder, int]:
    if "dinov3" not in str(model_id).lower():
        raise ValueError(
            "The LeRobot LaWAM adapter supports the released DINOv3 LAM checkpoint only; "
            f"got vision_model_id={model_id!r}."
        )
    encoder = DINOv3Encoder(
        model_id=model_id,
        num_latent_layers=num_latent_layers,
        norm_layer_type=norm_layer_type,
        enable_norm=enable_norm,
    )
    return encoder, encoder.feature_dim
