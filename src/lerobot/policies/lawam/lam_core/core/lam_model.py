from typing import Any

import torch
import torch.nn as nn
import yaml

from .utils.lam_decoder import LAMDecoderV2, StatePredictor
from .utils.lam_encoder import LAMEncoder
from .vjepa_encoder import build_vision_encoder
from .vq import VAEQuantizer

LAM_IMAGE_HW = (256, 256)
LAM_PATCH_SIZE = 16


class LatentLAMModel(nn.Module):
    """Released DINOv3/VAE latent action model used by LaWAM."""

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        ffn_expansion_factor: int = 2,
        enc_layers: int = 6,
        codebook_size: int = 16,
        code_dim: int = 256,
        max_state_dim: int = 32,
        num_frames: int = 5,
        num_queries: int = 1,
        vq_kwargs: dict[str, Any] | None = None,
        dec_layers: int = 6,
        dropout: float = 0.1,
        vq_type: str = "vae",
        disable_vq: bool = False,
        norm_latents: bool = False,
        norm_latents_type: str = "l2",
        vision_model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        enc_add_state: bool = False,
        enc_modal_mask: bool = False,
        latent_layer_to_use: int = -2,
        multi_input: bool = False,
        num_embodiments: int = 32,
        image_hw: tuple[int, int] = LAM_IMAGE_HW,
        patch_size: int = LAM_PATCH_SIZE,
        decoder_last_ln: bool = True,
        **kwargs,
    ):
        super().__init__()
        del codebook_size, kwargs
        if vq_type not in {"vae", "beta_vae"}:
            raise ValueError(f"The released LaWAM LAM requires vq_type='vae', got {vq_type!r}.")
        if disable_vq:
            raise ValueError("The released LaWAM LAM does not support disable_vq=true.")
        if enc_add_state:
            raise ValueError("The released LaWAM LAM requires enc_add_state=false.")
        if multi_input:
            raise ValueError("The released LaWAM LAM requires multi_input=false.")
        if not isinstance(latent_layer_to_use, int):
            raise ValueError("The released LaWAM LAM requires one integer latent_layer_to_use.")

        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.patch_size = int(patch_size)
        if self.patch_size != LAM_PATCH_SIZE:
            raise ValueError(f"Unsupported patch_size={self.patch_size}. Only {LAM_PATCH_SIZE} is supported.")
        if self.image_hw[0] % self.patch_size != 0 or self.image_hw[1] % self.patch_size != 0:
            raise ValueError(f"image_hw={self.image_hw} must be divisible by patch_size={self.patch_size}.")

        self.grid_height = self.image_hw[0] // self.patch_size
        self.grid_width = self.image_hw[1] // self.patch_size
        self.feature_dim = dim
        self.code_dim = code_dim
        self.latent_layer_to_use = latent_layer_to_use
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.num_embodiments = int(num_embodiments)

        self.vision_encoder, self.input_dim = build_vision_encoder(
            vision_model_id,
            norm_layer_type=norm_latents_type,
            enable_norm=norm_latents,
        )
        self.decoder = LAMDecoderV2(
            context_dim=dim,
            input_dim=self.input_dim,
            num_queries=num_queries,
            num_layers=dec_layers,
            num_heads=num_heads,
            dropout=dropout,
            train_in_latent=True,
            ffn_expansion_factor=ffn_expansion_factor,
            num_embodiments=self.num_embodiments,
            code_dim=code_dim,
            grid_hw=(self.grid_height, self.grid_width),
            last_ln=decoder_last_ln,
        )
        self.state_decoder = StatePredictor(
            latent_dim=dim,
            dropout=dropout,
            num_embodiments=self.num_embodiments,
            num_queries=num_queries,
            max_state_dim=max_state_dim,
            code_dim=code_dim,
        )
        self.encoder = LAMEncoder(
            context_dim=dim,
            input_dim=self.input_dim,
            add_state=False,
            modal_mask=enc_modal_mask,
            num_layers=enc_layers,
            num_heads=num_heads,
            dropout=dropout,
            ffn_expansion_factor=ffn_expansion_factor,
            num_frames=self.num_frames,
            grid_hw=(self.grid_height, self.grid_width),
            num_queries=num_queries,
            max_state_dim=max_state_dim,
            num_embodiments=self.num_embodiments,
            code_dim=code_dim,
        )
        self.vq = VAEQuantizer(code_dim=code_dim, **(vq_kwargs or {}))

    @torch.inference_mode()
    def get_latent_action(
        self,
        videos: torch.Tensor,
        states: torch.Tensor | None,
        dec_videos: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        predict_future_frame: bool = False,
        user_specific=None,
        embodiment_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del states, dec_videos, state_mask, predict_future_frame, user_specific
        features = self.vision_encoder.encode(videos, n=self.latent_layer_to_use)
        if not isinstance(features, torch.Tensor):
            raise TypeError("The released LaWAM LAM expects one DINOv3 feature tensor.")
        nodes = self.encoder(features, embodiment_id=embodiment_ids)
        autocast_device = "cuda" if nodes.is_cuda else "cpu"
        with torch.amp.autocast(device_type=autocast_device, enabled=False):
            quantized, _ = self.vq.inference(nodes.float())
        return {"quantized": quantized}

    @torch.no_grad()
    def extract_vision_features(self, videos: torch.Tensor, *, n: int = -2) -> torch.Tensor:
        features = self.vision_encoder.encode(videos, n=n)
        if not isinstance(features, torch.Tensor):
            raise TypeError("The released LaWAM LAM expects one DINOv3 feature tensor.")
        return features


def load_latent_action_model(ckpt_path, yaml_path):
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", cfg) or {}

    if "image_hw" not in model_cfg:
        raise ValueError("LAM config must provide `model.image_hw`.")
    if "patch_size" not in model_cfg:
        raise ValueError("LAM config must provide `model.patch_size`.")

    init_kwargs = dict(model_cfg)
    init_kwargs.pop("ar_prediction", None)
    latent_action_model = LatentLAMModel(**init_kwargs).to("cpu")

    try:
        lam_payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        lam_payload = torch.load(ckpt_path, map_location="cpu")  # nosec B614
    lam_ckpt = lam_payload["state_dict"]
    model_state = latent_action_model.state_dict()
    new_ckpt = {}
    for key, value in lam_ckpt.items():
        renamed = key.replace("lam.", "")
        if renamed.startswith("vision_encoder.model.layer."):
            candidate = renamed.replace(
                "vision_encoder.model.layer.",
                "vision_encoder.model.model.layer.",
                1,
            )
            if candidate in model_state:
                renamed = candidate
        new_ckpt[renamed] = value

    model_keys = set(model_state)
    checkpoint_keys = set(new_ckpt)
    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)
    shape_mismatches = [
        (key, tuple(model_state[key].shape), tuple(new_ckpt[key].shape))
        for key in sorted(model_keys & checkpoint_keys)
        if model_state[key].shape != new_ckpt[key].shape
    ]
    if missing_keys or unexpected_keys or shape_mismatches:
        error_lines = ["Failed to load LAM weights:"]
        if missing_keys:
            error_lines.append(
                f"Missing keys required by the model but absent from the checkpoint ({len(missing_keys)}):"
            )
            error_lines.extend(f"  - {key}" for key in missing_keys)
        if unexpected_keys:
            error_lines.append(
                f"Unexpected keys present in the checkpoint but unused by the model ({len(unexpected_keys)}):"
            )
            error_lines.extend(f"  - {key}" for key in unexpected_keys)
        if shape_mismatches:
            error_lines.append(f"Keys with shape mismatches ({len(shape_mismatches)}):")
            error_lines.extend(
                f"  - {key}: model{model_shape} vs checkpoint{checkpoint_shape}"
                for key, model_shape, checkpoint_shape in shape_mismatches
            )
        raise RuntimeError("\n".join(error_lines))

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    for parameter in latent_action_model.parameters():
        parameter.requires_grad = False
    return latent_action_model.eval()
