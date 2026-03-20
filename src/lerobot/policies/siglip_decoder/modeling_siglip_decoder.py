#!/usr/bin/env python

import builtins
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPaliGemma, get_gemma_config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.siglip_decoder.configuration_siglip_decoder import SiglipDecoderConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    _, _, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)
    return padded_images


class TokenToImageDecoder(nn.Module):
    def __init__(self, config: SiglipDecoderConfig, embed_dim: int):
        super().__init__()
        hidden_dim = config.decoder_hidden_dim
        self.hidden_dim = hidden_dim
        self.output_activation = config.output_activation
        self.image_resolution = config.image_resolution

        self.in_proj = nn.Linear(embed_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.decoder_n_heads,
            dim_feedforward=int(hidden_dim * config.decoder_mlp_ratio),
            dropout=config.decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(encoder_layer, num_layers=config.decoder_n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.spatial_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, 3, kernel_size=1),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        mixed = self.in_proj(tokens)
        mixed = self.token_mixer(mixed)
        mixed = self.norm(mixed)

        bsize, num_tokens, hidden_dim = mixed.shape
        grid = int(math.sqrt(num_tokens))
        if grid * grid != num_tokens:
            raise ValueError(
                f"Token count ({num_tokens}) is not a perfect square; cannot reshape into spatial grid."
            )

        feat = mixed.transpose(1, 2).reshape(bsize, hidden_dim, grid, grid)
        feat = F.interpolate(feat, size=self.image_resolution, mode="bilinear", align_corners=False)
        pred = self.spatial_head(feat)

        if self.output_activation == "tanh":
            pred = torch.tanh(pred)
        return pred


class SiglipDecoderModel(nn.Module):
    def __init__(self, config: SiglipDecoderConfig):
        super().__init__()
        self.config = config

        paligemma_config = get_gemma_config(config.paligemma_variant)
        self.encoder = PI0FastPaliGemma(
            paligemma_config,
            use_adarms=[False, False],
            precision=config.dtype,
        )
        embed_dim = paligemma_config.width
        self.decoder = TokenToImageDecoder(config, embed_dim=embed_dim)

        if config.load_pi0fast_vision_only_from:
            self.load_vision_only_from_pi0fast_checkpoint(
                ckpt_path=config.load_pi0fast_vision_only_from,
                include_projector=config.include_multi_modal_projector,
                use_model_safetensors=config.load_from_model_safetensors,
            )

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = True

    @staticmethod
    def _load_state_dict_from_path(ckpt_path: str | Path, use_model_safetensors: bool) -> dict[str, Tensor]:
        ckpt_spec = str(ckpt_path)
        local_path = Path(ckpt_spec).expanduser()

        if local_path.is_dir():
            if use_model_safetensors:
                model_file = local_path / "model.safetensors"
                if not model_file.exists():
                    raise FileNotFoundError(f"model.safetensors not found in {local_path}")
                return load_safetensors_file(str(model_file))

            bin_file = local_path / "pytorch_model.bin"
            if not bin_file.exists():
                raise FileNotFoundError(f"pytorch_model.bin not found in {local_path}")
            loaded = torch.load(bin_file, map_location="cpu")
            if isinstance(loaded, dict) and "state_dict" in loaded:
                return loaded["state_dict"]
            return loaded

        if local_path.is_file():
            if local_path.suffix == ".safetensors":
                return load_safetensors_file(str(local_path))
            loaded = torch.load(local_path, map_location="cpu")
            if isinstance(loaded, dict) and "state_dict" in loaded:
                return loaded["state_dict"]
            return loaded

        candidate_filenames = ["model.safetensors", "pytorch_model.bin"]
        if not use_model_safetensors:
            candidate_filenames = ["pytorch_model.bin", "model.safetensors"]

        download_errors = []
        for filename in candidate_filenames:
            try:
                model_file = hf_hub_download(repo_id=ckpt_spec, filename=filename)
                if filename.endswith(".safetensors"):
                    return load_safetensors_file(model_file)
                loaded = torch.load(model_file, map_location="cpu")
                if isinstance(loaded, dict) and "state_dict" in loaded:
                    return loaded["state_dict"]
                return loaded
            except (HfHubHTTPError, ValueError) as err:
                download_errors.append(f"{filename}: {err}")

        raise FileNotFoundError(
            f"Could not load checkpoint from '{ckpt_spec}'. Expected an existing local path or a Hugging Face repo "
            f"containing one of {candidate_filenames}. Download errors: {download_errors}"
        )

    def load_vision_only_from_pi0fast_checkpoint(
        self,
        ckpt_path: str | Path,
        include_projector: bool = True,
        use_model_safetensors: bool = True,
    ):
        state_dict = self._load_state_dict_from_path(ckpt_path, use_model_safetensors)

        vision_prefixes = [
            "model.paligemma_with_expert.paligemma.model.vision_tower.",
            "paligemma_with_expert.paligemma.model.vision_tower.",
        ]
        projector_prefixes = [
            "model.paligemma_with_expert.paligemma.model.multi_modal_projector.",
            "paligemma_with_expert.paligemma.model.multi_modal_projector.",
        ]

        vision_state = {}
        projector_state = {}
        for key, value in state_dict.items():
            for prefix in vision_prefixes:
                if key.startswith(prefix):
                    vision_state[key.removeprefix(prefix)] = value
                    break
            if include_projector:
                for prefix in projector_prefixes:
                    if key.startswith(prefix):
                        projector_state[key.removeprefix(prefix)] = value
                        break

        if not vision_state:
            raise ValueError(
                "No vision_tower parameters found in checkpoint. "
                "Expected PI0Fast-style keys with 'paligemma_with_expert.paligemma.model.vision_tower'."
            )

        missing_v, unexpected_v = self.encoder.paligemma.model.vision_tower.load_state_dict(vision_state, strict=False)
        logging.info(
            "Loaded vision_tower params from %s (keys=%d, missing=%d, unexpected=%d)",
            ckpt_path,
            len(vision_state),
            len(missing_v),
            len(unexpected_v),
        )

        if include_projector and projector_state:
            missing_p, unexpected_p = self.encoder.paligemma.model.multi_modal_projector.load_state_dict(
                projector_state, strict=False
            )
            logging.info(
                "Loaded multi_modal_projector params (keys=%d, missing=%d, unexpected=%d)",
                len(projector_state),
                len(missing_p),
                len(unexpected_p),
            )

    def encode_image_for_target(self, image: Tensor) -> Tensor:
        vision_outputs = self.encoder.paligemma.model.vision_tower(pixel_values=image)
        target_embs = self.encoder.paligemma.model.multi_modal_projector(vision_outputs.last_hidden_state)
        target_embs = F.normalize(target_embs.float(), p=2, dim=-1)
        return target_embs

    def reconstruct(self, image: Tensor) -> tuple[Tensor, Tensor]:
        emb = self.encode_image_for_target(image)
        pred = self.decoder(emb)
        return pred, emb


class SiglipDecoderPolicy(PreTrainedPolicy):
    config_class = SiglipDecoderConfig
    name = "siglip_decoder"

    def __init__(self, config: SiglipDecoderConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = SiglipDecoderModel(config)
        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        return super().from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )

    def _pick_target_image(self, batch: dict[str, Tensor]) -> Tensor:
        image_keys = list(self.config.image_features.keys())
        if self.config.target_image_key is not None:
            if self.config.target_image_key not in batch:
                raise ValueError(
                    f"Configured target_image_key='{self.config.target_image_key}' is not present in batch keys."
                )
            key = self.config.target_image_key
        else:
            present_keys = [key for key in image_keys if key in batch]
            if not present_keys:
                raise ValueError(
                    f"No configured image feature found in batch. configured={image_keys}, batch={list(batch.keys())}"
                )
            key = present_keys[0]

        img = batch[key]
        if img.dim() == 5:
            img = img[:, 0]

        device = next(self.parameters()).device
        if img.device != device:
            img = img.to(device)
        if img.dtype != torch.float32:
            img = img.to(torch.float32)

        is_channels_first = img.shape[1] == 3
        if is_channels_first:
            img = img.permute(0, 2, 3, 1)

        if img.shape[1:3] != self.config.image_resolution:
            img = resize_with_pad_torch(img, *self.config.image_resolution)

        img = img * 2.0 - 1.0
        if is_channels_first:
            img = img.permute(0, 3, 1, 2)
        return img

    def _compute_reconstruction_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.config.recon_loss_type == "l1":
            return F.l1_loss(pred, target)
        if self.config.recon_loss_type == "mse":
            return F.mse_loss(pred, target)
        if self.config.recon_loss_type == "charbonnier":
            return torch.sqrt((pred - target) ** 2 + self.config.charbonnier_eps**2).mean()
        raise ValueError(f"Unsupported recon_loss_type: {self.config.recon_loss_type}")

    @staticmethod
    def _tensor_to_uint8_hwc(img: Tensor) -> np.ndarray:
        frame = ((img.detach().cpu().float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
        return frame.permute(1, 2, 0).numpy()

    def _maybe_compute_lpips(self, pred: Tensor, target: Tensor) -> float | None:
        if not self.config.eval_compute_lpips:
            return None
        try:
            from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
        except ImportError:
            logging.warning(
                "LPIPS requested (eval_compute_lpips=True) but torchmetrics LPIPS is unavailable. Skipping LPIPS."
            )
            return None

        lpips = learned_perceptual_image_patch_similarity(pred, target, net_type="alex", normalize=False)
        return float(lpips.item())

    @torch.no_grad()
    def evaluate_reconstruction_batch(
        self,
        batch: dict[str, Tensor],
        *,
        video_path: str | Path | None = None,
        max_video_frames: int | None = None,
        video_fps: int = 8,
    ) -> dict[str, Any]:
        self.eval()

        target = self._pick_target_image(batch)
        pred, emb = self.model.reconstruct(target)
        loss = self._compute_reconstruction_loss(pred, target)
        mse = F.mse_loss(pred, target)
        psnr = -10.0 * torch.log10(mse.clamp(min=1e-8))
        emb_norm = emb.norm(dim=-1).mean()

        result: dict[str, Any] = {
            "recon_loss": float(loss.item()),
            "recon_mse": float(mse.item()),
            "recon_psnr": float(psnr.item()),
            "embedding_norm": float(emb_norm.item()),
        }

        lpips = self._maybe_compute_lpips(pred, target)
        if lpips is not None:
            result["recon_lpips"] = lpips

        if video_path is not None:
            out_path = Path(video_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            num_frames = target.shape[0]
            if max_video_frames is not None:
                num_frames = min(num_frames, max_video_frames)

            stacked_frames = []
            for frame_idx in range(num_frames):
                gt_frame = self._tensor_to_uint8_hwc(target[frame_idx])
                pred_frame = self._tensor_to_uint8_hwc(pred[frame_idx])
                stacked_frames.append(np.concatenate([gt_frame, pred_frame], axis=1))

            write_video(out_path, stacked_frames, fps=video_fps)
            result["video_path"] = str(out_path)

        return result

    def get_optim_params(self) -> dict:
        return (param for param in self.parameters() if param.requires_grad)

    def reset(self):
        return None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        img = self._pick_target_image(batch)
        pred, _ = self.model.reconstruct(img)
        return pred

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        pred = self.predict_action_chunk(batch, **kwargs)
        return pred[:, :, 0, 0]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        target = self._pick_target_image(batch)
        pred, emb = self.model.reconstruct(target)
        loss = self._compute_reconstruction_loss(pred, target)

        with torch.no_grad():
            mse = F.mse_loss(pred, target)
            psnr = -10.0 * torch.log10(mse.clamp(min=1e-8))
            emb_norm = emb.norm(dim=-1).mean()

        metrics = {
            "loss": float(loss.item()),
            "recon_loss": float(loss.item()),
            "recon_mse": float(mse.item()),
            "recon_psnr": float(psnr.item()),
            "embedding_norm": float(emb_norm.item()),
        }
        return loss, metrics
