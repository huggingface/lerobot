#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import warnings
from collections.abc import Sequence
from pathlib import Path
from transformers import AutoConfig, AutoModel

warnings.filterwarnings("ignore")
import math
import torch.nn.functional as F

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


_VJEPA_HUB_SPECS = {
    "vjepa2_1_vit_base_384": {"feature_dim": 768, "image_size": 384, "checkpoint_key": "ema_encoder"},
    "vjepa2_1_vit_large_384": {"feature_dim": 1024, "image_size": 384, "checkpoint_key": "ema_encoder"},
    "vjepa2_1_vit_giant_384": {"feature_dim": 1408, "image_size": 384, "checkpoint_key": "target_encoder"},
    "vjepa2_1_vit_gigantic_384": {"feature_dim": 1664, "image_size": 384, "checkpoint_key": "target_encoder"},
}

_VJEPA_CHECKPOINT_STEM_TO_HUB = {
    "vjepa2_1_vitb_dist_vitG_384": "vjepa2_1_vit_base_384",
    "vjepa2_1_vitl_dist_vitG_384": "vjepa2_1_vit_large_384",
    "vjepa2_1_vitg_384": "vjepa2_1_vit_giant_384",
    "vjepa2_1_vitG_384": "vjepa2_1_vit_gigantic_384",
}


def _get_existing_path(model_id: str) -> Path | None:
    path = Path(str(model_id)).expanduser()
    return path if path.exists() else None


def _find_local_vjepa_repo() -> Path | None:
    repo_path = os.environ.get("VJEPA2_REPO_PATH")
    if repo_path:
        path = Path(repo_path).expanduser()
        if path.exists() and (path / "hubconf.py").exists():
            return path
    return None


def _infer_vjepa_hub_name(model_id: str) -> str | None:
    key = str(model_id).strip()
    if key in _VJEPA_HUB_SPECS:
        return key

    path = Path(key).expanduser()
    stem = path.stem if path.suffix else path.name
    if stem in _VJEPA_HUB_SPECS:
        return stem
    return _VJEPA_CHECKPOINT_STEM_TO_HUB.get(stem)


def _is_cosmos_tokenizer_dir(model_id: str | Path) -> bool:
    path = Path(str(model_id)).expanduser()
    if not (path.exists() and path.is_dir()):
        return False
    has_full_autoencoder = (path / "autoencoder.jit").exists()
    has_split_autoencoder = (path / "encoder.jit").exists() and (path / "decoder.jit").exists()
    return has_full_autoencoder or has_split_autoencoder


def _looks_like_cosmos_model_id(model_id: str | Path) -> bool:
    key = str(model_id).lower()
    return "cosmos" in key or _is_cosmos_tokenizer_dir(model_id)


def _infer_cosmos_variant(model_id: str | Path) -> tuple[str | None, int | None]:
    name = Path(str(model_id)).expanduser().name.lower()
    tokenizer_kind = None
    if "ci" in name:
        tokenizer_kind = "continuous_image"
    elif "di" in name:
        tokenizer_kind = "discrete_image"
    elif "cv" in name:
        tokenizer_kind = "continuous_video"
    elif "dv" in name:
        tokenizer_kind = "discrete_video"

    spatial_compression = None
    if "16x16" in name:
        spatial_compression = 16
    elif "8x8" in name:
        spatial_compression = 8
    return tokenizer_kind, spatial_compression


def _clean_vjepa_backbone_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _safe_torch_load(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility for older PyTorch versions that do not support `weights_only`.
        return torch.load(path, map_location="cpu")  # nosec B614


class VJEPAEncoder(nn.Module):
    """Wrap V-JEPA2 as an image-sequence feature encoder returning [B, T, K, D]."""

    def __init__(
        self,
        model_id: str = "vjepa2_1_vit_large_384",
        norm_layer_type: str = "l2",
        enable_norm: bool = False,
    ):
        super().__init__()
        # Keep encoder construction on CPU; Lightning will place modules on per-rank devices.
        self.device = torch.device("cpu")
        self.model_id = str(model_id)
        self.norm_layer_type = str(norm_layer_type).lower()
        self.enable_norm = bool(enable_norm)
        self.feature_dim = 1024
        self.image_size = 256
        self.patch_size = 16
        self.tubelet_size = 2
        self.img_temporal_dim_size = None
        self._warned_resolution_mismatch = False

        existing_path = _get_existing_path(self.model_id)
        hub_name = _infer_vjepa_hub_name(self.model_id)
        if existing_path is not None and existing_path.is_dir():
            self._load_hf_model(str(existing_path))
        elif hub_name is not None:
            checkpoint_path = (
                str(existing_path) if existing_path is not None and existing_path.is_file() else None
            )
            self._load_torch_hub_model(hub_name=hub_name, checkpoint_path=checkpoint_path)
        else:
            self._load_hf_model(self.model_id)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        if self.norm_layer_type in ("bn", "ln"):
            if self.norm_layer_type == "bn":
                self.latent_norm = nn.SyncBatchNorm(self.feature_dim, affine=False)
            else:
                self.latent_norm = nn.LayerNorm(self.feature_dim, elementwise_affine=False)
        else:
            self.latent_norm = None

    def _load_hf_model(self, model_id: str) -> None:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.image_size = int(getattr(cfg, "image_size", getattr(cfg, "crop_size", self.image_size)))
        self.patch_size = int(getattr(cfg, "patch_size", self.patch_size))
        self.tubelet_size = int(getattr(cfg, "tubelet_size", self.tubelet_size))
        self.feature_dim = int(getattr(cfg, "hidden_size", self.feature_dim))
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        self.img_temporal_dim_size = getattr(self.model, "img_temporal_dim_size", None)

    def _load_torch_hub_model(self, hub_name: str, checkpoint_path: str | None) -> None:
        spec = _VJEPA_HUB_SPECS[hub_name]
        self.image_size = int(spec["image_size"])
        self.patch_size = 16
        self.tubelet_size = 2

        local_repo = _find_local_vjepa_repo()
        if local_repo is None:
            raise RuntimeError(
                "V-JEPA hub repo not found locally. Set VJEPA2_REPO_PATH to a clone "
                "that contains hubconf.py, or pass a Hugging Face model id/directory."
            )
        encoder, _ = torch.hub.load(  # nosec B614
            str(local_repo),
            hub_name,
            pretrained=checkpoint_path is None,
            source="local",
        )
        if checkpoint_path is not None:
            ckpt = _safe_torch_load(checkpoint_path)
            checkpoint_key = str(spec["checkpoint_key"])
            if checkpoint_key not in ckpt:
                raise KeyError(
                    f"Checkpoint `{checkpoint_path}` does not contain key `{checkpoint_key}`. "
                    f"Available top-level keys: {list(ckpt.keys())}"
                )
            msg = encoder.load_state_dict(
                _clean_vjepa_backbone_state_dict(ckpt[checkpoint_key]),
                strict=False,
            )
            missing = [k for k in msg.missing_keys if "pos_embed" not in k]
            unexpected = [k for k in msg.unexpected_keys if "pos_embed" not in k]
            if missing or unexpected:
                raise RuntimeError(
                    f"Failed to load V-JEPA checkpoint `{checkpoint_path}` cleanly. "
                    f"Missing keys: {missing}; unexpected keys: {unexpected}"
                )

        self.model = encoder
        self.feature_dim = int(getattr(encoder, "embed_dim", spec["feature_dim"]))
        self.img_temporal_dim_size = getattr(encoder, "img_temporal_dim_size", None)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def _apply_latent_norm(self, encoded_features: torch.Tensor, use_norm: bool) -> torch.Tensor:
        if not use_norm:
            return encoded_features
        if self.norm_layer_type == "bn":
            if self.latent_norm is None:
                raise ValueError("VJEPAEncoder has no BN layer initialized; set norm_layer_type to 'bn'.")
            flat_features = encoded_features.reshape(-1, self.feature_dim)
            flat_features = self.latent_norm(flat_features)
            return flat_features.view_as(encoded_features)
        if self.norm_layer_type == "ln":
            if self.latent_norm is None:
                raise ValueError("VJEPAEncoder has no LN layer initialized; set norm_layer_type to 'ln'.")
            return self.latent_norm(encoded_features)
        if self.norm_layer_type == "l2":
            return F.normalize(encoded_features, p=2, dim=-1)
        raise ValueError(
            f"Unsupported norm_layer_type={self.norm_layer_type!r} for VJEPAEncoder. "
            "Expected one of: 'l2', 'ln', 'bn'."
        )

    @torch.no_grad()
    def _extract_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_vision_features"):
            return self.model.get_vision_features(pixel_values_videos)
        # Official torch hub backbone expects [B, C, T, H, W].
        pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4).contiguous()
        return self.model(pixel_values_videos)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, norm_latents: bool | None = None, n: int = -1) -> torch.Tensor:
        if images.dim() != 4:
            B, T, C, H, W = images.shape
            images = images.reshape(-1, C, H, W)
        else:
            B, C, H, W = images.shape
            T = 1
        assert images.dim() == 4, (
            f"Expected a 4D tensor [B*T, C, H, W], got: {images.shape}; invalid image dimensions"
        )
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"V-JEPA expects image size divisible by patch_size={self.patch_size}, got {(H, W)}."
            )
        if (H, W) != (self.image_size, self.image_size) and not self._warned_resolution_mismatch:
            warnings.warn(
                f"V-JEPA encoder `{self.model_id}` is configured for {self.image_size}x{self.image_size}, "
                f"but received {(H, W)}. This is supported only if your LAM config uses the same token grid.",
                stacklevel=2,
            )
            self._warned_resolution_mismatch = True
        # V-JEPA 2.1 backbones switch to image patch embedding when T=1.
        video_like = images.unsqueeze(1)  # [B*T, 1, C, H, W]

        encoded_features = self._extract_features(video_like)  # [B*T, K, D]
        use_norm = self.enable_norm if norm_latents is None else bool(norm_latents)
        encoded_features = self._apply_latent_norm(encoded_features, use_norm=use_norm)
        # print(encoded_features.min(), encoded_features.max(), encoded_features.mean(), encoded_features.std())
        return encoded_features.reshape(
            B, T, encoded_features.shape[-2], encoded_features.shape[-1]
        ).detach()  # [B, T, K, D]

    @torch.no_grad()
    def encode_video(
        self,
        videos: torch.Tensor,
        norm_latents: bool | None = None,
    ) -> torch.Tensor:
        B, T, C, H, W = videos.shape
        if T % self.tubelet_size != 0:
            raise ValueError(
                f"V-JEPA video encoder requires T divisible by tubelet_size={self.tubelet_size}, got T={T}."
            )
        with torch.no_grad():
            encoded_features = self._extract_features(videos)
        use_norm = self.enable_norm if norm_latents is None else bool(norm_latents)
        encoded_features = self._apply_latent_norm(encoded_features, use_norm=use_norm)
        temporal_tokens = T // self.tubelet_size
        spatial_tokens = encoded_features.shape[-2] // temporal_tokens
        return encoded_features.reshape(B, temporal_tokens, spatial_tokens, self.feature_dim).detach()


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_latent_layers: int = 1,
        norm_layer_type: str = "l2",
        enable_norm: bool = False,
    ):
        super().__init__()
        # Keep encoder construction on CPU; Lightning will place modules on per-rank devices.
        self.device = torch.device("cpu")
        self.model_id = model_id
        self.num_latent_layers = max(int(num_latent_layers), 1)
        self.norm_layer_type = norm_layer_type
        self.enable_norm = enable_norm
        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, dtype=torch.float32)
        model.eval()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        hidden_size = getattr(self.model.config, "hidden_size", None)
        self.feature_dim = int(hidden_size) if hidden_size is not None else 1024
        if self.norm_layer_type in ("bn", "ln"):
            if self.norm_layer_type == "bn":
                norm_builder = lambda: nn.SyncBatchNorm(self.feature_dim, affine=False)
            else:
                norm_builder = lambda: nn.LayerNorm(self.feature_dim, elementwise_affine=False)
            self.latent_norms = nn.ModuleList([norm_builder() for _ in range(self.num_latent_layers)])
        else:
            self.latent_norms = None

    def train(self, mode: bool = True):
        # Keep the frozen DINO backbone and optional normalization layers in eval mode.
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode(self, images: torch.Tensor, remove_cls: bool = True, n: int | Sequence = -1) -> torch.Tensor:
        # DINOv3 hidden_states[-2] is the penultimate block output; this is the
        # layer used by the default LAM configs through latent_layer_to_use=-2.
        if images.dim() != 4:
            B, T, C, H, W = images.shape
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        else:
            B, C, H, W = images.shape
            T = 1
        need_all_layers = not (isinstance(n, int) and n == -1)
        outputs = self.model(pixel_values=images, output_hidden_states=need_all_layers)
        if not need_all_layers:
            last = outputs.last_hidden_state
            if remove_cls:
                tokens = last[:, 5:, :]  # [B*T, K, D]
            else:
                tokens = last  # [B*T, 5+K, D]

            if self.enable_norm:
                if self.norm_layer_type == "bn":
                    if self.latent_norms is None:
                        raise ValueError(
                            "DINOv3Encoder has no BN layer initialized; set norm_layer_type to 'bn'."
                        )
                    tokens_2d = tokens.reshape(-1, self.feature_dim)  # [B*T*K, D]
                    tokens_2d = self.latent_norms[0](tokens_2d)
                    tokens = tokens_2d.view(tokens.shape[0], tokens.shape[1], self.feature_dim)
                elif self.norm_layer_type == "ln":
                    if self.latent_norms is None:
                        raise ValueError(
                            "DINOv3Encoder has no LN layer initialized; set norm_layer_type to 'ln'."
                        )
                    tokens = self.latent_norms[0](tokens)
                elif self.norm_layer_type == "l2":
                    tokens = F.normalize(tokens, p=2, dim=-1)

            features = tokens.reshape(B, T, -1, self.feature_dim)
            return features.detach()
        else:
            hidden_states = outputs.hidden_states
            if isinstance(n, int):
                list_n = [n]
            else:
                list_n = n
            assert len(list_n) <= self.num_latent_layers, (
                f"DINOv3Encoder expected at most {self.num_latent_layers} normalization layers, "
                f"but received {len(list_n)} feature layers. Ensure this matches len(latent_layer_to_use)."
            )
            features = []
            for idx, i in enumerate(list_n):
                layer_tokens = hidden_states[i]  # [B*T, 5+K, D]
                if remove_cls:
                    layer_tokens = layer_tokens[:, 5:, :]  # [B*T, K, D]

                if self.enable_norm:
                    if self.norm_layer_type == "bn":
                        if self.latent_norms is None:
                            raise ValueError(
                                "DINOv3Encoder has no BN layer initialized; set norm_layer_type to 'bn'."
                            )
                        lt_2d = layer_tokens.reshape(-1, self.feature_dim)  # [B*T*K, D]
                        lt_2d = self.latent_norms[idx](lt_2d)
                        layer_tokens = lt_2d.view(
                            layer_tokens.shape[0], layer_tokens.shape[1], self.feature_dim
                        )
                    elif self.norm_layer_type == "ln":
                        if self.latent_norms is None:
                            raise ValueError(
                                "DINOv3Encoder has no LN layer initialized; set norm_layer_type to 'ln'."
                            )
                        layer_tokens = self.latent_norms[idx](layer_tokens)
                    elif self.norm_layer_type == "l2":
                        layer_tokens = F.normalize(layer_tokens, p=2, dim=-1)

                features.append(layer_tokens.reshape(B, T, -1, self.feature_dim))

            return features[0].detach() if isinstance(n, int) else [f.detach() for f in features]


class CosmosAutoencoder(nn.Module):
    def __init__(
        self,
        model_id: str = "cosmos:Cosmos-0.1-Tokenizer-CI16x16",
        dtype: str = "bfloat16",
        tokenizer_config: dict | None = None,
        norm_layer_type: str = "l2",
        enable_norm: bool = False,
    ):
        super().__init__()
        self.device = torch.device("cpu")
        self.model_id = str(model_id)
        self.dtype_str = str(dtype)
        self.norm_layer_type = str(norm_layer_type).lower()
        self.enable_norm = bool(enable_norm)
        self.feature_dim = 16
        self.image_size = None

        model_path = Path(self.model_id).expanduser()
        if not (model_path.exists() and model_path.is_dir()):
            raise ValueError(
                "CosmosAutoencoder expects model_id to be a local directory containing weight files."
            )

        tokenizer_kind, spatial_compression = _infer_cosmos_variant(model_path)
        if tokenizer_kind != "continuous_image":
            raise NotImplementedError(
                "This branch only supports the Cosmos continuous image tokenizer, such as `Cosmos-0.1-Tokenizer-CI16x16`."
            )
        if spatial_compression != 16:
            raise NotImplementedError(
                "This LAM branch only supports `Cosmos-Tokenizer-CI16x16` because the backbone assumes "
                "patch_size=16 and aligns to the visual feature grid."
            )
        self.patch_size = int(spatial_compression)

        ae = model_path / "autoencoder.jit"
        enc = model_path / "encoder.jit"
        dec = model_path / "decoder.jit"

        if enc.exists() and dec.exists():
            self.encoder_model = torch.jit.load(str(enc), map_location="cpu").eval()  # nosec B614
            self.decoder_model = torch.jit.load(str(dec), map_location="cpu").eval()  # nosec B614
            self.autoencoder_model = (
                torch.jit.load(str(ae), map_location="cpu").eval() if ae.exists() else None  # nosec B614
            )
        elif ae.exists():
            self.encoder_model = None
            self.decoder_model = None
            self.autoencoder_model = torch.jit.load(str(ae), map_location="cpu").eval()  # nosec B614
        else:
            raise FileNotFoundError(
                f"No valid weights found: expected {ae} or the paired files {enc} and {dec}"
            )

        for module in (self.encoder_model, self.decoder_model, self.autoencoder_model):
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = False

        self.register_buffer("_runtime_anchor", torch.empty(0), persistent=False)
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        if self.norm_layer_type in ("bn", "ln"):
            if self.norm_layer_type == "bn":
                self.latent_norm = nn.SyncBatchNorm(self.feature_dim, affine=False)
            else:
                self.latent_norm = nn.LayerNorm(self.feature_dim, elementwise_affine=False)
        else:
            self.latent_norm = None

    def train(self, mode: bool = True):
        super().train(False)
        for module in (self.encoder_model, self.decoder_model, self.autoencoder_model):
            if module is not None:
                module.eval()
        return self

    def _apply_latent_norm(self, encoded_features: torch.Tensor, use_norm: bool) -> torch.Tensor:
        if not use_norm:
            return encoded_features
        if self.norm_layer_type == "bn":
            if self.latent_norm is None:
                raise ValueError(
                    "CosmosAutoencoder has no BN layer initialized; set norm_layer_type to 'bn'."
                )
            flat_features = encoded_features.reshape(-1, self.feature_dim)
            flat_features = self.latent_norm(flat_features)
            return flat_features.view_as(encoded_features)
        if self.norm_layer_type == "ln":
            if self.latent_norm is None:
                raise ValueError(
                    "CosmosAutoencoder has no LN layer initialized; set norm_layer_type to 'ln'."
                )
            return self.latent_norm(encoded_features)
        if self.norm_layer_type == "l2":
            return F.normalize(encoded_features, p=2, dim=-1)
        raise ValueError(
            f"Unsupported norm_layer_type={self.norm_layer_type!r} for CosmosAutoencoder. "
            "Expected one of: 'l2', 'ln', 'bn'."
        )

    def _module_param_dtype(
        self, module: nn.Module | None, default: torch.dtype = torch.float32
    ) -> torch.dtype:
        if module is None:
            return default
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return default

    @torch.no_grad()
    def _denormalize_imagenet_to_unit(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float() * self._imagenet_std + self._imagenet_mean
        return images.clamp_(0.0, 1.0)

    @torch.no_grad()
    def _unit_to_negone_posone(self, images01: torch.Tensor) -> torch.Tensor:
        return images01.mul(2.0).sub(1.0).clamp(-1.0, 1.0)

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        norm_latents: bool | None = None,
        n: int = -1,
    ) -> torch.Tensor:
        if images.dim() != 4:
            B, T, C, H, W = images.shape
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        else:
            B, C, H, W = images.shape
            T = 1
        images = images.to(device=self._runtime_anchor.device, dtype=torch.float32)
        images = self._denormalize_imagenet_to_unit(images)
        encoder_dtype = self._module_param_dtype(self.encoder_model)
        imgs11 = self._unit_to_negone_posone(images).to(dtype=encoder_dtype)
        if self.encoder_model is None:
            raise RuntimeError(
                "The current Cosmos weight directory does not provide `encoder.jit`, so encode cannot run."
            )
        token = self.encoder_model(imgs11)
        if isinstance(token, (tuple, list)):
            token = token[0]
        if not isinstance(token, torch.Tensor) or token.ndim != 4:
            raise RuntimeError(
                "CosmosAutoencoder.encode expected a continuous image latent, "
                f"but got {type(token).__name__} with shape {getattr(token, 'shape', None)}"
            )
        bt, latent_channels, latent_h, latent_w = token.shape
        token = token.reshape(B, T, latent_channels, latent_h * latent_w).permute(0, 1, 3, 2).contiguous()
        use_norm = self.enable_norm if norm_latents is None else bool(norm_latents)
        token = self._apply_latent_norm(token, use_norm=use_norm)
        return token.detach()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            B, T, num_tokens, channels = latent.shape
            latent = latent.reshape(B * T, num_tokens, channels)
            restore_video = True
        elif latent.ndim == 3:
            B, num_tokens, channels = latent.shape
            T = 1
            restore_video = False
        else:
            raise ValueError(f"CosmosAutoencoder.decode expected a 3D/4D latent, got: {tuple(latent.shape)}")

        if channels != self.feature_dim:
            raise ValueError(
                f"CosmosAutoencoder.decode expected latent channel count {self.feature_dim}, got: {channels}"
            )
        grid_size = int(round(math.sqrt(num_tokens)))
        if grid_size * grid_size != num_tokens:
            raise ValueError(
                f"CosmosAutoencoder.decode only supports square-grid latents, got num_tokens={num_tokens}."
            )

        decoder_dtype = self._module_param_dtype(self.decoder_model)
        latent = latent.to(device=self._runtime_anchor.device, dtype=decoder_dtype)
        latent = latent.permute(0, 2, 1).contiguous().reshape(latent.shape[0], channels, grid_size, grid_size)
        if self.decoder_model is None:
            raise RuntimeError(
                "The current Cosmos weight directory does not provide `decoder.jit`, so decode cannot run."
            )
        decoded = self.decoder_model(latent).clamp(-1.0, 1.0)
        if restore_video:
            decoded = decoded.reshape(B, T, *decoded.shape[1:])
        return decoded

    @torch.no_grad()
    def autoencode(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(f"Expected 4D [B*T,3,H,W], got: {images.shape}")
        images = images.to(device=self._runtime_anchor.device, dtype=torch.float32)
        imgs01 = self._denormalize_imagenet_to_unit(images)
        autoencoder_dtype = self._module_param_dtype(self.autoencoder_model or self.encoder_model)
        imgs11 = self._unit_to_negone_posone(imgs01).to(dtype=autoencoder_dtype)
        if self.autoencoder_model is not None:
            out = self.autoencoder_model(imgs11)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out.clamp(-1.0, 1.0)
        if self.encoder_model is None or self.decoder_model is None:
            raise RuntimeError(
                "The current Cosmos weight directory is missing model files required for autoencode."
            )
        return self.decoder_model(self.encoder_model(imgs11)).clamp(-1.0, 1.0)


def build_vision_encoder(
    model_id: str,
    num_latent_layers: int = 1,
    norm_layer_type: str = "l2",
    enable_norm: bool = False,
) -> nn.Module:
    key = str(model_id).lower()
    vjepa_hub_name = _infer_vjepa_hub_name(model_id)
    if "dinov3" in key:
        encoder = DINOv3Encoder(
            model_id=model_id,
            num_latent_layers=num_latent_layers,
            norm_layer_type=norm_layer_type,
            enable_norm=enable_norm,
        )
        return encoder, encoder.feature_dim
    elif "vjepa" in key or "jepa" in key or vjepa_hub_name is not None:
        encoder = VJEPAEncoder(
            model_id=model_id,
            norm_layer_type=norm_layer_type,
            enable_norm=enable_norm,
        )
        return encoder, encoder.feature_dim
    elif _looks_like_cosmos_model_id(model_id):
        encoder = CosmosAutoencoder(
            model_id=model_id,
            norm_layer_type=norm_layer_type,
            enable_norm=enable_norm,
        )
        return encoder, encoder.feature_dim

    else:
        print("No pretrained model is used; falling back to the PatchEmbed encoder")
        return None, 0
