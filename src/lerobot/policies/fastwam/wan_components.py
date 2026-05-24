# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file

if TYPE_CHECKING:
    from .wan.modules.tokenizers import HuggingfaceTokenizer
    from .wan_adapters import WanVideoVAE38
    from .wan_video_dit import WanVideoDiT

logger = logging.getLogger(__name__)

WAN_DIT_PATTERN = "diffusion_pytorch_model*.safetensors"
WAN_T5_CHECKPOINT = "models_t5_umt5-xxl-enc-bf16.pth"
WAN_T5_TOKENIZER = "google/umt5-xxl"
WAN_VAE_CHECKPOINT = "Wan2.2_VAE.pth"


@dataclass(frozen=True)
class WanCheckpointPaths:
    root: Path
    dit: list[Path]
    vae: Path
    text_encoder: Path | None
    tokenizer: Path | None


@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE38
    text_encoder: torch.nn.Module | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: list[str]
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


def resolve_wan_checkpoint_dir(
    model_id_or_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
) -> Path:
    """Return a local Wan2.2 checkpoint directory.

    Local paths are used directly. Hub repos are downloaded with the same fixed
    component names used by the upstream Wan2.2 inference code.
    """

    path = Path(model_id_or_path).expanduser()
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=str(model_id_or_path),
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=[
            WAN_DIT_PATTERN,
            WAN_T5_CHECKPOINT,
            WAN_VAE_CHECKPOINT,
            f"{WAN_T5_TOKENIZER}/**",
        ],
    )
    return Path(snapshot_path)


def resolve_wan_checkpoint_paths(
    checkpoint_dir: str | Path,
    *,
    tokenizer_dir: str | Path | None = None,
    load_dit: bool = True,
    load_text_encoder: bool = True,
) -> WanCheckpointPaths:
    root = Path(checkpoint_dir).expanduser()
    tokenizer_root = Path(tokenizer_dir).expanduser() if tokenizer_dir is not None else root
    dit = sorted(root.glob(WAN_DIT_PATTERN)) if load_dit else []
    vae = root / WAN_VAE_CHECKPOINT
    text_encoder = root / WAN_T5_CHECKPOINT if load_text_encoder else None
    tokenizer = tokenizer_root / WAN_T5_TOKENIZER if load_text_encoder else None

    missing = []
    if load_dit and len(dit) == 0:
        missing.append(f"DiT ({WAN_DIT_PATTERN})")
    if not vae.exists():
        missing.append(f"VAE ({WAN_VAE_CHECKPOINT})")
    if load_text_encoder:
        if text_encoder is None or not text_encoder.exists():
            missing.append(f"text encoder ({WAN_T5_CHECKPOINT})")
        if tokenizer is None or not tokenizer.exists():
            missing.append(f"tokenizer ({WAN_T5_TOKENIZER})")
    if missing:
        raise FileNotFoundError(
            f"Incomplete Wan2.2 checkpoint directory {root}: missing {', '.join(missing)}."
        )

    return WanCheckpointPaths(
        root=root,
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )


def load_wan_video_dit(
    paths: list[str | Path],
    *,
    dit_config: dict[str, Any],
    torch_dtype: torch.dtype,
    device: str,
) -> WanVideoDiT:
    from .wan_video_dit import WanVideoDiT

    model = WanVideoDiT(**dit_config)
    state_dict = _read_wan_dit_safetensors(paths)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device=device, dtype=torch_dtype)


def load_wan_text_encoder(
    checkpoint_path: str | Path,
    *,
    torch_dtype: torch.dtype,
    device: str,
) -> torch.nn.Module:
    from .wan.modules.t5 import umt5_xxl

    model = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=torch_dtype,
        device=device,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device=device, dtype=torch_dtype)


def load_wan_tokenizer(tokenizer_path: str | Path, *, tokenizer_max_len: int) -> HuggingfaceTokenizer:
    from .wan.modules.tokenizers import HuggingfaceTokenizer

    return HuggingfaceTokenizer(
        name=str(tokenizer_path),
        seq_len=int(tokenizer_max_len),
        clean="whitespace",
    )


def load_wan_vae(checkpoint_path: str | Path, *, torch_dtype: torch.dtype, device: str) -> WanVideoVAE38:
    from .wan_adapters import WanVideoVAE38

    return WanVideoVAE38(vae_pth=str(checkpoint_path), dtype=torch_dtype, device=device)


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_max_len: int = 512,
    dit_config: dict[str, Any] | None = None,
    load_text_encoder: bool = True,
):
    logger.info("Loading Wan2.2-TI2V-5B components...")
    start = time.time()

    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-TI2V-5B loading.")

    checkpoint_dir = resolve_wan_checkpoint_dir(model_id)
    tokenizer_dir = (
        checkpoint_dir if tokenizer_model_id == model_id else resolve_wan_checkpoint_dir(tokenizer_model_id)
    )
    paths = resolve_wan_checkpoint_paths(
        checkpoint_dir,
        tokenizer_dir=tokenizer_dir,
        load_text_encoder=load_text_encoder,
    )

    dit = load_wan_video_dit(
        paths.dit,
        dit_config=dit_config,
        torch_dtype=torch_dtype,
        device=device,
    )
    vae = load_wan_vae(paths.vae, torch_dtype=torch_dtype, device=device)

    text_encoder: torch.nn.Module | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    if load_text_encoder:
        if paths.text_encoder is None or paths.tokenizer is None:
            raise FileNotFoundError("Wan2.2 text encoder/tokenizer paths were not resolved.")
        text_encoder = load_wan_text_encoder(
            paths.text_encoder,
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = load_wan_tokenizer(paths.tokenizer, tokenizer_max_len=tokenizer_max_len)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )

    logger.info("Finished loading Wan2.2-TI2V-5B components in %.2f seconds.", time.time() - start)
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=[str(path) for path in paths.dit],
        vae_path=str(paths.vae),
        text_encoder_path=str(paths.text_encoder) if paths.text_encoder is not None else None,
        tokenizer_path=str(paths.tokenizer) if paths.tokenizer is not None else None,
    )


def _read_wan_dit_safetensors(paths: list[str | Path]) -> dict[str, torch.Tensor]:
    state_dict = {}
    for path in paths:
        state_dict.update(load_file(str(path), device="cpu"))
    return state_dict


__all__ = [
    "WAN_DIT_PATTERN",
    "WAN_T5_CHECKPOINT",
    "WAN_T5_TOKENIZER",
    "WAN_VAE_CHECKPOINT",
    "Wan22LoadedComponents",
    "WanCheckpointPaths",
    "load_wan22_ti2v_5b_components",
    "load_wan_text_encoder",
    "load_wan_tokenizer",
    "load_wan_vae",
    "load_wan_video_dit",
    "resolve_wan_checkpoint_dir",
    "resolve_wan_checkpoint_paths",
]
