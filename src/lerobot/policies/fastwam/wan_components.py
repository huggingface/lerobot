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
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, UMT5EncoderModel

if TYPE_CHECKING:
    from .wan_adapters import WanVideoVAE38
    from .wan_video_dit import WanVideoDiT

from diffusers import AutoencoderKLWan

from .wan_adapters import WanVideoVAE38
from .wan_video_dit import WanVideoDiT

logger = logging.getLogger(__name__)

# The custom MoT video DiT still ships in the original (non-diffusers) Wan2.2
# repo as sharded `diffusion_pytorch_model*.safetensors`; the VAE and UMT5 text
# encoder come from the diffusers conversion. Tokenizer is the stock UMT5 one.
WAN_DIT_PATTERN = "diffusion_pytorch_model*.safetensors"
WAN_T5_TOKENIZER = "google/umt5-xxl"
WAN22_DIFFUSERS_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class WanTextEncoder(torch.nn.Module):
    """FastWAM text-encoder contract over `transformers.UMT5EncoderModel`.

    Exposes `.dim` (hidden size) and `forward(ids, mask) -> [B, L, dim]`, matching
    the call in `FastWAM.encode_prompt`.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        *,
        pretrained: torch.nn.Module,
    ) -> None:
        super().__init__()
        # UMT5-XXL is a fixed pretrained encoder — never trained from scratch, so a real
        # `UMT5EncoderModel` (with weights) must always be supplied (loaded from the
        # diffusers repo by `load_pretrained_wan_text_encoder`). No random/offline build.
        self.model = pretrained.to(device=device, dtype=dtype)
        self.dim = int(self.model.config.d_model)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids, attention_mask=mask.long()).last_hidden_state


class WanTokenizer:
    """UMT5 tokenizer wrapper returning `(input_ids, attention_mask)` like the
    FastWAM call site expects."""

    def __init__(self, name: str = WAN_T5_TOKENIZER, seq_len: int = 512) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.seq_len = int(seq_len)

    def __call__(
        self,
        sequence: str | Sequence[str],
        return_mask: bool = False,
        add_special_tokens: bool = True,
        **_: Any,
    ):
        if isinstance(sequence, str):
            sequence = [sequence]
        out = self.tokenizer(
            list(sequence),
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        if return_mask:
            return out.input_ids, out.attention_mask
        return out.input_ids


def build_wan_tokenizer(*, tokenizer_max_len: int) -> WanTokenizer:
    return WanTokenizer(name=WAN_T5_TOKENIZER, seq_len=int(tokenizer_max_len))


def load_pretrained_wan_vae(*, torch_dtype: torch.dtype, device: str) -> WanVideoVAE38:
    """Load real Wan2.2 VAE weights from the diffusers repo (offline base creation)."""
    vae = AutoencoderKLWan.from_pretrained(WAN22_DIFFUSERS_MODEL_ID, subfolder="vae", torch_dtype=torch_dtype)
    return WanVideoVAE38(dtype=torch_dtype, device=device, pretrained=vae)


def load_pretrained_wan_text_encoder(*, torch_dtype: torch.dtype, device: str) -> WanTextEncoder:
    """Load real UMT5-XXL encoder weights from the diffusers repo (offline base creation)."""
    encoder = UMT5EncoderModel.from_pretrained(
        WAN22_DIFFUSERS_MODEL_ID, subfolder="text_encoder", torch_dtype=torch_dtype
    )
    return WanTextEncoder(dtype=torch_dtype, device=device, pretrained=encoder)


def resolve_wan_dit_paths(
    model_id_or_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
) -> list[Path]:
    """Resolve the custom MoT DiT shards from the original Wan2.2 repo or a local dir."""
    path = Path(model_id_or_path).expanduser()
    if path.is_dir():
        return sorted(path.glob(WAN_DIT_PATTERN))

    snapshot_path = snapshot_download(
        repo_id=str(model_id_or_path),
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=[WAN_DIT_PATTERN],
    )
    return sorted(Path(snapshot_path).glob(WAN_DIT_PATTERN))


def load_wan_video_dit(
    paths: list[str | Path],
    *,
    dit_config: dict[str, Any],
    torch_dtype: torch.dtype,
    device: str,
) -> WanVideoDiT:
    model = WanVideoDiT(**dit_config)
    state_dict = _read_wan_dit_safetensors(paths)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device=device, dtype=torch_dtype)


def _read_wan_dit_safetensors(paths: list[str | Path]) -> dict[str, torch.Tensor]:
    state_dict = {}
    for path in paths:
        state_dict.update(load_file(str(path), device="cpu"))
    return state_dict


__all__ = [
    "WAN22_DIFFUSERS_MODEL_ID",
    "WAN_DIT_PATTERN",
    "WAN_T5_TOKENIZER",
    "WanTextEncoder",
    "WanTokenizer",
    "build_wan_tokenizer",
    "load_pretrained_wan_text_encoder",
    "load_pretrained_wan_vae",
    "load_wan_video_dit",
    "resolve_wan_dit_paths",
]
