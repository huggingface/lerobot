# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Public facade for the first-party DINO-video teacher.

Weight-compatible reimplementation of the published frozen video teacher:
inputs are 5D clips that arrive already resized and ImageNet-normalized, the
runtime casts them to bf16 (weights bf16, RoPE buffers fp32), runs the
frame-block-causal ViT and returns distillation targets. Nothing here imports
or redistributes third-party source, and no fallback to a checkout may be
added.

The return contract of :meth:`DinoVideoTeacher.get_future_feature` is fixed by
``DepthTeacherBundle.video_targets`` and must not drift:

===========  ============================  ======================================
return_cls   return_current               returns
===========  ============================  ======================================
False        False                        ``future_patch [B, 256, 1024]``
True         False                        ``(future_patch, future_cls)``
False        True                         ``(future_patch, current_patch)``
True         True                         ``(future_patch, future_cls,
                                           current_patch, current_cls)``
===========  ============================  ======================================

``future_cls`` is the pooled CLS (``cls_pool`` from config) when
``return_current`` is off, and the future frame's CLS token when it is on;
``current_*`` selects frame ``current_index`` (warmup clips use
``current_index=1``). All outputs are bf16.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .attention import verify_flex_backend
from .backbone import DinoVideoBackboneConfig, FirstPartyDinoVideoBackbone, pack_video_tokens
from .checkpoint import LoadReport, load_backbone_strict, load_dino_video_checkpoint

_BLOCK_CAUSAL_ATTENTION_MODES = frozenset({"flex_block_causal", "sdpa_block_causal"})
_CLS_POOLS = frozenset({"mean", "last"})
_ATTENTION_BACKENDS = frozenset({"sdpa", "flex"})

#: Keys consumed by this builder.
_TEACHER_KEYS = (
    "ckpt_path",
    "config_path",
    "attention_mode",
    "input_size",
    "n_blocks",
    "cls_pool",
    "effective_fps",
)
#: Optional runtime-only keys.
_RUNTIME_KEYS = ("attention_backend", "strict_checkpoint", "device", "runtime", "compile")
#: Documented ``align_params.video`` keys owned by the bundle / loss heads; they
#: are tolerated so the whole video dict can be forwarded, and ignored here.
_BUNDLE_KEYS = (
    "num_future_frames",
    "use_warmup_frame",
    "block_suffix_to_future_video",
    "share_future_depth_query",
    "use_shared_future_task_proj",
    "use_current_shared_task_proj",
    "detach_image_feats",
    "num_layers",
    "num_heads",
    "dim_head",
    "ff_mult",
    "num_backbone_tokens",
    "dim_out",
    "future_video_loss_weight",
    "cls_loss_type",
    "cls_loss_weight",
    "cosine_loss_weight",
    "log_max_samples",
    "log_scale",
    "mse_loss_weight",
    "use_cls_loss",
    "use_cosine_loss",
    "use_current_patch_loss",
    "use_mse_loss",
    "use_patch_loss",
    "use_smooth_l1_loss",
)
_ACCEPTED_KEYS = frozenset(_TEACHER_KEYS + _RUNTIME_KEYS + _BUNDLE_KEYS)
#: Substrings that mark a repository/checkout/provider key.
_REPOSITORY_KEY_HINTS = ("upstream", "provider", "checkout", "repo_root", "vendored")


class DinoVideoTeacher(nn.Module):
    """Frozen video DINO teacher producing patch / CLS distillation targets."""

    def __init__(
        self,
        backbone: FirstPartyDinoVideoBackbone,
        *,
        device: torch.device | str = "cuda",
        attention_mode: str = "flex_block_causal",
        input_size: int = 256,
        n_blocks: int = 1,
        cls_pool: str = "mean",
        effective_fps: float | None = None,
        load_report: LoadReport | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.device = torch.device(device)
        self.attention_mode = attention_mode
        self.input_size = int(input_size)
        self.n_blocks = int(n_blocks)
        self.cls_pool = cls_pool
        self.effective_fps = None if effective_fps is None else float(effective_fps)
        if self.effective_fps is not None and self.effective_fps <= 0:
            raise ValueError(f"effective_fps must be positive, got {self.effective_fps}.")
        self.load_report = load_report

    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str | Path,
        config_path: str | Path,
        *,
        device: torch.device | str = "cuda",
        attention_mode: str | None = None,
        input_size: int = 256,
        n_blocks: int = 1,
        cls_pool: str = "mean",
        effective_fps: float | None = None,
        attention_backend: str = "sdpa",
        strict_checkpoint: bool = True,
    ) -> DinoVideoTeacher:
        """Load the published ``teacher_step_10000.pth`` + ``config.yaml``.

        Must load with ``torch.load(..., weights_only=True)``, take only the
        top-level ``teacher`` state dict, and strictly cover every backbone
        tensor (see :mod:`.checkpoint`).
        """
        ckpt_file = Path(ckpt_path)
        config_file = Path(config_path)
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"DINO-video teacher checkpoint not found at {str(ckpt_file)!r}.")
        if not config_file.is_file():
            raise FileNotFoundError(f"DINO-video teacher config not found at {str(config_file)!r}.")

        published = _read_published_config(config_file)
        student = published.get("dinov3", {}).get("student", {})
        if not isinstance(student, dict):
            raise ValueError(f"{str(config_file)!r}: 'dinov3.student' must be a mapping.")
        backbone_config = _backbone_config_from_published(student, input_size=int(input_size))

        resolved_mode = attention_mode or published.get("attention_mode") or "flex_block_causal"
        if resolved_mode not in _BLOCK_CAUSAL_ATTENTION_MODES:
            raise ValueError(
                f"attention_mode {resolved_mode!r} is not frame-block-causal; get_future_feature "
                f"requires one of {sorted(_BLOCK_CAUSAL_ATTENTION_MODES)}."
            )
        if cls_pool not in _CLS_POOLS:
            raise ValueError(f"cls_pool must be one of {sorted(_CLS_POOLS)}, got {cls_pool!r}.")
        if not 0 < int(n_blocks) <= backbone_config.depth:
            raise ValueError(f"n_blocks must be in [1, {backbone_config.depth}], got {n_blocks}.")
        if attention_backend not in _ATTENTION_BACKENDS:
            raise ValueError(
                f"attention_backend must be one of {sorted(_ATTENTION_BACKENDS)}, got {attention_backend!r}."
            )

        backbone = FirstPartyDinoVideoBackbone(backbone_config, attention_backend=attention_backend)
        state = load_dino_video_checkpoint(ckpt_file)
        report = load_backbone_strict(backbone, state, strict=strict_checkpoint)
        # bf16 weights + fp32 RoPE buffers, frozen; matches the published teacher.
        backbone.cast_for_inference(torch.device(device))
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        if attention_backend == "flex":
            verify_flex_backend(
                num_heads=backbone_config.num_heads,
                head_dim=backbone_config.embed_dim // backbone_config.num_heads,
                device=torch.device(device),
            )
        return cls(
            backbone,
            device=device,
            attention_mode=resolved_mode,
            input_size=int(input_size),
            n_blocks=int(n_blocks),
            cls_pool=cls_pool,
            effective_fps=effective_fps,
            load_report=report,
        )

    @torch.no_grad()
    def get_future_feature(
        self,
        video: torch.Tensor,
        *,
        return_cls: bool = False,
        return_current: bool = False,
        current_index: int = 0,
        fps: float | torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Extract features from a normalized ``[B, C, T, H, W]`` clip.

        Inputs arrive already resized to ``input_size`` and ImageNet-normalized
        by ``DepthTeacherBundle.video_targets`` — this runtime must not repeat
        that preprocessing. ``fps`` scales the temporal RoPE coordinates.
        """
        if video.ndim != 5:
            raise ValueError(f"expected video tensor [B,C,T,H,W], got {tuple(video.shape)}.")
        batch_size, _channels, frames, height, width = video.shape
        if frames < 2:
            raise ValueError("future-video distillation requires at least current and future frames.")
        if not 0 <= int(current_index) < frames:
            raise ValueError(f"current_index must be in [0, {frames}), got {current_index}.")

        video = video.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
        backbone = self.backbone
        patch_tokens = backbone.embed_video(video)  # [B, T, Hp*Wp, D]
        cls_token = backbone.cls_token.to(dtype=video.dtype)
        storage_tokens = (
            None if backbone.storage_tokens is None else backbone.storage_tokens.to(dtype=video.dtype)
        )
        packed = pack_video_tokens(patch_tokens, cls_token, storage_tokens, current_index=int(current_index))
        block_outputs, _ = backbone(packed, fps=self.effective_fps if fps is None else fps)
        # The published teacher reads the last of the blocks it takes, i.e. the
        # final block, regardless of n_blocks; keep that exact semantics.
        raw = block_outputs[-1]

        tokens_per_frame = packed.extras["tokens_per_frame"]
        patches_per_frame = packed.extras["patch_tokens_per_frame"]
        frame_tokens = raw.reshape(batch_size, frames, tokens_per_frame, -1)
        embed_dim = frame_tokens.shape[-1]
        prefix = tokens_per_frame - patches_per_frame
        raw_cls = frame_tokens[:, :, 0, :]
        raw_patches = frame_tokens[:, :, prefix:, :]

        pooled_cls = raw_cls.mean(dim=1) if self.cls_pool == "mean" else raw_cls[:, -1]
        norm = backbone.norm
        pooled_cls = norm(pooled_cls)
        frame_cls = norm(raw_cls)
        patches = norm(raw_patches.reshape(batch_size, frames * patches_per_frame, embed_dim))
        patches = patches.view(batch_size, frames, patches_per_frame, embed_dim)

        future_patches = patches[:, -1].detach().to(dtype=torch.bfloat16)
        if return_cls:
            if return_current:
                current_patches = patches[:, current_index].detach().to(dtype=torch.bfloat16)
                current_cls = frame_cls[:, current_index].detach().to(dtype=torch.bfloat16)
                future_cls = frame_cls[:, -1].detach().to(dtype=torch.bfloat16)
                return future_patches, future_cls, current_patches, current_cls
            future_cls = pooled_cls.detach().to(dtype=torch.bfloat16)
            return future_patches, future_cls
        if return_current:
            current_patches = patches[:, current_index].detach().to(dtype=torch.bfloat16)
            return future_patches, current_patches
        return future_patches


def build_dino_video_teacher(config: dict) -> DinoVideoTeacher:
    """Build the teacher from an ``align_params.video`` dict.

    Recognized keys are the documented user-facing ones (``ckpt_path``,
    ``config_path``, ``attention_mode``, ``input_size``, ``n_blocks``,
    ``cls_pool``, ``effective_fps``) plus optional runtime-only keys
    (``attention_backend``, ``strict_checkpoint``, ``device``, ``runtime``,
    ``compile``) and the bundle-owned ``align_params.video`` keys the rest of
    the recipe forwards (loss weights, projection head shapes, ...), which are
    ignored here. Repository/checkout keys such as ``upstream_root`` must be
    rejected — there is deliberately no upstream fallback.
    """
    if not isinstance(config, dict):
        raise TypeError(f"align_params.video must be a dict, got {type(config).__name__}.")
    rejected = sorted(
        name for name in config if any(hint in str(name).lower() for hint in _REPOSITORY_KEY_HINTS)
    )
    if rejected:
        raise ValueError(
            f"first-party DINO-video runtime only: repository/checkout keys {rejected} are rejected; "
            "supply the published teacher_step_*.pth + config.yaml via ckpt_path/config_path instead."
        )
    unknown = sorted(str(name) for name in config if name not in _ACCEPTED_KEYS)
    if unknown:
        raise ValueError(
            f"unknown align_params.video keys {unknown}; accepted keys are {sorted(_ACCEPTED_KEYS)}."
        )

    ckpt_path = config.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("DINO video teacher requires video.ckpt_path.")
    runtime = config.get("runtime", "first_party")
    if runtime != "first_party":
        raise ValueError(f"only runtime='first_party' exists, got {runtime!r}.")
    if config.get("compile", False):
        raise ValueError("compile=true is not supported by the first-party DINO-video runtime yet.")

    config_path = config.get("config_path")
    if not config_path:
        config_path = Path(ckpt_path).parent / "config.yaml"

    return DinoVideoTeacher.from_pretrained(
        ckpt_path,
        config_path,
        device=config.get("device", "cuda"),
        attention_mode=config.get("attention_mode"),
        input_size=int(config.get("input_size", 256)),
        n_blocks=int(config.get("n_blocks", 1)),
        cls_pool=config.get("cls_pool", "mean"),
        effective_fps=config.get("effective_fps"),
        attention_backend=config.get("attention_backend", "sdpa"),
        strict_checkpoint=bool(config.get("strict_checkpoint", True)),
    )


def _backbone_config_from_published(student: dict, *, input_size: int) -> DinoVideoBackboneConfig:
    """Map the flat published ``dinov3.student`` mapping to a backbone config."""
    rope_3d = bool(student.get("pos_embed_rope_3d", False))
    if not rope_3d:
        raise ValueError(
            "the first-party runtime implements the published 3D-RoPE teacher; "
            "pos_embed_rope_3d=false is not supported."
        )
    return DinoVideoBackboneConfig(
        arch=str(student.get("arch", "vit_large")),
        img_size=int(input_size),
        patch_size=int(student.get("patch_size", 16)),
        n_storage_tokens=int(student.get("n_storage_tokens", 4)),
        qkv_bias=bool(student.get("qkv_bias", True)),
        proj_bias=bool(student.get("proj_bias", True)),
        ffn_bias=bool(student.get("ffn_bias", True)),
        layerscale_init=float(student.get("layerscale", 1e-5)),
        norm_layer=str(student.get("norm_layer", "layernormbf16")),
        rope_normalize_coords=str(student.get("pos_embed_rope_normalize_coords", "separate")),
        rope_prefix_temporal=bool(student.get("pos_embed_rope_prefix_temporal", False)),
        rope_base=float(student.get("pos_embed_rope_base", 100.0)),
        rope_temporal_base=float(student.get("pos_embed_rope_temporal_base", 10000.0)),
        rope_base_fps=float(student.get("pos_embed_rope_base_fps", 24.0)),
        rope_dtype=_rope_dtype(str(student.get("pos_embed_rope_dtype", "fp32"))),
    )


def _rope_dtype(name: str) -> torch.dtype:
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name not in dtypes:
        raise ValueError(f"unknown pos_embed_rope_dtype {name!r}; expected one of {sorted(dtypes)}.")
    return dtypes[name]


def _read_published_config(config_path: Path) -> dict:
    """Parse the published flat ``config.yaml`` without adding a YAML dependency."""
    text = config_path.read_text(encoding="utf-8")
    try:
        return _parse_simple_yaml(text)
    except ValueError as error:
        raise ValueError(f"{str(config_path)!r}: {error}") from error


def _parse_simple_yaml(text: str) -> dict:
    """Parse the mapping/scalar subset used by the published ``config.yaml``.

    Supports nested mappings, block (``- ``) and inline (``[a, b]``) sequences
    of scalars, comments, and null/bool/int/float/quoted-string scalars. The
    subset is deliberately tiny and raises on anything fancier instead of
    guessing — the alternative would be making PyYAML a hard dependency.
    """
    entries: list[tuple[int, int, str]] = []  # (lineno, indent, content)
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        stripped = _strip_yaml_comment(raw_line)
        if not stripped.strip():
            continue
        if stripped.startswith("\t") or " \t" in stripped[: len(stripped) - len(stripped.lstrip(" "))]:
            raise ValueError(f"line {lineno}: tab indentation is not supported.")
        indent = len(stripped) - len(stripped.lstrip(" "))
        entries.append((lineno, indent, stripped.strip()))
    if not entries:
        return {}
    value, index = _parse_yaml_block(entries, 0, entries[0][1])
    if index != len(entries):
        lineno, _indent, _content = entries[index]
        raise ValueError(f"line {lineno}: unexpected indentation.")
    return value


def _parse_yaml_block(
    entries: list[tuple[int, int, str]], start: int, indent: int
) -> tuple[dict | list, int]:
    if entries[start][2].startswith("- "):
        return _parse_yaml_sequence(entries, start, indent)
    mapping: dict = {}
    index = start
    while index < len(entries) and entries[index][1] == indent:
        lineno, _entry_indent, content = entries[index]
        if content.startswith("- "):
            raise ValueError(f"line {lineno}: mixed sequence and mapping entries.")
        key, separator, rest = content.partition(":")
        if not separator:
            raise ValueError(f"line {lineno}: expected 'key: value', got {content!r}.")
        key = _parse_yaml_scalar(key.strip())
        rest = rest.strip()
        if rest:
            mapping[key] = _parse_yaml_flow(rest, lineno)
            index += 1
        elif index + 1 < len(entries) and entries[index + 1][1] > indent:
            mapping[key], index = _parse_yaml_block(entries, index + 1, entries[index + 1][1])
        elif (
            index + 1 < len(entries)
            and entries[index + 1][1] == indent
            and entries[index + 1][2].startswith("- ")
        ):
            # YAML allows the block sequence of a key to sit at the parent indent.
            mapping[key], index = _parse_yaml_sequence(entries, index + 1, indent)
        else:
            mapping[key] = None
            index += 1
    return mapping, index


def _parse_yaml_sequence(entries: list[tuple[int, int, str]], start: int, indent: int) -> tuple[list, int]:
    items: list = []
    index = start
    while index < len(entries) and entries[index][1] == indent and entries[index][2].startswith("- "):
        lineno, _entry_indent, content = entries[index]
        item = content[2:].strip()
        if item.startswith("- ") or ":" in item:
            raise ValueError(f"line {lineno}: only scalar sequence items are supported, got {item!r}.")
        items.append(_parse_yaml_scalar(item))
        index += 1
    return items, index


def _parse_yaml_flow(rest: str, lineno: int) -> object:
    if rest.startswith("[") != rest.endswith("]"):
        raise ValueError(f"line {lineno}: unbalanced inline sequence {rest!r}.")
    if rest.startswith("["):
        inner = rest[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(part.strip()) for part in inner.split(",")]
    return _parse_yaml_scalar(rest)


def _parse_yaml_scalar(token: str) -> object:
    if len(token) >= 2 and token[0] in "'\"" and token[-1] == token[0]:
        return token[1:-1]
    lowered = token.lower()
    if lowered in ("null", "~", ""):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    for index, char in enumerate(line):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t"):
            return line[:index]
    return line
