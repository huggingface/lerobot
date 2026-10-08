# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Wire an embodiment-specific action modality into the FLUX 3 Action DiT.

Vendored from black-forest-labs/flux-action (``src/flux_action/models/wiring.py``).

The action-pretrained trunk carries the video model plus the co-trained action branch:
``action_prediction`` / ``action_prediction_cond`` mode blocks and stream modulations. It may also carry
``emb_in`` / ``final_layer`` heads of action modalities that are not yours; their channel count and meaning
are not yours, so they are dropped. A finetune adds YOUR modality on top of it:

* mode blocks and modulations for ``<modality>`` / ``<modality>_cond`` are initialized from the
  co-trained ``action_prediction`` branch (dataset-specific action modalities route through one shared
  action backbone);
* ``emb_in.<modality>``, ``emb_in.<modality>_cond``, ``final_layer.<modality>`` and
  ``final_layer.<modality>_cond`` are fresh, sized to your action channels;
* everything else (video model, text path, single-stream blocks) loads as is.

A checkpoint written by a finetune already contains the embodiment keys and loads with
``strict_heads=True``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import replace

import torch
from safetensors import safe_open
from torch import nn

from ..utils import default_dtype
from .transformer import JointSingleSeq, JointSingleSeqParams, LastLayer

SHARED_ACTION_MODE = "action_prediction"
SHARED_KINDS = (
    "content_mode_blocks",
    "early_stream_modulations",
    "single_stream_modulations",
    "late_stream_modulations",
    "late_content_mode_blocks",
)
HEAD_KINDS = ("emb_in", "final_layer")
# Full generative stream names identify unused checkpoint weights to filter.
# Action policies build only video/video_cond; image/audio streams never enter their joint attention.
CONTENT_STREAMS = ("video", "video_cond", "image", "image_cond", "audio", "audio_cond")
REQUIRED_CONTENT_STREAMS = ("video", "video_cond")
_STREAM_MODULE_KINDS = (
    "emb_in",
    "final_layer",
    "content_mode_blocks",
    "early_stream_modulations",
    "single_stream_modulations",
    "late_stream_modulations",
    "late_content_mode_blocks",
)


def action_dit_params(
    base: JointSingleSeqParams,
    modality: str,
    channels: int,
    attn_mode: str | None = None,
    conditioning_channels: int | None = None,
) -> JointSingleSeqParams:
    """``base`` plus the ``<modality>`` and ``<modality>_cond`` content streams."""
    in_channels = dict(base.in_channels)
    in_channels[modality] = channels
    in_channels[f"{modality}_cond"] = conditioning_channels or channels
    sequence = dict(base.sequence)
    sequence[f"x_{modality}"] = modality
    sequence[f"x_{modality}_cond"] = f"{modality}_cond"
    params = replace(base, in_channels=in_channels, sequence=sequence)
    if attn_mode is not None:
        params = replace(params, attn_mode=attn_mode)
    return params


def restrict_content_streams(base: JointSingleSeqParams, streams: Sequence[str]) -> JointSingleSeqParams:
    """``base`` with only ``streams`` among its content streams (the action modality is added afterwards)."""
    unknown = [s for s in streams if s not in base.in_channels]
    if unknown:
        raise ValueError(f"unknown content streams {unknown}; the DiT defines {sorted(base.in_channels)}")
    keep = set(streams)
    in_channels = {m: c for m, c in base.in_channels.items() if m in keep}
    sequence = {k: v for k, v in base.sequence.items() if v in keep}
    return replace(base, in_channels=in_channels, sequence=sequence)


def stream_of_key(key: str) -> str | None:
    """Name of the stream a DiT parameter belongs to (``video``, ``txt``, a head modality, ...), else ``None``."""
    parts = key.removeprefix("dit.").split(".")
    if len(parts) >= 3 and parts[0] in _STREAM_MODULE_KINDS:
        return parts[1]
    return None


def unused_content_key(key: str, streams: Collection[str]) -> bool:
    """Only discard recognized content-stream weights absent from the requested architecture."""
    stream = stream_of_key(key)
    return stream in CONTENT_STREAMS and stream not in streams


def detect_content_streams(ckpt_path: str) -> list[str]:
    """Content streams a trunk / finetune ``.safetensors`` carries (``emb_in.<stream>.weight``), canonical order."""
    with safe_open(ckpt_path, framework="pt") as f:
        keys = set(f.keys())
    found = [s for s in CONTENT_STREAMS if f"emb_in.{s}.weight" in keys or f"dit.emb_in.{s}.weight" in keys]
    missing = [s for s in REQUIRED_CONTENT_STREAMS if s not in found]
    if missing:
        raise ValueError(f"{ckpt_path} lacks the {missing} stream(s) the policy feeds")
    return found


def fresh_module_names(modality: str) -> list[str]:
    return [f"{kind}.{m}" for kind in HEAD_KINDS for m in (modality, f"{modality}_cond")]


def remap_shared_action_keys(state_dict: dict[str, torch.Tensor], modality: str) -> dict[str, torch.Tensor]:
    """Route the co-trained action backbone to ``modality``; drop every other action head.

    ``content_mode_blocks.action_prediction.*``      -> ``content_mode_blocks.<modality>.*``
    ``*_stream_modulations.action_prediction_cond.*`` -> ``*_stream_modulations.<modality>_cond.*``
    ``emb_in.action_prediction*.*`` / ``final_layer.action_prediction*.*`` -> dropped, unless they
    already belong to ``modality`` (a finetuned checkpoint)
    """
    ours = (modality, f"{modality}_cond")
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] in HEAD_KINDS and parts[1].startswith(SHARED_ACTION_MODE):
            if parts[1] not in ours:
                continue  # generic head or another embodiment's head: channels are not ours
        elif (
            modality != SHARED_ACTION_MODE
            and len(parts) >= 3
            and parts[0] in SHARED_KINDS
            and parts[1] in (SHARED_ACTION_MODE, f"{SHARED_ACTION_MODE}_cond")
        ):
            parts[1] = modality if parts[1] == SHARED_ACTION_MODE else f"{modality}_cond"
            out[".".join(parts)] = value
            continue
        out[key] = value
    return out


def fresh_head_state_dict(
    hidden_size: int, modality: str, channels: int, seed: int = 0, cond_channels: int | None = None
) -> dict[str, torch.Tensor]:
    """Tensors for fresh embodiment heads with the reference model's initialization.

    The reference initializes every linear layer xavier-uniform and then zeroes the final layers
    (output projection and its modulation), so a new head predicts zero at first and only its output
    projection moves at the first update. The released action trunk carries untrained embodiment heads
    in exactly this state. ``seed`` drives the xavier draw of the input projections through a local
    generator (the global RNG is untouched); ``cond_channels`` sizes the conditioning head when it
    differs from the action width (the SO-101 history token).
    """
    generator = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for m, width in ((modality, channels), (f"{modality}_cond", cond_channels or channels)):
        weight = torch.empty(hidden_size, width)
        nn.init.xavier_uniform_(weight, generator=generator)
        out[f"emb_in.{m}.weight"] = weight
        with torch.random.fork_rng(devices=[]):
            layer_state = LastLayer(hidden_size, width).state_dict()
        for k, v in layer_state.items():
            out[f"final_layer.{m}.{k}"] = torch.zeros_like(v)
    return out


def missing_shared_keys(model: nn.Module, sd: dict[str, torch.Tensor], modality: str) -> list[str]:
    pat = re.compile(rf"^({'|'.join(SHARED_KINDS)})\.({re.escape(modality)}|{re.escape(modality)}_cond)\.")
    return [k for k in model.state_dict() if pat.match(k) and k not in sd]


def load_action_checkpoint(
    model: JointSingleSeq,
    ckpt_path: str,
    modality: str,
    *,
    strict_heads: bool = False,
    head_seed: int = 0,
) -> None:
    """Load a trunk / finetuned ``.safetensors`` into an action DiT built with :func:`action_dit_params`.

    ``strict_heads=True`` demands that the checkpoint already contains the embodiment heads (a finetuned
    checkpoint); ``False`` accepts an action-pretrained trunk and gives the missing head tensors the
    reference initialization of :func:`fresh_head_state_dict`, drawn from ``head_seed`` so every rank
    builds the same heads whatever its own RNG state.
    """
    dtype = next(model.parameters()).dtype
    # Filter before materializing tensors, so full checkpoints do not allocate the unused streams.
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        sd = {k: f.get_tensor(k) for k in f.keys() if not unused_content_key(k, model.in_channels)}  # noqa: SIM118
    sd = remap_shared_action_keys(sd, modality)
    expected = model.state_dict()
    fresh_prefixes = tuple(n + "." for n in fresh_module_names(modality))
    if strict_heads:
        missing_heads = [k for k in expected if k.startswith(fresh_prefixes) and k not in sd]
        if missing_heads:
            raise ValueError(f"checkpoint missing required embodiment heads: {missing_heads[:6]}")
    missing = [k for k in expected.keys() - sd.keys() if not k.startswith(fresh_prefixes)]
    if missing:
        raise ValueError(f"checkpoint missing required keys, e.g. {missing[:6]}")
    shared_missing = [k for k in missing_shared_keys(model, sd, modality) if not k.startswith(fresh_prefixes)]
    if shared_missing:
        raise ValueError(f"checkpoint lacks the co-trained action branch, e.g. {shared_missing[:4]}")
    unexpected = sd.keys() - expected.keys()
    if unexpected:
        raise ValueError(f"checkpoint has unexpected keys, e.g. {sorted(unexpected)[:6]}")
    fresh_missing = [k for k in expected.keys() - sd.keys() if k.startswith(fresh_prefixes)]
    if fresh_missing:
        emb, emb_cond = model.emb_in[modality].weight, model.emb_in[f"{modality}_cond"].weight
        fresh = fresh_head_state_dict(emb.shape[0], modality, emb.shape[1], head_seed, emb_cond.shape[1])
        sd.update({k: fresh[k] for k in fresh_missing})
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)


def build_action_dit(
    params: JointSingleSeqParams,
    ckpt_path: str | None = None,
    *,
    modality: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    strict_heads: bool = False,
    head_seed: int = 0,
) -> JointSingleSeq:
    """Instantiate the DiT (``params`` already carry the action modality) and load ``ckpt_path``.

    ``ckpt_path=None`` -> random init (wiring tests, or a policy whose weights are loaded afterwards by
    the lerobot ``from_pretrained`` machinery).
    """
    with torch.device(device), default_dtype(dtype):
        model = JointSingleSeq(params)  # built directly in ``dtype``: no fp32 copy on the device
    if ckpt_path is not None:
        load_action_checkpoint(model, ckpt_path, modality, strict_heads=strict_heads, head_seed=head_seed)
    model.eval()
    return model


def head_parameter_names(model: nn.Module, modality: str) -> set[str]:
    """Parameters of the fresh embodiment heads (the 5x learning-rate group)."""
    prefixes = tuple(n + "." for n in fresh_module_names(modality))
    return {n for n, _ in model.named_parameters() if n.startswith(prefixes)}
