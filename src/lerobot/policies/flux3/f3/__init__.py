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
"""FLUX 3 Action model components used by the ``flux3`` policy.

``transformer`` (the JointSingleSeq DiT, vendored), ``positional`` (4-axis position ids), ``packing``
(observation -> tokens, noise, loss), ``sampling`` (Cosmos UniPC + Euler with two-pass CFG), ``wiring``
(action modality on the shared action branch, fresh heads, checkpoint key remap), ``text_encoder``
(Qwen3-VL-4B embedder) and ``video_vae`` (Video VAE; requires NATTEN).

Policy code imports these components through this package. Missing NATTEN is reported
when ``load_video_vae`` is called, so the other components remain importable without NATTEN.
"""

from . import packing, sampling
from .positional import batched_prc_audio, batched_prc_vid, times_to_ids
from .text_encoder import VEC_DIM, load_text_encoder, text_context
from .transformer import JointSingleSeq, JointSingleSeqParams
from .video_vae import load_video_vae
from .wiring import (
    REQUIRED_CONTENT_STREAMS,
    action_dit_params,
    build_action_dit,
    fresh_head_state_dict,
    fresh_module_names,
    restrict_content_streams,
    stream_of_key,
    unused_content_key,
)

__all__ = [
    "REQUIRED_CONTENT_STREAMS",
    "VEC_DIM",
    "JointSingleSeq",
    "JointSingleSeqParams",
    "action_dit_params",
    "batched_prc_audio",
    "batched_prc_vid",
    "build_action_dit",
    "fresh_head_state_dict",
    "fresh_module_names",
    "load_text_encoder",
    "load_video_vae",
    "packing",
    "restrict_content_streams",
    "sampling",
    "stream_of_key",
    "text_context",
    "times_to_ids",
    "unused_content_key",
]
