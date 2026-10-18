# Adapted from NVlabs/LongLive at 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9.
# Changes: LeRobot package imports and reference-only runtime integration.
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM research integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/src/fastwam/models/wan22/ar_mask.py
# License texts and provenance: see the Long-WAM _core/NOTICE.md.
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
# End Long-WAM attribution.

"""AR-WAM joint attention mask (teacher-forcing clean/noisy memory + action).

Builds the boolean attention mask consumed by `MoT.forward` (square `[S, S]`, `True`
= query may attend key). Token layout follows the MoT expert order ``["video","action"]``:

    doubled=True  : S = 2*f*P + Sa,  layout [ clean(t=0) | noisy(t=t_v) | action ]
    doubled=False : S =   f*P + Sa,  layout [ clean(t=0)               | action ]

This reproduces LongLive stage-1 teacher forcing for the video branch (clean memory is
block-causal full-history; the noisy prediction tokens read clean 0..i and their own
block) and LongWAMBase-Joint's action rule pointed at the clean memory half (action attends
to all clean tokens + full self-attention within the chunk; the video branch never
attends to action; the action never attends to noise).
"""

import torch


def _frame_allowed(
    num_latent_frames: int,
    num_frame_per_block: int,
    local_attn_size: int,
    sink_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Frame-level [f, f] bool: query frame may attend key frame (block-causal)."""
    fb = torch.arange(num_latent_frames, device=device)
    q_blk = (fb // num_frame_per_block).view(-1, 1)
    k_blk = (fb // num_frame_per_block).view(1, -1)
    allowed = k_blk <= q_blk  # block-causal
    if local_attn_size > 0:
        win_blocks = max(1, local_attn_size // num_frame_per_block)
        allowed &= k_blk > q_blk - win_blocks  # sliding window
    if sink_size > 0:
        allowed |= fb.view(1, -1) < sink_size  # sink frames always visible
    return allowed  # [f, f] bool


def build_longwam_mask(
    num_latent_frames: int,
    video_tokens_per_frame: int,
    action_seq_len: int,
    device: torch.device,
    num_frame_per_block: int = 1,
    local_attn_size: int = -1,
    sink_size: int = 0,
    doubled: bool = True,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    P = video_tokens_per_frame
    S_clean = num_latent_frames * P
    S_video = 2 * S_clean if doubled else S_clean
    S = S_video + action_seq_len
    mask = torch.zeros((S, S), dtype=torch.bool, device=device)

    allowed = _frame_allowed(num_latent_frames, num_frame_per_block, local_attn_size, sink_size, device)
    allowed_tok = allowed.repeat_interleave(P, 0).repeat_interleave(P, 1)  # [f*P, f*P]

    frame_id = torch.arange(num_latent_frames, device=device).repeat_interleave(P)  # [f*P]
    block_id = frame_id // num_frame_per_block
    same_block = block_id.view(-1, 1) == block_id.view(1, -1)  # [f*P, f*P]

    C = slice(0, S_clean)
    if doubled:
        Nz = slice(S_clean, S_video)
        A = slice(S_video, S)
        mask[C, C] = allowed_tok  # clean  -> clean : block-causal memory self-attn
        # clean -> noisy / clean -> action : stay False (memory never sees noise/action)
        mask[Nz, C] = allowed_tok  # noisy  -> clean : reads clean 0..i (LongLive C2)
        mask[Nz, Nz] = same_block  # noisy  -> noisy : own block only (LongLive C1)
        # noisy -> action : stays False
        mask[A, C] = True  # action -> ALL clean memory (D3)
        # action -> noisy : stays False (action reads clean memory, never noise)
        mask[A, A] = True  # action -> action : full attn within chunk (D3)
    else:
        A = slice(S_clean, S)
        mask[C, C] = allowed_tok  # clean self-attn (prefill)
        mask[A, C] = True  # action -> all clean memory
        mask[A, A] = True  # action -> action

    mask |= torch.eye(S, dtype=torch.bool, device=device)  # SDPA NaN guard: self always attends

    # Cheap always-on structural invariants.
    assert mask.shape == (S, S) and mask.dtype == torch.bool
    assert mask.any(dim=1).all()  # no all-False query row
    if doubled:
        assert not mask[C, Nz].any()  # clean !-> noisy
        assert not mask[C, A].any()  # clean !-> action
        assert not mask[A, Nz].any()  # action !-> noisy
        assert mask[A, C].all() and mask[A, A].all()  # action -> all clean + own chunk
    return mask if dtype == torch.bool else mask.to(dtype)
