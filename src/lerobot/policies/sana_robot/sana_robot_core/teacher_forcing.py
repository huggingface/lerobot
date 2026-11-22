# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/policy/layout/teacher_forcing.py. Import paths are rewritten to the LeRobot package;
# the constants and helpers a streaming window reads; the teacher-forcing layout is not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

from collections.abc import Mapping

SINK_HORIZON_CHUNKS = 7


def resolve_actions_per_chunk(actions_per_chunk: int | Mapping[str, int], fps: float) -> int:
    """Return the control steps of one action chunk at the control frequency ``fps``."""
    if isinstance(actions_per_chunk, Mapping):
        return int(actions_per_chunk[str(int(round(float(fps))))])
    return int(actions_per_chunk)


def latent_centre_offset(vae_stride: int) -> float:
    """Return the middle of the ``vae_stride`` video frames that one target latent covers, in latent frames after the first of them: 0.5625 for a stride of 8."""
    return (int(vae_stride) + 1) / (2 * int(vae_stride))
