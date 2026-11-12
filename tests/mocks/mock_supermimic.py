# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""A synthetic SuperMimic export (scripted linear actor + manifest) and reference clip."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lerobot.policies.supermimic.export import GOAL_FIELDS, OBSERVATION_FIELDS
from lerobot.policies.supermimic.robot_spec import g1_29dof_spec

HISTORY = 10
GOAL_FRAMES = 10


class LinearActor(nn.Module):
    """Deterministic stand-in: a fixed linear map of the newest observation and the goal."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.obs_weight = nn.Parameter(torch.randn(99, 29, generator=generator) * 1e-2)
        self.goal_weight = nn.Parameter(torch.randn(730, 29, generator=generator) * 1e-3)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return obs_hist[:, -1] @ self.obs_weight + goal @ self.goal_weight


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest(actor_sha: str = "", golden_sha: str = "") -> dict:
    spec = g1_29dof_spec()
    return {
        "schema_version": 1,
        "checkpoint": {"iteration": 0},
        "architecture": {
            "observation_dim": 99,
            "history_length": HISTORY,
            "goal_frames": GOAL_FRAMES,
            "goal_frame_dim": 73,
            "goal_dim": 73 * GOAL_FRAMES,
            "action_dim": 29,
            "normalization_included": True,
            "parameter_count": 99 * 29 + 730 * 29,
        },
        "inputs": {
            "obs_hist": {"history_order": "oldest first, newest last"},
            "observation_fields": [{"name": n, "slice": [a, b]} for n, a, b in OBSERVATION_FIELDS],
            "goal_fields_per_frame": [{"name": n, "slice": [a, b]} for n, a, b in GOAL_FIELDS],
        },
        "frames": {"rotation_6d_order": ["R00", "R01", "R10", "R11", "R20", "R21"]},
        "robot": {
            "joint_names": list(spec.joint_names),
            "default_q_rad": list(spec.default_q),
            "kp": list(spec.kp),
            "kd": list(spec.kd),
            "action_scale_rad": list(spec.action_scale),
            "control_dt_s": spec.control_dt,
            "physics_dt_s": spec.physics_dt,
            "nominal_calibration_offset_rad": [0.0] * 29,
        },
        "artifacts": {"actor": {"sha256": actor_sha}, "golden_vectors": {"sha256": golden_sha}},
    }


def make_export(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    actor = torch.jit.script(LinearActor().eval())
    actor.save(str(directory / "actor.ts"))
    generator = torch.Generator().manual_seed(1)
    golden = {}
    for case, batch in enumerate((1, 3)):
        obs_hist = torch.randn(batch, HISTORY, 99, generator=generator)
        goal = torch.randn(batch, 73 * GOAL_FRAMES, generator=generator)
        with torch.no_grad():
            golden[f"obs_hist_{case}"] = obs_hist.numpy()
            golden[f"goal_{case}"] = goal.numpy()
            golden[f"action_{case}"] = actor(obs_hist, goal).numpy()
    np.savez(directory / "golden_vectors.npz", **golden)
    manifest = make_manifest(_sha256(directory / "actor.ts"), _sha256(directory / "golden_vectors.npz"))
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def make_clip(path: Path, frames: int = 61, fps: float = 30.0, speed: float = 0.5, yaw: float = 0.3) -> Path:
    """A reference clip walking along x at ``speed`` m/s, rotated by ``yaw`` about z, in the default pose."""
    spec = g1_29dof_spec()
    t = np.arange(frames) / fps
    position = np.stack((speed * t * np.cos(yaw), speed * t * np.sin(yaw), np.full(frames, 0.79)), -1)
    quat_xyzw = np.tile([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], (frames, 1))
    joints = np.tile(spec.default_q, (frames, 1)) + 0.05 * np.sin(2 * np.pi * t)[:, None]
    np.savetxt(path, np.concatenate((position, quat_xyzw, joints), -1), delimiter=",")
    return path
