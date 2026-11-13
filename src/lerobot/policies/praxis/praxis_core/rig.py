# Vendored from the praxis repository (praxis/rig.py, the base-checkpoint tooling); import paths rewritten to the LeRobot package.
# Not vendored: `init_heads`, which writes a praxis checkpoint; the LeRobot form is
# `examples/praxis/init_heads.py`, which links the base shards and writes the heads next to them.
"""Make a single-arm checkpoint of your rig from the base checkpoint.

The base checkpoint (``"embodiment": "base"``) holds the backbone and the
action expert. It has no tensors and no config fields of one rig. This module
adds them:

* :func:`single_arm_config` adds the rig fields to the base config: the
  action space and the pose frame of the single-arm rig, the patch dims of
  the expert, the tile canvas of the two cameras and the normalization.
* :func:`init_head_tensors` makes the six tensors of the rig
  (``BASE_HEAD_TENSORS``): the action input layer
  (``action_encoder.w1``), the proprio input layer (``embed_proprio``) and
  the action output layer (``head``).
* ``examples/praxis/init_heads.py`` writes the LeRobot policy directory of
  the rig from the base and these tensors; ``PraxisPolicy.from_pretrained``,
  ``lerobot-train`` and ``lerobot-eval`` read it like the single-arm
  checkpoint.

The rig must be a single arm with an exterior camera and a wrist camera, the
rig of ``praxis.single_arm``: the action is the flange pose and the gripper
(10 values per row: ``xyz3``, ``rot6d``, ``grip``). Only the normalization,
the tile size and the antialias of the canvas belong to your rig.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .action_expert import ActionExpert
from .single_arm import ACTION_DIM as ACTION_DIM, GRIP_DIMS, Q99_DIMS as _Q99_DIMS, VIEWS

# The geometry of the rig comes from praxis.single_arm, the serve path of the
# single-arm checkpoint, so the two cannot drift apart.

#: The action and the proprio of the single-arm rig: 10 values per row.
ACTION_SPACE: dict[str, object] = {
    "dim": ACTION_DIM,
    "layout": "[xyz(3) rot6d(6) grip(1)]",
    "pose_layout": "[xyz(3) quat_wxyz(4) grip(1)]",
}
#: The action dims with the q01/q99 normalization: xyz and the gripper. The
#: rot6d dims stay as they are.
Q99_DIMS: list[int] = _Q99_DIMS.tolist()
#: The dim of a 10-D row that holds the gripper.
(GRIP_DIM,) = GRIP_DIMS.tolist()
#: The spatial patch of the canvas (VAE stride 16 x patch 2): the tile sides
#: must be multiples of it.
SPATIAL_PATCH = 32
ACTION_FORMULA = (
    "clip(2 * (a - q01) / (q99 - q01 + 1e-8) - 1, -1, 1) on q99_normed_dims, identity on the other dims"
)
PROPRIO_FORMULA = "(p - mean) / max(std, 1e-6) on all dims"
#: The init std of the output head.
HEAD_INIT_STD = ActionExpert.HEAD_INIT_STD

_NORM_VECTORS = ("action_q01", "action_q99", "proprio_mean", "proprio_std")


def normalization_from_stats(stats: Mapping[str, object], domain: str) -> dict:
    """Return the ``normalization`` section of the config from ``stats``.

    ``stats`` holds the four vectors of ``examples/praxis/compute_norm_stats.py``:
    ``action_q01``, ``action_q99``, ``proprio_mean`` and ``proprio_std``,
    10 values each. ``domain`` names the data (for example the dataset).
    """
    out: dict[str, object] = {"domain": str(domain)}
    for key in _NORM_VECTORS:
        vec = stats.get(key)
        if not isinstance(vec, list) or len(vec) != ACTION_DIM:
            raise ValueError(f"stats.{key} must be a list of {ACTION_DIM} numbers")
        if not all(math.isfinite(float(x)) for x in vec):
            raise ValueError(f"stats.{key} has a non-finite value")
        out[key] = [float(x) for x in vec]
    out["q99_normed_dims"] = list(Q99_DIMS)
    out["action_formula"] = ACTION_FORMULA
    out["proprio_formula"] = PROPRIO_FORMULA
    return out


def single_arm_config(
    base: Mapping[str, object],
    *,
    tile_height: int,
    tile_width: int,
    antialias: bool,
    normalization: Mapping[str, object] | None = None,
) -> dict:
    """Return the single-arm config of a base config.

    The model fields stay as they are. The function adds the rig fields: the
    embodiment, the action space, the flange pose frame, the patch dims of
    the expert (``ACTION_DIM`` times the action frames of a token, and
    ``ACTION_DIM`` times the proprio frames), the tile canvas of the two
    views and ``normalization``. It drops the ``heads`` note of the base.
    Without ``normalization`` the config is enough for the data code
    (``examples/praxis/compute_norm_stats.py``) but not for training or serving.
    """
    cfg = copy.deepcopy(dict(base))
    cfg.pop("heads", None)
    cfg["embodiment"] = "single_arm"
    chunking = cfg["chunking"]
    assert isinstance(chunking, dict)
    expert = cfg["action_expert"]
    assert isinstance(expert, dict)
    expert["action_patch_dim"] = ACTION_DIM * int(chunking["action_frames_per_token"])
    expert["proprio_patch_dim"] = ACTION_DIM * int(chunking.get("proprio_frames", 1))
    video = cfg.setdefault("video", {})
    assert isinstance(video, dict)
    th, tw = int(tile_height), int(tile_width)
    if th <= 0 or tw <= 0 or th % SPATIAL_PATCH or tw % SPATIAL_PATCH:
        raise ValueError(
            f"the tile {th}x{tw} must be positive multiples of {SPATIAL_PATCH} px "
            "(the VAE stride 16 times the patch 2)"
        )
    video.update(
        height=th,
        width=tw * len(VIEWS),
        views=list(VIEWS),
        canvas="tiles",
        antialias=bool(antialias),
        tile_bounds_rows_cols={v: [0, th, i * tw, (i + 1) * tw] for i, v in enumerate(VIEWS)},
        tile_height=th,
        tile_width=tw,
    )
    cfg["action_space"] = dict(ACTION_SPACE)
    cfg["pose_frame"] = "flange"
    if normalization is not None:
        cfg["normalization"] = dict(normalization)
    return cfg


def init_head_tensors(config: Mapping[str, object], seed: int = 0) -> dict[str, Tensor]:
    """Return new random tensors of the rig (``BASE_HEAD_TENSORS``) in bfloat16.

    The init is the init of a new ``ActionExpert``: the default init of
    ``nn.Linear`` for the two input layers, and a normal init with the std
    ``HEAD_INIT_STD`` and a zero bias for the output head. The output head is
    small, so the first steps of a fine-tune start from small action
    velocities (the loss starts near the variance of the target, not at a
    random large value).
    """
    expert = config["action_expert"]
    assert isinstance(expert, Mapping)
    dim = int(expert["dim"])
    a_dim = int(expert["action_patch_dim"])
    p_dim = int(expert["proprio_patch_dim"])
    # fork_rng restores the CPU state. Seed only the CPU generator:
    # torch.manual_seed would also seed the CUDA generators, which fork_rng
    # with devices=[] does not restore.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        w1 = nn.Linear(a_dim, dim)
        proprio = nn.Linear(p_dim, dim)
        head = nn.Linear(dim, a_dim)
        nn.init.normal_(head.weight, std=HEAD_INIT_STD)
        nn.init.zeros_(head.bias)
    out = {
        "action_expert.action_encoder.w1.weight": w1.weight,
        "action_expert.action_encoder.w1.bias": w1.bias,
        "action_expert.embed_proprio.weight": proprio.weight,
        "action_expert.embed_proprio.bias": proprio.bias,
        "action_expert.head.weight": head.weight,
        "action_expert.head.bias": head.bias,
    }
    return {k: v.detach().to(torch.bfloat16).contiguous() for k, v in out.items()}
