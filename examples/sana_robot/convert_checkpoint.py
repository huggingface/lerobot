# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Convert a released SANA-Robot checkpoint to a LeRobot pretrained policy directory.

A released checkpoint is a folder with one ``.pth`` weights file and, when it is finetuned on a dataset,
``meta.json`` (the cameras, the robot slots, the prompt fields and the control frequency of the dataset) and
``normalization.json`` (a center and a scale per slot of the 80-slot robot row). The LeRobot directory holds
``config.json``, the weights as safetensors and the serialized pre- and post-processors.

    python examples/sana_robot/convert_checkpoint.py \\
        --checkpoint checkpoints/SANA-Robot-AR-LIBERO --out outputs/SANA-Robot-AR-LIBERO-LeRobot

The released files do not say which observation keys the cameras have, the size the policy reads them at, the
spatial layout or the sampling settings: the defaults are those of the LIBERO checkpoint, and the flags set them
for another one. The pretrained checkpoint holds weights only and gets no statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.sana_robot.configuration_sana_robot import SanaRobotConfig
from lerobot.policies.sana_robot.modeling_sana_robot import SanaRobotPolicy
from lerobot.policies.sana_robot.processor_sana_robot import make_sana_robot_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)

META_NAME = "meta.json"
NORMALIZATION_NAME = "normalization.json"


def weights_file(checkpoint: Path) -> Path:
    """Return the weights file of a released checkpoint, its only ``.pth`` file."""
    files = sorted(checkpoint.glob("*.pth"))
    if len(files) != 1:
        raise ValueError(
            f"{checkpoint} must hold exactly one .pth weights file, found {[f.name for f in files]}"
        )
    return files[0]


def build_config(checkpoint: Path, **overrides: Any) -> SanaRobotConfig:
    """Return the config of a released checkpoint: its ``meta.json`` when it has one, then ``overrides``."""
    fields: dict[str, Any] = {}
    meta_file = checkpoint / META_NAME
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text())
        fields = {
            "view_slots": [int(view["slot"]) for view in meta["views"]],
            "view_descriptions": [view["description"] for view in meta["views"]],
            "resize": meta.get("resize", "stretch"),
            "embodiment": meta["embodiment"],
            "action_mode": meta["action_mode"],
            "state_slots": [int(slot) for slot in meta["state_slots"]],
            "action_slots": [int(slot) for slot in meta["action_slots"]],
            "fps": float(meta["fps"]),
        }
    return SanaRobotConfig(**{**fields, **overrides})


def normalization_stats(
    checkpoint: Path, config: SanaRobotConfig
) -> dict[str, dict[str, torch.Tensor]] | None:
    """Return the statistics of ``observation.state`` and ``action`` from ``normalization.json``: the center and
    the scale of their slots, 0 and 1 on the slots that are not normalized. None without the file."""
    file = checkpoint / NORMALIZATION_NAME
    if not file.is_file():
        return None
    normalization = json.loads(file.read_text())
    (center, scale) = (torch.tensor(normalization["center"]), torch.tensor(normalization["scale"]))
    normalized = torch.tensor(normalization["mask"])
    stats = {}
    for key, slots in ((OBS_STATE, config.state_slots), (ACTION, config.action_slots)):
        stats[key] = {
            "mean": torch.where(normalized[slots], center[slots], 0.0).float(),
            "std": torch.where(normalized[slots], scale[slots], 1.0).float(),
        }
    return stats


def convert(
    checkpoint: str | Path, out: str | Path, device: str = "cuda", **overrides: Any
) -> SanaRobotConfig:
    """Write the LeRobot policy directory ``out`` of the released checkpoint folder ``checkpoint``.

    ``device`` is the device written into the config and ``overrides`` are fields of `SanaRobotConfig`. The
    weights keep the precision of the released file; the config's ``dtype`` is the precision the policy runs at.
    """
    checkpoint = Path(checkpoint)
    config = build_config(checkpoint, **overrides)
    policy = SanaRobotPolicy(replace(config, device="cpu", dtype=None))
    state = torch.load(weights_file(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    for key in ("state_dict_ema", "state_dict"):
        state = state.get(key, state)
    policy.model.load_state_dict(state, strict=True)
    # Set after `__post_init__`, which replaces a device this machine does not have.
    config.device = device
    policy.config = config
    policy.save_pretrained(out)
    stats = normalization_stats(checkpoint, config)
    for processor in make_sana_robot_pre_post_processors(config, stats):
        processor.save_pretrained(out)
    logger.info(
        "%s: %d state and %d action slots, %s",
        out,
        len(config.state_slots),
        len(config.action_slots),
        "with statistics" if stats else "weights only, no statistics",
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="the released checkpoint folder")
    parser.add_argument("--out", required=True, help="the LeRobot policy directory to write")
    parser.add_argument("--device", default="cuda", help="the device written into the config")
    parser.add_argument(
        "--camera-keys", default=None, help="comma list of the camera keys, in checkpoint order"
    )
    parser.add_argument(
        "--view-sizes", default=None, help='JSON list of (height, width), e.g. "[[256, 256]]"'
    )
    parser.add_argument("--spatial-layout", default=None, choices=("2x2", "t_shape"))
    parser.add_argument("--n-action-steps", type=int, default=None, help="control steps between two chunks")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="denoising steps per chunk")
    parser.add_argument("--video-cfg-scale", type=float, default=None, help="guidance of the video velocity")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides: dict[str, Any] = {}
    if args.camera_keys:
        overrides["camera_keys"] = [key.strip() for key in args.camera_keys.split(",") if key.strip()]
    if args.view_sizes:
        overrides["view_sizes"] = json.loads(args.view_sizes)
    for name in ("spatial_layout", "n_action_steps", "num_inference_steps", "video_cfg_scale"):
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    convert(args.checkpoint, args.out, device=args.device, **overrides)


if __name__ == "__main__":
    main()
