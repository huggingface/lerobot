#!/usr/bin/env python

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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig
from .config_rebot_102_leader import RebotArm102LeaderConfig


@dataclass
class RebotArm102LeaderMetalConfig(RebotArm102LeaderConfig):
    """The reBot Arm 102 / Star Arm 102 leader mapped onto the Metal follower.

    Same physical leader as `RebotArm102LeaderConfig`, driven over the same servo bus with the
    same `joint_ids`. Only the joint *mapping* differs, because the parent's defaults describe
    the reBot B601 follower: point them at a Metal arm unchanged and four of seven joints are
    wrong, two of them silently non-functional (elbow_flex and gripper both saturate against the
    follower's soft limits and stop moving, while teleop keeps reporting a healthy 60 Hz loop).

    `joint_ranges` here is not a set of free parameters: it is `MetalFollowerConfig.joint_limits`
    copied verbatim, so the leader can never emit a target outside the follower's mechanical
    envelope. `test_metal_ranges_match_follower_limits` enforces that identity, with the gripper
    as the one deliberate exception (see below). Update the follower's limits and that test tells
    you to update these.
    """

    # Sign flips travel; magnitude rescales a leader joint onto a follower joint whose travel
    # differs. Only three entries differ from the parent's B601 defaults:
    #   elbow_flex  +1.0 -> -0.667  flipped, and the leader's ~270 deg scaled onto the
    #                               follower's 180 deg of travel
    #   wrist_flex  +1.0 -> -1.0    flipped
    #   gripper     -6.0 -> +1.895  flipped, and the leader's ~61 deg of grip travel widened
    #                               onto the follower's 115 deg jaw range
    joint_directions: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": -1.0,
            "shoulder_lift": -1.0,
            "elbow_flex": -0.667,
            "wrist_flex": -1.0,
            "wrist_yaw": 1.0,
            "wrist_roll": -1.0,
            "gripper": 1.895,
        }
    )

    # Mirrors MetalFollowerConfig.joint_limits so leader output is bounded by the follower's own
    # soft limits before it ever reaches the bus. Note these bounds are load-bearing beyond
    # clipping: `_round_to_valid_range` centres the multi-turn unwrap window on (min+max)/2, so a
    # range borrowed from the wrong follower can land a joint on the wrong 360 deg branch.
    #
    # The gripper is the single intentional departure: the follower's limit is the vendor's
    # 0..2.4 rad (137.5 deg), but its own stroke table only documents jaw opening up to 116.4 deg,
    # so the last stretch is past the widest measured opening. Stop at 115.
    joint_ranges: dict[str, list[int]] = field(
        default_factory=lambda: {
            "shoulder_pan": [-160, 160],
            "shoulder_lift": [-180, 0],
            "elbow_flex": [0, 180],
            "wrist_flex": [-123, 81],
            "wrist_yaw": [-85, 85],
            "wrist_roll": [-145, 145],
            "gripper": [0, 115],
        }
    )


@TeleoperatorConfig.register_subclass("rebot_102_leader_metal")
@dataclass
class RebotArm102LeaderMetalTeleopConfig(TeleoperatorConfig, RebotArm102LeaderMetalConfig):
    """Registered configuration for the reBot Arm 102 leader driving a Metal follower."""

    pass
