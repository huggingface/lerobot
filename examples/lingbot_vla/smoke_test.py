# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""End-to-end smoke test for the LingBot-VLA policy.

Loads the pretrained 4B checkpoint, feeds a synthetic observation (one camera
view + real-dim state + a task string) through the processor pipeline, runs
``select_action``, and prints the unnormalized action. Verifies the full
install -> load -> infer path on a single command.

Usage:
    python examples/lingbot_vla/smoke_test.py \
        --pretrained_path robbyant/lingbot-vla-4b --device cuda
"""

from __future__ import annotations

import argparse

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.lingbot_vla.configuration_lingbot_vla import LingbotVLAConfig
from lerobot.policies.lingbot_vla.modeling_lingbot_vla import LingbotVLAPolicy
from lerobot.policies.lingbot_vla.processor_lingbot_vla import (
    make_lingbot_vla_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_STATE

CAM = "observation.images.cam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LingBot-VLA smoke test")
    parser.add_argument("--pretrained_path", default="robbyant/lingbot-vla-4b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state_dim", type=int, default=6)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--task", default="pick up the red cube")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    config = LingbotVLAConfig(device=device, attention_implementation="eager")
    config.input_features = {
        CAM: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,))}

    dataset_stats = {
        OBS_STATE: {"mean": torch.zeros(args.state_dim), "std": torch.ones(args.state_dim)},
        ACTION: {"mean": torch.zeros(args.action_dim), "std": torch.ones(args.action_dim)},
    }

    print(f"Loading {args.pretrained_path} (strict) ...", flush=True)
    policy = LingbotVLAPolicy.from_pretrained(args.pretrained_path, config=config, strict=True)
    policy = policy.to(device=device, dtype=dtype)
    policy.eval()

    preprocessor, postprocessor = make_lingbot_vla_pre_post_processors(
        config=config, dataset_stats=dataset_stats
    )

    obs = {
        CAM: torch.rand(3, 480, 640),
        OBS_STATE: torch.randn(args.state_dim),
        "task": args.task,
    }

    batch = preprocessor(obs)
    with torch.no_grad():
        action = policy.select_action(batch)
        action = postprocessor(action)

    print(f"action shape: {tuple(action.shape)} device: {action.device}", flush=True)
    assert action.shape[-1] == args.action_dim, action.shape
    print("LINGBOT_VLA_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
