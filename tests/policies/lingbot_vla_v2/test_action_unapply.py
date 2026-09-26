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

"""Processor-free tests for the LingBot-VLA 2.0 postprocessor action path.

These exercise the inverse slot mapping
(``LingbotVLAV2InverseSlotMappingProcessorStep``) paired with the standard
``UnnormalizerProcessorStep`` — the path that turns the model's normalized 55-D
canonical chunks back into raw robot actions — on plain CPU without a Qwen3-VL
processor or checkpoint.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.lerobot_types import TransitionKey  # noqa: E402
from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import (  # noqa: E402
    LingbotVLAV2InverseSlotMappingProcessorStep,
    _raw_stats_from_slot_stats,
)
from lerobot.processor import UnnormalizerProcessorStep  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


def _so101_robot_config():
    return {
        "states": [
            {
                "observation.state.arm.position": {
                    "origin_keys": [{"observation.state": {"start": 0, "end": 6}}]
                }
            },
        ],
        "actions": [
            {"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 6}}]}},
        ],
        "images": [
            {"observation.images.camera_top": {"origin_keys": "observation.images.front"}},
        ],
    }


def _canonical_joints():
    return {"arm.position": 6, "reserved.slots": 4}


def test_action_unapply_denormalizes():
    """Inverse slot mapping + unnormalize recover the raw action chunk."""
    mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    std = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    inverse = LingbotVLAV2InverseSlotMappingProcessorStep(
        robot_config=_so101_robot_config(),
        canonical_joints=_canonical_joints(),
    )
    unnormalizer = UnnormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        norm_map={"ACTION": NormalizationMode.MEAN_STD},
        stats={"action": {"mean": mean, "std": std}},
    )

    raw = torch.arange(18, dtype=torch.float32).reshape(3, 6)  # known raw action chunk
    normalized = (raw - torch.tensor(mean)) / (torch.tensor(std) + 1e-8)
    # 10-D canonical chunk (6 arm + 4 reserved): real joints first, rest padding.
    chunk = torch.zeros(3, 10)
    chunk[:, :6] = normalized

    transition = inverse({TransitionKey.ACTION: chunk})
    recovered = unnormalizer(transition)[TransitionKey.ACTION]
    assert recovered.shape == (3, 6)
    torch.testing.assert_close(recovered, raw, atol=1e-4, rtol=1e-4)


def test_inverse_slot_mapping_handles_single_step_chunks():
    """select_action feeds (B, max_action_dim) single-step actions through the same step."""
    inverse = LingbotVLAV2InverseSlotMappingProcessorStep(
        robot_config=_so101_robot_config(),
        canonical_joints=_canonical_joints(),
    )
    chunk = torch.randn(2, 10)
    raw = inverse({TransitionKey.ACTION: chunk})[TransitionKey.ACTION]
    assert raw.shape == (2, 6)
    torch.testing.assert_close(raw, chunk[:, :6])


def test_inverse_slot_mapping_without_action_spans_raises():
    inverse = LingbotVLAV2InverseSlotMappingProcessorStep(
        robot_config={"states": [], "actions": []},
        canonical_joints=_canonical_joints(),
    )
    with pytest.raises(ValueError, match="no action slot mapping"):
        inverse({TransitionKey.ACTION: torch.zeros(1, 10)})


def test_raw_stats_from_slot_stats_scatters_spans():
    """Checkpoint-embedded per-slot stats are rewritten into raw-feature space."""
    robot_config = {
        "states": [
            {
                "observation.state.arm.position": {
                    "origin_keys": [{"observation.state": {"start": 2, "end": 5}}]
                }
            },
        ],
        "actions": [
            {"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 3}}]}},
        ],
    }
    slot_stats = {
        "norm_stats": {
            "observation.state.arm.position": {"mean": [10.0, 11.0, 12.0], "std": [1.0, 2.0, 3.0]},
            "action.arm.position": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        }
    }
    raw = _raw_stats_from_slot_stats(robot_config, slot_stats)
    assert raw is not None
    # The state spans sit at raw dims [2:5]; the uncovered dims get identity stats.
    torch.testing.assert_close(
        torch.tensor(raw["observation.state"]["mean"]),
        torch.tensor([0.0, 0.0, 10.0, 11.0, 12.0]),
    )
    torch.testing.assert_close(
        torch.tensor(raw["observation.state"]["std"]),
        torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0]),
    )
    torch.testing.assert_close(torch.tensor(raw["action"]["mean"]), torch.zeros(3))


def test_raw_stats_absent_returns_none():
    """No embedded stats → None → the normalizer steps are inert (identity path)."""
    assert _raw_stats_from_slot_stats(_so101_robot_config(), None) is None
    assert _raw_stats_from_slot_stats(_so101_robot_config(), {"norm_stats": {}}) is None
