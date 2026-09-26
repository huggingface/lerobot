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

"""End-to-end tests for the LingBot-VLA 2.0 standard processor pipeline.

Covers the full preprocessor (rename → batch-dim → relative actions → normalize
→ slot mapping → Qwen3-VL image → chat template → tokenize → device) and
postprocessor (inverse slot mapping → unnormalize → absolute actions → device)
against the real Qwen3-VL processor: a raw SO101-style item goes in, the padded
55-D canonical model inputs come out, and predicted canonical chunks come back
out as raw-dim actions.

Requires a local Qwen3-VL processor (config + tokenizer + image processor, not the
weights). Point ``LINGBOT_VLA_V2_QWEN3VL`` at it, or place it at the default path
below. Skipped otherwise (mirrors the weight-guarded tests of the v1 policy).
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import (  # noqa: E402
    LingbotVLAV2Config,
    SlotMapping,
)
from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import (  # noqa: E402
    make_lingbot_vla_v2_pre_post_processors,
    make_lingbot_vla_v2_pre_post_processors_from_pretrained,
)
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

DEFAULT_QWEN3VL = os.path.expanduser("~/lingbot/Qwen3-VL-4B-Instruct-proc")
QWEN3VL_PATH = os.environ.get("LINGBOT_VLA_V2_QWEN3VL", DEFAULT_QWEN3VL)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(QWEN3VL_PATH),
    reason=f"Qwen3-VL processor not found at {QWEN3VL_PATH}; set LINGBOT_VLA_V2_QWEN3VL.",
)

STATE_MEAN = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
STATE_STD = [2.0] * 6
ACTION_MEAN = [0.0] * 6
ACTION_STD = [4.0] * 6


def _so101_config() -> LingbotVLAV2Config:
    return LingbotVLAV2Config(
        processor_path=QWEN3VL_PATH,
        tokenizer_path=QWEN3VL_PATH,
        device="cpu",
        chunk_size=50,
        n_action_steps=50,
        max_state_dim=55,
        max_action_dim=55,
        tokenizer_max_length=72,
        state_slots={
            "observation.state.arm.position": SlotMapping(
                origin_keys=[{OBS_STATE: {"start": 0, "end": 6}}],
            ),
        },
        action_slots={
            "action.arm.position": SlotMapping(
                origin_keys=[{ACTION: {"start": 0, "end": 6}}],
                subtract_state=False,
            ),
        },
        camera_mapping={"observation.images.camera_top": "observation.images.front"},
        norm_stats={
            "norm_stats": {
                "observation.state.arm.position": {
                    "mean": STATE_MEAN,
                    "std": STATE_STD,
                },
                "action.arm.position": {"mean": ACTION_MEAN, "std": ACTION_STD},
            }
        },
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )


@pytest.fixture
def pipelines():
    return make_lingbot_vla_v2_pre_post_processors(_so101_config())


def _so101_item():
    return {
        OBS_STATE: torch.randn(6),
        ACTION: torch.randn(50, 6),
        "observation.images.front": torch.randint(0, 255, (3, 480, 640)).float(),
        "task": "pick up the red cube",
    }


def test_pipeline_is_composed_of_standard_and_custom_steps(pipelines):
    """Only the slot mapping (+ image / chat-template glue) is LingBot-specific."""
    preprocessor, postprocessor = pipelines
    assert [type(step).__name__ for step in preprocessor.steps] == [
        "RenameObservationsProcessorStep",
        "AddBatchDimensionProcessorStep",
        "RelativeActionsProcessorStep",
        "NormalizerProcessorStep",
        "LingbotVLAV2SlotMappingProcessorStep",
        "LingbotVLAV2ImageProcessorStep",
        "LingbotVLAV2ChatTemplateProcessorStep",
        "TokenizerProcessorStep",
        "DeviceProcessorStep",
    ]
    assert [type(step).__name__ for step in postprocessor.steps] == [
        "LingbotVLAV2InverseSlotMappingProcessorStep",
        "UnnormalizerProcessorStep",
        "AbsoluteActionsProcessorStep",
        "DeviceProcessorStep",
    ]
    # Embedded per-slot stats were rewritten into raw-feature stats.
    assert set(preprocessor.steps[3].stats) == {OBS_STATE, ACTION}


def test_preprocessor_maps_raw_item_to_model_inputs(pipelines):
    """A raw SO101 item becomes padded, Qwen3-VL-ready canonical model inputs."""
    preprocessor, _ = pipelines
    item = _so101_item()
    out = preprocessor(item)

    # State / action normalized in raw space, then padded to the 55-D canonical.
    assert out[OBS_STATE].shape == (1, 55)
    # The standard batch-dim step leaves 2D action chunks unbatched.
    assert out[ACTION].shape == (50, 55)
    expected_state = (item[OBS_STATE] - torch.tensor(STATE_MEAN)) / torch.tensor(STATE_STD)
    torch.testing.assert_close(out[OBS_STATE][0, :6], expected_state, atol=1e-4, rtol=1e-4)
    assert out[OBS_STATE][0, 6:].abs().sum() == 0  # padding

    # Joint masks mark the 6 real arm dims valid, the rest padding.
    assert out["state_joint_mask"].shape[-1] == 55
    assert out["action_joint_mask"].shape[-1] == 55
    assert int(out["state_joint_mask"][0].sum()) == 6
    assert int(out["action_joint_mask"][0].sum()) == 6

    # Native-resolution Qwen3-VL image tokens + grid for the mapped camera.
    assert out["images"].ndim == 4  # (B, n_views, num_patches, patch_dim)
    assert out["image_grid_thw"].shape[-1] == 3
    assert out["img_masks"].dtype == torch.bool
    # Only camera_top was mapped; the other two canonical views are zero-filled.
    assert int(out["img_masks"][0].sum()) == 1

    # Language: Qwen3 chat template tokenized to the padded max length.
    assert out[OBS_LANGUAGE_TOKENS].shape == (1, 72)
    assert out[OBS_LANGUAGE_ATTENTION_MASK].shape == (1, 72)


def test_postprocessor_roundtrips_raw_actions(pipelines):
    """Inverse slot mapping + unnormalize recover the raw action chunk."""
    preprocessor, postprocessor = pipelines
    item = _so101_item()
    raw_action = item[ACTION]
    normalized = preprocessor(item)[ACTION]

    # Postprocessor takes the model's predicted canonical chunk (tensor in, tensor out).
    recovered = postprocessor(normalized)
    assert recovered.shape == (50, 6)
    torch.testing.assert_close(recovered, raw_action, atol=1e-4, rtol=1e-4)


def test_saved_checkpoint_rebuilds_the_same_pipelines(pipelines, tmp_path):
    """save_pretrained + from_pretrained preserves the slot mapping and stats."""
    preprocessor, postprocessor = pipelines
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    # A config without explicit slot mappings must inherit the checkpoint's saved ones.
    config = LingbotVLAV2Config(
        processor_path=QWEN3VL_PATH,
        tokenizer_path=QWEN3VL_PATH,
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    pre2, post2 = make_lingbot_vla_v2_pre_post_processors_from_pretrained(config, str(tmp_path))

    item = _so101_item()
    out2 = pre2(item)
    ref = preprocessor(item)
    for key in (OBS_STATE, ACTION, OBS_LANGUAGE_TOKENS, "images", "image_grid_thw"):
        assert key in out2
        torch.testing.assert_close(out2[key], ref[key])
    # The reloaded normalizer kept the checkpoint's stats (not the inert default).
    assert set(pre2.steps[3].stats) == {OBS_STATE, ACTION}

    recovered = post2(ref[ACTION])
    torch.testing.assert_close(recovered, item[ACTION], atol=1e-4, rtol=1e-4)
