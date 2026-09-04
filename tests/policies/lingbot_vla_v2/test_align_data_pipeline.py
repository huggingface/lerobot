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

"""Data-side wiring for the native-depth / DINO-video distillation branch.

Pins the contract between the three pieces that future-frame sampling touches:

1. ``LingbotVLAV2Config.observation_delta_indices`` — [current, future] per camera
   at the upstream action-horizon spacing (config-level tests live in
   test_depth_alignment_support.py; the delta consequences are asserted here);
2. ``LingbotVLAV2FeatureTransformStep`` — a delta-sampled batch yields
   ``pil_images`` / ``future_pil_images`` (raw pre-Qwen [0,255] frames the frozen
   teachers consume), the state is sliced back to the current frame, and the
   ``future_video_effective_fps`` is synthesized exactly like the upstream
   dataset (fps / max(1, chunk_size - 1));
3. inference safety — a depth-aligned checkpoint must still accept plain
   single-frame [C, H, W] items (no temporal slicing).

The full-pipeline tests need a local Qwen3-VL *processor* (tokenizer + image
processor, no weights); they are skipped when neither the
``LINGBOT_VLA_V2_QWEN3VL`` path nor the HF cache holds one.
"""

import glob
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

DEFAULT_QWEN3VL = os.path.expanduser("~/lingbot/Qwen3-VL-4B-Instruct-proc")


def _hf_cache_processor() -> str | None:
    """A complete Qwen3-VL processor snapshot in the local HF cache, if any."""
    for pattern in (
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*/"),
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-*Instruct/snapshots/*/"),
    ):
        for candidate in sorted(glob.glob(pattern)):
            if os.path.isfile(os.path.join(candidate, "preprocessor_config.json")) and os.path.isfile(
                os.path.join(candidate, "tokenizer.json")
            ):
                return candidate
    return None


QWEN3VL_PATH = os.environ.get("LINGBOT_VLA_V2_QWEN3VL") or DEFAULT_QWEN3VL
if not os.path.isdir(QWEN3VL_PATH):
    QWEN3VL_PATH = _hf_cache_processor()
pytestmark = pytest.mark.skipif(
    not QWEN3VL_PATH or not os.path.isdir(QWEN3VL_PATH),
    reason="no local Qwen3-VL processor (set LINGBOT_VLA_V2_QWEN3VL or populate the HF cache)",
)


SO101_ROBOT_CONFIG = """
states:
  - observation.state.arm.position:
      origin_keys:
        - observation.state:
            start: 0
            end: 6
actions:
  - action.arm.position:
      origin_keys:
        - action:
            start: 0
            end: 6
      subtract_state: False
images:
  - observation.images.camera_top:
      origin_keys: observation.images.front
norm_stats: {norm_stats_path}
"""

CHUNK_SIZE = 50
NUM_CAMERAS = 1
IMG_SHAPE = (3, 480, 640)


def _make_step(tmp_path, *, use_future_image: bool, dataset_fps: int | None = None):
    from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import (
        LingbotVLAV2FeatureTransformStep,
    )

    norm_stats_path = tmp_path / "norm_stats.json"
    norm_stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "observation.state.arm.position": {"mean": [0.0] * 6, "std": [1.0] * 6},
                    "action.arm.position": {"mean": [0.0] * 6, "std": [1.0] * 6},
                }
            }
        )
    )
    robot_config_path = tmp_path / "so101_robot.yaml"
    robot_config_path.write_text(SO101_ROBOT_CONFIG.format(norm_stats_path=norm_stats_path))
    return LingbotVLAV2FeatureTransformStep(
        robot_config_path=str(robot_config_path),
        processor_path=QWEN3VL_PATH,
        chunk_size=CHUNK_SIZE,
        canonical_joints={"arm.position": 6},
        canonical_norm_type={"arm.position": "meanstd"},
        cameras=["camera_top"],
        use_depth_align=True,
        use_future_image=use_future_image,
        dataset_fps=dataset_fps,
    )


def _transition(batch_size: int = 2, *, future: bool, with_action: bool = True):
    """A collated batch shaped like LeRobot's delta-timestamps output."""
    observation: dict = {}
    if future:
        observation["observation.state"] = torch.randn(batch_size, 2, 6)
        observation["observation.images.front"] = torch.rand(batch_size, 2, *IMG_SHAPE) * 255.0
        observation["observation.images.front_is_pad"] = torch.zeros(batch_size, 2, dtype=torch.bool)
    else:
        observation["observation.state"] = torch.randn(batch_size, 6)
        observation["observation.images.front"] = torch.rand(batch_size, *IMG_SHAPE) * 255.0
    complementary = {"task": "pick up the red cube"}
    transition: dict = {"observation": observation, "complementary_data": complementary}
    if with_action:
        transition["action"] = torch.randn(batch_size, CHUNK_SIZE, 6)
    return transition


def test_delta_sampled_batch_produces_teacher_inputs(tmp_path):
    """Training batch: [B, T, ...] items -> pil_images/future_pil_images + current state."""
    step = _make_step(tmp_path, use_future_image=True, dataset_fps=30)
    out = step(_transition(future=True))

    obs = out["observation"]
    # Current-frame [C,H,W] raw float frames in [0, 255], canonical camera order.
    assert obs["pil_images"].shape == (2, NUM_CAMERAS, *IMG_SHAPE)
    assert obs["pil_images"].dtype.is_floating_point
    assert float(obs["pil_images"].min()) >= 0.0 and float(obs["pil_images"].max()) <= 255.0
    # Future frames are the *last* sampled frame per camera.
    assert obs["future_pil_images"].shape == (2, NUM_CAMERAS, *IMG_SHAPE)
    # State is sliced back to the current frame (2-frame sampling must not reach the model).
    assert obs["observation.state"].shape == (2, 55)
    # Upstream injection: fps / max(1, chunk_size - 1).
    assert obs["future_video_effective_fps"] == pytest.approx(30 / (CHUNK_SIZE - 1))


def test_future_frames_match_last_sampled_frame(tmp_path):
    """future_pil_images must carry the [-1] frame, not a duplicate of [0]."""
    step = _make_step(tmp_path, use_future_image=True, dataset_fps=30)
    transition = _transition(future=True)
    images = transition["observation"]["observation.images.front"]
    images[:, 0] = 0.0
    images[:, 1] = 200.0
    out = step(transition)
    obs = out["observation"]
    assert float(obs["pil_images"].max()) == pytest.approx(0.0, abs=1e-3)
    assert float(obs["future_pil_images"].min()) == pytest.approx(200.0, abs=1e-2)


def test_explicit_future_frame_offset_changes_effective_fps(tmp_path):
    step = _make_step(tmp_path, use_future_image=True, dataset_fps=30)
    step.future_frame_offset = 10
    out = step(_transition(future=True))
    assert out["observation"]["future_video_effective_fps"] == pytest.approx(30 / 10)


def test_missing_dataset_fps_skips_effective_fps_synthesis(tmp_path):
    """Without fps the teacher falls back to its config.yaml effective_fps."""
    step = _make_step(tmp_path, use_future_image=True, dataset_fps=None)
    out = step(_transition(future=True))
    assert "future_video_effective_fps" not in out["observation"]
    assert "future_pil_images" in out["observation"]


def test_inference_items_are_never_temporally_sliced(tmp_path):
    """A depth-aligned checkpoint at rollout sees single [C,H,W] frames: the
    pad_and_concat split must not run on them."""
    step = _make_step(tmp_path, use_future_image=True, dataset_fps=30)
    transition = _transition(batch_size=1, future=False, with_action=False)
    out = step(transition)
    obs = out["observation"]
    assert obs["pil_images"].shape == (1, NUM_CAMERAS, *IMG_SHAPE)
    assert "future_pil_images" not in obs
    assert obs["observation.state"].shape == (1, 55)


def test_depth_only_branch_emits_current_frames_only(tmp_path):
    """use_future_image=False (no future depth / video): no future keys at all."""
    step = _make_step(tmp_path, use_future_image=False)
    out = step(_transition(future=False))
    obs = out["observation"]
    assert obs["pil_images"].shape == (2, NUM_CAMERAS, *IMG_SHAPE)
    assert "future_pil_images" not in obs
    assert "future_video_effective_fps" not in obs


def test_step_config_roundtrip():
    from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import (
        LingbotVLAV2FeatureTransformStep,
    )

    config = LingbotVLAV2FeatureTransformStep.get_config  # attribute exists on class
    assert callable(config)
    # get_config output must carry the new keys so saved preprocessors reload.
    signature_keys = set(LingbotVLAV2FeatureTransformStep.__dataclass_fields__)
    assert {"use_depth_align", "use_future_image", "dataset_fps", "future_frame_offset"} <= signature_keys
