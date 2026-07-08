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

"""LingBot-VLA 2.0 policy processor.

Unlike the v1 policy (which reimplemented image/language handling as granular
LeRobot steps), v2 wraps the faithful upstream ``FeatureTransform`` in a single
step so the robot-config slot mapping, per-slot normalization, canonical padding,
Qwen3-VL native-resolution image tokens (``image_grid_thw``) and language
tokenization stay exactly as trained.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available

from .configuration_lingbot_vla_v2 import LingbotVLAV2Config

if _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None

DEFAULT_TASK = "Execute the robot action."


def _collate(values: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-item tensors, right-padding 1-D ragged tensors (e.g. language)."""
    shapes = {tuple(v.shape) for v in values}
    if len(shapes) == 1:
        return torch.stack(values, dim=0)
    if all(v.ndim == 1 for v in values):
        max_len = max(v.shape[0] for v in values)
        fill = False if values[0].dtype == torch.bool else 0
        padded = []
        for v in values:
            out = torch.full((max_len,), fill, dtype=v.dtype, device=v.device)
            out[: v.shape[0]] = v
            padded.append(out)
        return torch.stack(padded, dim=0)
    raise ValueError(f"Cannot collate tensors with shapes {shapes}")


@dataclass
@ProcessorStepRegistry.register(name="lingbot_vla_v2_feature_transform")
class LingbotVLAV2FeatureTransformStep(ProcessorStep):
    """Run the LingBot-VLA 2.0 ``FeatureTransform`` over a (batched) transition.

    The batched observation/action are split per item, passed through
    ``FeatureTransform.apply`` (training) — which produces the canonical, padded,
    Qwen3-VL-ready tensors — then re-collated. ``FeatureTransform.unapply`` runs on
    the postprocessing side to map model actions back to the raw dataset keys.
    """

    robot_config_path: str
    norm_stats_path: str | None = None
    processor_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    chunk_size: int = 50
    max_state_dim: int = 55
    max_action_dim: int = 55
    tokenizer_max_length: int = 72
    canonical_joints: dict = field(default_factory=dict)
    canonical_norm_type: dict = field(default_factory=dict)
    cameras: list = field(default_factory=list)
    resize_imgs_with_padding: tuple = (224, 224)
    # Cap the Qwen3-VL image processor's dynamic-resolution token budget. Left uncapped,
    # a native 1080x1920 frame explodes to ~8k vision tokens -> an O(N^2) eager-attention
    # tensor that OOMs and does not match the checkpoint's training resolution. Qwen3-VL
    # uses 16px patches + 2x2 merge (=1024 px/token), so 1,048,576 px ~= 1024 tokens.
    image_max_pixels: int = 262144
    image_min_pixels: int = 131072

    _feature_transform: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "transformers is required for LingbotVLAV2FeatureTransformStep. "
                "Install it with `pip install 'lerobot[lingbot-v2]'`."
            )
        from .feature_transform import FeatureTransform
        from .qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch

        apply_lingbot_qwen3_vl_patch()
        processor = AutoProcessor.from_pretrained(
            self.processor_path,
            padding_side="right",
            trust_remote_code=True,
            max_pixels=self.image_max_pixels,
            min_pixels=self.image_min_pixels,
        )
        data_config = SimpleNamespace(
            joints=[f"{{'{k}': {v}}}" for k, v in self.canonical_joints.items()],
            norm_type=[f"{{'{k}': '{v}'}}" for k, v in self.canonical_norm_type.items()],
            cameras=list(self.cameras),
            img_size=self.resize_imgs_with_padding[0],
            chat_template="default",
            text_keys="task",
        )
        model_config = SimpleNamespace(
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            chunk_size=self.chunk_size,
            tokenizer_max_length=self.tokenizer_max_length,
            use_qwen3_chat_template=True,
            return_image_grid_thw=True,
            qwen3vl_use_vision_boundaries=True,
            resize_imgs_with_padding=tuple(self.resize_imgs_with_padding),
        )
        self._feature_transform = FeatureTransform(
            robot_config_path=self.robot_config_path,
            data_config=data_config,
            model_config=model_config,
            processor=processor,
            chunk_size=self.chunk_size,
            norm_stats_path=self.norm_stats_path,
        )

    def _iter_items(self, observation: dict, action, task):
        """Yield the per-item dicts ``FeatureTransform.apply`` expects."""
        # Batch size from the state feature.
        batch_size = observation[OBS_STATE].shape[0]
        image_keys = [k for k in observation if k.startswith("observation.images.")]
        # The FeatureTransform runs on CPU (numpy stats + the Qwen image processor),
        # but Accelerate hands us batches already on the training device. Move each
        # per-item tensor to CPU here; the trailing DeviceProcessorStep re-uploads the
        # transformed, model-ready tensors to the accelerator device.
        def _cpu(x):
            return x.cpu() if isinstance(x, torch.Tensor) else x

        for i in range(batch_size):
            item: dict[str, Any] = {OBS_STATE: _cpu(observation[OBS_STATE][i])}
            for k in image_keys:
                img = _cpu(observation[k][i])
                # LeRobot images are float CHW in [0, 1]; the Qwen image processor
                # expects [0, 255]. Scale only if clearly normalized.
                if img.dtype.is_floating_point and float(img.max()) <= 1.0 + 1e-4:
                    img = img * 255.0
                item[k] = img
            if action is not None:
                item[ACTION] = _cpu(action[i])
                item["action_is_pad"] = torch.zeros(self.chunk_size, dtype=torch.bool)
            item["task"] = task[i] if isinstance(task, (list, tuple)) else task
            yield item, (action is None)

    def __call__(self, transition):
        self._current_transition = transition.copy()
        observation = self._current_transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            raise ValueError("LingbotVLAV2FeatureTransformStep requires an observation dict.")
        action = self._current_transition.get(TransitionKey.ACTION)
        complementary = self._current_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        task = complementary.get("task", DEFAULT_TASK)

        applied = [self._feature_transform.apply(item, policy_eval=policy_eval)
                   for item, policy_eval in self._iter_items(observation, action, task)]

        collated: dict[str, Any] = {}
        for key in applied[0]:
            values = [a[key] for a in applied]
            collated[key] = _collate(values) if isinstance(values[0], torch.Tensor) else values

        # Route model-ready tensors into the observation; keep the padded action out.
        new_obs = dict(observation)
        new_obs["images"] = collated["images"]
        new_obs["img_masks"] = collated["img_masks"]
        new_obs["lang_tokens"] = collated["lang_tokens"]
        new_obs["lang_masks"] = collated["lang_masks"]
        new_obs[OBS_STATE] = collated["state"]
        new_obs["joint_mask"] = collated["joint_mask"]
        new_obs["state_joint_mask"] = collated["state_joint_mask"]
        new_obs["action_joint_mask"] = collated["action_joint_mask"]
        if "image_grid_thw" in collated:
            new_obs["image_grid_thw"] = collated["image_grid_thw"]

        self._current_transition[TransitionKey.OBSERVATION] = new_obs
        if action is not None:
            self._current_transition[TransitionKey.ACTION] = collated["actions"]
        return self._current_transition

    def unapply_actions(self, actions: torch.Tensor) -> dict:
        """Map a padded canonical action chunk back to the raw dataset keys."""
        return self._feature_transform.unapply({"actions": actions})

    def get_config(self) -> dict[str, Any]:
        return {
            "robot_config_path": self.robot_config_path,
            "norm_stats_path": self.norm_stats_path,
            "processor_path": self.processor_path,
            "chunk_size": self.chunk_size,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "tokenizer_max_length": self.tokenizer_max_length,
            "canonical_joints": self.canonical_joints,
            "canonical_norm_type": self.canonical_norm_type,
            "cameras": self.cameras,
            "resize_imgs_with_padding": list(self.resize_imgs_with_padding),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        obs = features[PipelineFeatureType.OBSERVATION]
        obs["lang_tokens"] = PolicyFeature(type=FeatureType.LANGUAGE, shape=(self.tokenizer_max_length,))
        obs["lang_masks"] = PolicyFeature(type=FeatureType.LANGUAGE, shape=(self.tokenizer_max_length,))
        return features


def make_lingbot_vla_v2_pre_post_processors(
    config: LingbotVLAV2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the LingBot-VLA 2.0 pre- and post-processing pipelines.

    Normalization + slot mapping live inside the feature-transform step (using the
    per-slot ``norm_stats`` JSON), so this pipeline does not add a separate
    LeRobot normalizer.
    """
    if not config.robot_config_path:
        raise ValueError(
            "LingBot-VLA 2.0 requires `config.robot_config_path` (the per-embodiment "
            "robot config mapping raw features onto the canonical slots)."
        )

    feature_step = LingbotVLAV2FeatureTransformStep(
        robot_config_path=config.robot_config_path,
        norm_stats_path=config.norm_stats_path,
        processor_path=config.processor_path or config.tokenizer_path,
        chunk_size=config.chunk_size,
        max_state_dim=config.max_state_dim,
        max_action_dim=config.max_action_dim,
        tokenizer_max_length=config.tokenizer_max_length,
        canonical_joints=config.canonical_joints,
        canonical_norm_type=config.canonical_norm_type,
        cameras=config.canonical_cameras,
        resize_imgs_with_padding=tuple(config.resize_imgs_with_padding),
    )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        feature_step,
        DeviceProcessorStep(device=config.device),
    ]
    output_steps: list[ProcessorStep] = [
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
