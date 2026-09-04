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
from typing import Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available

from .configuration_lingbot_vla_v2 import (
    LingbotVLAV2Config,
    build_feature_transform_configs,
    resolve_robot_config_and_stats,
)

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

    robot_config_path: str | None = None
    norm_stats_path: str | None = None
    # Parsed contents of the robot config / normalization stats. When set, they take
    # precedence over the paths and are what ``get_config`` serializes, so a saved
    # checkpoint is self-contained and portable across machines.
    robot_config: dict | None = None
    norm_stats: dict | None = None
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
    # When set (e.g. "cuda"), camera images are uploaded to this device and run
    # through the HF image processor in one batched call, with the outputs staying
    # on-device for the vision tower. None keeps the per-camera CPU path.
    preprocess_device: str | None = None
    # Qwen3-VL specific token/vision handling (mirrors the policy config fields).
    use_qwen3_chat_template: bool = True
    return_image_grid_thw: bool = True
    qwen3vl_use_vision_boundaries: bool = True
    # Native-depth / DINO-video distillation branch: keep raw pre-Qwen-processor
    # camera frames (CHW float [0,255], canonical camera order) as ``pil_images``
    # in the batch, and — with ``use_future_image`` — also a ``future_pil_images``
    # tensor from the last sampled frame. Both feed the frozen teachers inside
    # ``LingbotVLAV2Policy.forward``; inference never consumes them.
    use_depth_align: bool = False
    use_future_image: bool = False
    # Dataset fps and future-frame spacing, used to synthesize the per-item
    # ``future_video_effective_fps`` for the DINO-video teacher exactly like the
    # upstream dataset does: fps / max(1, future_frame_offset), where the offset
    # defaults to chunk_size - 1. None for either disables synthesis and the
    # teacher falls back to the effective_fps in its config.yaml.
    dataset_fps: int | None = None
    future_frame_offset: int | None = None

    _feature_transform: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "transformers is required for LingbotVLAV2FeatureTransformStep. "
                "Install it with `pip install 'lerobot[lingbot_vla2]'`."
            )
        from .model_core.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
        from .preprocessing.feature_transform import FeatureTransform

        apply_lingbot_qwen3_vl_patch()
        processor = AutoProcessor.from_pretrained(
            self.processor_path,
            padding_side="right",
            max_pixels=self.image_max_pixels,
            min_pixels=self.image_min_pixels,
        )
        # Resolve the robot config / norm stats contents from the paths when they were
        # not supplied directly, so ``get_config`` can serialize a portable checkpoint.
        if self.robot_config is None and self.robot_config_path:
            import yaml

            with open(self.robot_config_path) as f:
                self.robot_config = yaml.safe_load(f)
        if self.norm_stats is None:
            import json

            stats_path = self.norm_stats_path
            if stats_path is None and self.robot_config:
                stats_path = self.robot_config.get("norm_stats")
            if stats_path:
                with open(stats_path) as f:
                    self.norm_stats = json.load(f)
        data_config, model_config = build_feature_transform_configs(self)
        self._feature_transform = FeatureTransform(
            robot_config_path=self.robot_config_path,
            data_config=data_config,
            model_config=model_config,
            processor=processor,
            chunk_size=self.chunk_size,
            norm_stats_path=self.norm_stats_path,
            robot_config=self.robot_config,
            norm_stats=self.norm_stats,
            preprocess_device=self.preprocess_device,
            use_depth_align=self.use_depth_align,
            use_future_image=self.use_future_image,
        )

    def _iter_items(self, observation: dict, action, task):
        """Yield the per-item dicts ``FeatureTransform.apply`` expects."""
        # Batch size from the state feature.
        batch_size = observation[OBS_STATE].shape[0]
        # ``*_is_pad`` columns (emitted whenever delta_timestamps sample multiple
        # frames) are per-frame flags, not camera tensors.
        image_keys = [
            k for k in observation if k.startswith("observation.images.") and not k.endswith("_is_pad")
        ]

        # The FeatureTransform runs on CPU (numpy stats + the Qwen image processor),
        # but Accelerate hands us batches already on the training device. Move each
        # per-item tensor to CPU here; the trailing DeviceProcessorStep re-uploads the
        # transformed, model-ready tensors to the accelerator device.
        def _cpu(x):
            return x.cpu() if isinstance(x, torch.Tensor) else x

        for i in range(batch_size):
            state = _cpu(observation[OBS_STATE][i])
            # Future-frame sampling stacks T frames on every observation key; the
            # policy state is the current frame only (images keep their [T,C,H,W]
            # so FeatureTransform can split current/future per camera).
            if self.use_future_image and state.ndim == 2:
                state = state[0]
            item: dict[str, Any] = {OBS_STATE: state}
            for k in image_keys:
                img = _cpu(observation[k][i])
                # LeRobot images are float CHW in [0, 1]; the Qwen image processor
                # expects [0, 255]. Scale only if clearly normalized.
                if img.dtype.is_floating_point and float(img.max()) <= 1.0 + 1e-4:
                    img = img * 255.0
                item[k] = img
            if action is not None:
                item[ACTION] = _cpu(action[i])
                # FeatureTransform.apply reads the pad mask from the raw action key
                # (``f"{org_actions[0]}_is_pad"``); fill it dynamically and let the
                # apply side fall back to an all-False mask when absent.
                org_actions = self._feature_transform.org_features["actions"]
                pad_key = f"{org_actions[0]}_is_pad" if org_actions else "action_is_pad"
                item[pad_key] = torch.zeros(self.chunk_size, dtype=torch.bool)
            # Task text can arrive as a list of strings, a collated tensor of indices,
            # or a plain scalar; normalize to a string for the chat template.
            if isinstance(task, torch.Tensor):
                t = task[i].item() if task.ndim > 0 else task.item()
            elif isinstance(task, (list, tuple)):
                t = task[i]
            else:
                t = task
            item["task"] = t if isinstance(t, str) else str(t)
            yield item, (action is None)

    def __call__(self, transition):
        self._current_transition = transition.copy()
        observation = self._current_transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            raise ValueError("LingbotVLAV2FeatureTransformStep requires an observation dict.")
        action = self._current_transition.get(TransitionKey.ACTION)
        complementary = self._current_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        task = complementary.get("task", DEFAULT_TASK)

        applied = [
            self._feature_transform.apply(item, policy_eval=policy_eval)
            for item, policy_eval in self._iter_items(observation, action, task)
        ]

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
        if self.use_depth_align:
            new_obs["pil_images"] = collated["pil_images"]
            # Single-frame (inference) items produce None per item, not a tensor —
            # only route real future frames so teachers never see a junk key.
            future = collated.get("future_pil_images") if self.use_future_image else None
            if isinstance(future, torch.Tensor):
                new_obs["future_pil_images"] = future
                if self.dataset_fps is not None:
                    offset = (
                        self.future_frame_offset
                        if self.future_frame_offset is not None
                        else max(1, self.chunk_size - 1)
                    )
                    new_obs["future_video_effective_fps"] = self.dataset_fps / max(1, offset)

        self._current_transition[TransitionKey.OBSERVATION] = new_obs
        if action is not None:
            self._current_transition[TransitionKey.ACTION] = collated["actions"]
        return self._current_transition

    def get_config(self) -> dict[str, Any]:
        # Serialize the parsed contents rather than the (machine-specific) paths so a
        # pushed checkpoint reloads anywhere.
        return {
            "robot_config": self.robot_config,
            "norm_stats": self.norm_stats,
            "processor_path": self.processor_path,
            "chunk_size": self.chunk_size,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "tokenizer_max_length": self.tokenizer_max_length,
            "canonical_joints": self.canonical_joints,
            "canonical_norm_type": self.canonical_norm_type,
            "cameras": self.cameras,
            "resize_imgs_with_padding": list(self.resize_imgs_with_padding),
            # Must round-trip: these cap the Qwen3-VL vision-token budget and change
            # the effective input resolution. Omitting them silently reset the reload
            # to defaults and mismatched the checkpoint's training resolution.
            "image_max_pixels": self.image_max_pixels,
            "image_min_pixels": self.image_min_pixels,
            "preprocess_device": self.preprocess_device,
            "use_qwen3_chat_template": self.use_qwen3_chat_template,
            "return_image_grid_thw": self.return_image_grid_thw,
            "qwen3vl_use_vision_boundaries": self.qwen3vl_use_vision_boundaries,
            "use_depth_align": self.use_depth_align,
            "use_future_image": self.use_future_image,
            "dataset_fps": self.dataset_fps,
            "future_frame_offset": self.future_frame_offset,
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
    per-slot ``norm_stats``), so this pipeline does not add a separate
    LeRobot normalizer.
    """
    resolve_robot_config_and_stats(config)
    if not config.robot_config:
        raise ValueError(
            "LingBot-VLA 2.0 requires `config.robot_config_path` (the per-embodiment "
            "robot config mapping raw features onto the canonical slots)."
        )

    feature_step = LingbotVLAV2FeatureTransformStep(
        robot_config_path=config.robot_config_path,
        norm_stats_path=config.norm_stats_path,
        robot_config=config.robot_config,
        norm_stats=config.norm_stats,
        processor_path=config.processor_path or config.tokenizer_path,
        chunk_size=config.chunk_size,
        max_state_dim=config.max_state_dim,
        max_action_dim=config.max_action_dim,
        tokenizer_max_length=config.tokenizer_max_length,
        canonical_joints=config.canonical_joints,
        canonical_norm_type=config.canonical_norm_type,
        cameras=config.canonical_cameras,
        resize_imgs_with_padding=tuple(config.resize_imgs_with_padding),
        image_max_pixels=config.image_max_pixels,
        image_min_pixels=config.image_min_pixels,
        preprocess_device=config.preprocess_device,
        use_qwen3_chat_template=config.use_qwen3_chat_template,
        return_image_grid_thw=config.return_image_grid_thw,
        qwen3vl_use_vision_boundaries=config.qwen3vl_use_vision_boundaries,
        use_depth_align=config.use_depth_align,
        use_future_image=config.use_future_image,
        dataset_fps=config.dataset_fps,
        future_frame_offset=config.future_frame_offset,
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


def make_lingbot_vla_v2_pre_post_processors_from_pretrained(
    config: LingbotVLAV2Config,
    pretrained_path: str,
    *,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    pretrained_revision: str | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load the processors saved alongside a LeRobot checkpoint.

    The LingBot pipeline carries no LeRobot normalizer / unnormalizer steps (slot
    mapping and normalization live inside the feature-transform step), so the generic
    normalizer overrides injected by the training script are filtered out here — the
    pipeline loader rejects override keys that match no step.
    """
    preprocessor_overrides = {
        key: value
        for key, value in (preprocessor_overrides or {}).items()
        if key in {"device_processor", "rename_observations_processor"}
    }
    postprocessor_overrides = {
        key: value for key, value in (postprocessor_overrides or {}).items() if key == "device_processor"
    }
    if "device_processor" not in postprocessor_overrides and "device_processor" in preprocessor_overrides:
        postprocessor_overrides["device_processor"] = preprocessor_overrides["device_processor"]

    # The saved feature-transform step carries the slot mapping / normalization stats of
    # the checkpoint's *source* embodiment. When fine-tuning on a new embodiment the
    # policy config's assets must win here too — explicit ``robot_config_path`` /
    # ``norm_stats_path`` first, the config's embedded contents as fallback (same rule
    # as ``resolve_robot_config_and_stats``) — otherwise the training preprocessor
    # silently keeps the source embodiment's mapping while the policy itself was
    # already re-resolved onto the new one.
    resolve_robot_config_and_stats(config)
    # Same config -> step parameter set as ``make_lingbot_vla_v2_pre_post_processors``
    # (paths excluded: their resolved contents are forwarded instead). Only fields the
    # config actually carries (non-None) override the checkpoint's saved values.
    feature_step_overrides: dict[str, Any] = {}
    for step_param, config_attr in (
        ("robot_config", "robot_config"),
        ("norm_stats", "norm_stats"),
        ("chunk_size", "chunk_size"),
        ("max_state_dim", "max_state_dim"),
        ("max_action_dim", "max_action_dim"),
        ("tokenizer_max_length", "tokenizer_max_length"),
        ("canonical_joints", "canonical_joints"),
        ("canonical_norm_type", "canonical_norm_type"),
        ("cameras", "canonical_cameras"),
        ("resize_imgs_with_padding", "resize_imgs_with_padding"),
        ("image_max_pixels", "image_max_pixels"),
        ("image_min_pixels", "image_min_pixels"),
        ("preprocess_device", "preprocess_device"),
        ("use_qwen3_chat_template", "use_qwen3_chat_template"),
        ("return_image_grid_thw", "return_image_grid_thw"),
        ("qwen3vl_use_vision_boundaries", "qwen3vl_use_vision_boundaries"),
        ("use_depth_align", "use_depth_align"),
        ("use_future_image", "use_future_image"),
        ("dataset_fps", "dataset_fps"),
        ("future_frame_offset", "future_frame_offset"),
    ):
        value = getattr(config, config_attr, None)
        if value is not None:
            feature_step_overrides[step_param] = value
    processor_path = config.processor_path or config.tokenizer_path
    if processor_path is not None:
        feature_step_overrides["processor_path"] = processor_path

    # GPU preprocessing default: when the rollout inference device is CUDA and nobody
    # explicitly configured preprocess_device (policy config, saved checkpoint, or an
    # override), default it to that device. The fast path (prepare_images_on_device) is
    # pure torch/torchvision — bit-exact vs the CPU path (bench/check_gpu_preprocess.py)
    # — and saves ~171ms of per-tick host preprocessing on the measured 4090 setup
    # (x86 shared-host CPU contention; on GB10 both paths measure ~5ms). Explicit
    # config always wins, and ``preprocess_device="cpu"`` is the documented opt-out
    # (keeps the original per-camera HF processor path).
    if "preprocess_device" not in feature_step_overrides and config.preprocess_device is None:
        dev_override = (preprocessor_overrides.get("device_processor") or {}).get("device")
        target_dev = dev_override or getattr(config, "device", None)
        if target_dev is not None and str(target_dev).startswith("cuda") and torch.cuda.is_available():
            feature_step_overrides["preprocess_device"] = target_dev

    if feature_step_overrides:
        preprocessor_overrides["lingbot_vla_v2_feature_transform"] = feature_step_overrides

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=preprocessor_config_filename,
        overrides=preprocessor_overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        revision=pretrained_revision,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=postprocessor_config_filename,
        overrides=postprocessor_overrides,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        revision=pretrained_revision,
    )
    return preprocessor, postprocessor
