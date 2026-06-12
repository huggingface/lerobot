# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE

from .configuration_fastwam import FastWAMConfig


class FastWAMPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for FastWAM.

    Args:
        config (FastWAMConfig): FastWAM policy configuration.
        dataset_stats (dict[str, dict[str, Tensor]] | None): Optional LeRobot
            dataset statistics passed by the training/evaluation stack.
    """

    config_class = FastWAMConfig
    name = "fastwam"

    def __init__(
        self,
        config: FastWAMConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        self.model = self._build_core_model(config)
        self.reset()

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        """Shape-aware load that supports cross-embodiment fine-tuning.

        `safetensors.load_model(strict=False)` ignores missing/unexpected keys but
        still raises on a shape mismatch for a shared key. When fine-tuning from a
        checkpoint trained on a different embodiment (e.g. the LIBERO 7-DoF / 8-dim
        checkpoint adapted to a 6-DoF / 6-dim arm), the action encoder/head and
        proprio encoder legitimately differ in shape. With `strict=False` we drop
        only those shape-mismatched tensors — leaving them at their freshly
        initialized values — and load every compatible tensor. With `strict=True`
        the standard exact-match loader is used.
        """
        from safetensors import safe_open

        model_state_dict = model.state_dict()
        mismatched = []
        with safe_open(model_file, framework="pt") as f:
            checkpoint_keys = list(f.keys())
            for key in checkpoint_keys:
                if key in model_state_dict and tuple(model_state_dict[key].shape) != tuple(
                    f.get_slice(key).get_shape()
                ):
                    mismatched.append(key)

        if not mismatched:
            return super()._load_as_safetensor(model, model_file, map_location, strict)
        if strict:
            raise RuntimeError(
                f"FastWAM: {len(mismatched)} checkpoint tensors have a shape mismatch under "
                f"strict=True: {mismatched}"
            )

        from safetensors.torch import load_file

        logging.warning(
            "FastWAM cross-embodiment load: reinitializing %d shape-mismatched tensor(s), keeping "
            "every compatible weight: %s",
            len(mismatched),
            mismatched,
        )
        state_dict = load_file(model_file, device="cpu")
        for key in mismatched:
            state_dict.pop(key, None)
        model.load_state_dict(state_dict, strict=False)
        if map_location and map_location != "cpu":
            model.to(map_location)
        return model

    def get_optim_params(self) -> dict[str, Any]:
        params = (
            list(self.model.dit.parameters()) if hasattr(self.model, "dit") else list(self.model.parameters())
        )
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            params.extend(list(proprio_encoder.parameters()))
        return {"params": [p for p in params if p.requires_grad]}

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def _batch_to_training_sample(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Adapt a standard LeRobot batch to the FastWAM-native sample that
        `FastWAM.build_inputs` consumes (`video`, `action`, `context`/`context_mask`,
        per-frame `proprio`).

        The LeRobot training loop passes raw `observation.images.*`, a single-step
        `observation.state` `[B, D]`, `action`, and a language `task` string. We do
        only the translation `build_inputs` can't: stack the camera frames into a
        video, encode the prompt with the (frozen) text encoder (mirroring inference,
        so language-conditioned datasets need no precomputed context), and give proprio
        the per-frame axis `build_inputs` indexes. All shape/presence validation is
        left to `build_inputs`, the single authority on the contract.
        """
        sample = dict(batch)
        if "video" not in sample:
            sample["video"] = _stack_video_from_images(batch, self.config)
        if "context" not in sample or "context_mask" not in sample:
            prompt = _prompt_from_batch(batch=batch, config=self.config)
            if prompt is None:
                raise KeyError(
                    "FastWAM training requires a `task`/`prompt` to encode text context, "
                    "or precomputed `context`/`context_mask` in the batch."
                )
            sample["context"], sample["context_mask"] = self.model.encode_prompt(prompt)
        if self.config.proprio_dim is not None and "proprio" not in sample:
            state = sample.get(OBS_STATE)
            if state is not None:
                # LeRobot gives a single-step state [B, D]; build_inputs expects
                # per-frame [B, T, D] and uses frame 0, so add a T=1 axis.
                sample["proprio"] = state.unsqueeze(1) if state.ndim == 2 else state
        return sample

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute FastWAM training loss for a LeRobot batch.

        Args:
            batch (dict[str, Tensor]): Batch containing FastWAM-ready keys
                (`video`, `action`, `context`, `context_mask`) or LeRobot keys
                that can be adapted (`observation.images.*`, `observation.state`,
                `action`, `action_is_pad`).

        Returns:
            dict[str, Tensor]: Output dictionary containing the scalar `loss`
            key required by LeRobot and optional tensor metrics.
        """

        sample = self._batch_to_training_sample(batch)
        loss, metrics = self.model.training_loss(sample)
        output = {"loss": loss}
        for key, value in (metrics or {}).items():
            if isinstance(value, Tensor):
                output[key] = value.to(device=loss.device)
            else:
                output[key] = torch.as_tensor(value, device=loss.device)
        return output

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        """Predict a chunk of actions from the current FastWAM observation.

        Args:
            batch (dict[str, Tensor]): Inference batch with `input_image` or
                image observation keys, plus `context/context_mask` or `prompt`.

        Returns:
            Tensor: Action chunk with shape `[B, action_horizon, action_dim]`.
        """

        self.eval()
        infer_kwargs = _batch_to_infer_kwargs(batch=batch, config=self.config)
        batch_size = _infer_kwargs_batch_size(infer_kwargs)
        if batch_size == 1:
            action = _action_from_model_output(self.model.infer_action(**infer_kwargs))
        else:
            action = torch.cat(
                [
                    _action_from_model_output(
                        self.model.infer_action(
                            **_slice_infer_kwargs(infer_kwargs, index=i, batch_size=batch_size)
                        )
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )
        return action.to(device=batch_device(batch), dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _build_core_model(self, config: FastWAMConfig) -> torch.nn.Module:
        """Build the FastWAM core for training / inference.

        Only the trainable parts (the MoT DiT and the proprio encoder) are
        materialized empty here and then filled from the policy's
        `model.safetensors` by the base `from_pretrained`. The *frozen* Wan2.2 VAE
        and UMT5 text encoder are loaded with their real weights from the
        `Wan-AI/Wan2.2-TI2V-5B-Diffusers` repo (cached in the HF cache, shared
        across checkpoints) and are intentionally excluded from `model.safetensors`
        — see `FastWAM.__init__`. The tokenizer comes from `google/umt5-xxl`.
        """
        from .modular_fastwam import ActionDiT, FastWAM, MoT
        from .wan_components import (
            build_wan_tokenizer,
            load_pretrained_wan_text_encoder,
            load_pretrained_wan_vae,
        )
        from .wan_video_dit import WanVideoDiT

        dtype = _dtype_from_name(config.torch_dtype)
        device = config.device
        video_expert = WanVideoDiT(**config.video_dit_config).to(device=device, dtype=dtype)
        action_expert = ActionDiT(**config.action_dit_config).to(device=device, dtype=dtype)
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
        )
        text_encoder = (
            load_pretrained_wan_text_encoder(torch_dtype=dtype, device=device)
            if config.load_text_encoder
            else None
        )
        return FastWAM(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=load_pretrained_wan_vae(torch_dtype=dtype, device=device),
            text_encoder=text_encoder,
            tokenizer=build_wan_tokenizer(tokenizer_max_len=config.tokenizer_max_len),
            text_dim=int(config.video_dit_config["text_dim"]),
            proprio_dim=config.proprio_dim,
            device=device,
            torch_dtype=dtype,
            video_train_shift=float(config.video_scheduler["train_shift"]),
            video_infer_shift=float(config.video_scheduler["infer_shift"]),
            video_num_train_timesteps=int(config.video_scheduler["num_train_timesteps"]),
            action_train_shift=float(config.action_scheduler["train_shift"]),
            action_infer_shift=float(config.action_scheduler["infer_shift"]),
            action_num_train_timesteps=int(config.action_scheduler["num_train_timesteps"]),
            loss_lambda_video=float(config.loss["lambda_video"]),
            loss_lambda_action=float(config.loss["lambda_action"]),
        )


def _batch_to_infer_kwargs(batch: dict[str, Tensor], config: FastWAMConfig) -> dict[str, Any]:
    return {
        "prompt": _prompt_from_batch(batch=batch, config=config),
        "input_image": _input_image_from_batch(batch, config),
        "action_horizon": config.action_horizon,
        "proprio": batch.get("proprio", batch.get(OBS_STATE)),
        "context": batch.get("context"),
        "context_mask": batch.get("context_mask"),
        "negative_prompt": batch.get("negative_prompt", config.negative_prompt),
        "text_cfg_scale": float(batch.get("text_cfg_scale", config.text_cfg_scale)),
        "num_inference_steps": int(batch.get("num_inference_steps", config.num_inference_steps)),
        "sigma_shift": batch.get("sigma_shift", config.sigma_shift),
        "seed": batch.get("seed", config.inference_seed),
        "rand_device": batch.get("rand_device", config.rand_device),
        "tiled": bool(batch.get("tiled", config.tiled)),
    }


def _prompt_from_batch(batch: dict[str, Tensor], config: FastWAMConfig) -> Any:
    prompt = batch.get("prompt")
    if prompt is not None:
        return prompt

    task = batch.get("task")
    if task is None:
        return None
    if isinstance(task, str):
        return config.prompt_template.format(task=task)
    if isinstance(task, (list, tuple)):
        return [config.prompt_template.format(task=str(item)) for item in task]
    return config.prompt_template.format(task=str(task))


def _action_from_model_output(output: Any) -> Tensor:
    action = output["action"] if isinstance(output, dict) else output
    if action.ndim == 2:
        action = action.unsqueeze(0)
    return action


def _infer_kwargs_batch_size(infer_kwargs: dict[str, Any]) -> int:
    image = infer_kwargs["input_image"]
    if not isinstance(image, Tensor):
        raise TypeError(f"`input_image` must be a tensor, got {type(image).__name__}.")
    if image.ndim == 3:
        return 1
    if image.ndim == 4:
        return int(image.shape[0])
    raise ValueError(f"`input_image` must be [B,C,H,W] or [C,H,W], got {tuple(image.shape)}.")


def _slice_infer_kwargs(infer_kwargs: dict[str, Any], *, index: int, batch_size: int) -> dict[str, Any]:
    return {
        key: _slice_infer_value(value, index=index, batch_size=batch_size)
        for key, value in infer_kwargs.items()
    }


def _slice_infer_value(value: Any, *, index: int, batch_size: int) -> Any:
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index : index + 1]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]
    return value


def _dtype_from_name(name: str) -> torch.dtype:
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype `{name}`.")
    return dtype_map[name]


def batch_device(batch: dict[str, Any]) -> torch.device:
    for value in batch.values():
        if isinstance(value, Tensor):
            return value.device
    return torch.device("cpu")


def _stack_video_from_images(batch: dict[str, Tensor], config: FastWAMConfig) -> Tensor:
    image_keys = sorted(k for k in batch if k.startswith("observation.images."))
    if not image_keys:
        raise KeyError("FastWAM batch must contain `video` or `observation.images.*` keys.")
    images = [batch[key] for key in image_keys]
    image = torch.cat(images, dim=-1) if len(images) > 1 else images[0]
    if image.ndim == 4:
        image = image.unsqueeze(2).repeat(1, 1, config.num_video_frames, 1, 1)
    if image.ndim != 5:
        raise ValueError(f"Expected image batch [B,C,H,W] or video [B,C,T,H,W], got {tuple(image.shape)}.")
    return image


def _input_image_from_batch(batch: dict[str, Tensor], config: FastWAMConfig) -> Tensor:
    if "input_image" in batch:
        return _prepare_infer_image(batch["input_image"], config)
    video = batch.get("video")
    if video is None:
        video = _stack_video_from_images(batch, config)
    if video.ndim == 5:
        return _prepare_infer_image(video[:, :, 0], config)
    if video.ndim == 4:
        return _prepare_infer_image(video, config)
    raise ValueError(f"Cannot build input image from tensor with shape {tuple(video.shape)}.")


def _prepare_infer_image(image: Tensor, config: FastWAMConfig) -> Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor [B,C,H,W] or [C,H,W], got {tuple(image.shape)}.")

    target_h, target_w = config.image_size
    if tuple(image.shape[-2:]) != (target_h, target_w):
        raise ValueError(
            "FastWAM policy expects preprocessed image tensors with shape "
            f"[B,C,{target_h},{target_w}], got {tuple(image.shape)}. "
            "Run the FastWAM preprocessor before calling the policy."
        )
    return image
