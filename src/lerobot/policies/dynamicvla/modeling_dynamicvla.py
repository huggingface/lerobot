#!/usr/bin/env python

# Copyright 2026 S-Lab and The HuggingFace Inc. team. All rights reserved.
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

import logging
import math
import os
import re
import time
from collections import deque

import safetensors
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F  # noqa: N812
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.utils import get_safe_dtype
from transformers import AutoConfig, SmolVLMForConditionalGeneration

from lerobot.policies.dynamicvla.configuration_dynamicvla import DynamicVLAConfig
from lerobot.policies.dynamicvla.modeling_fastvlm import (
    FastViTConfig,
    FastVLMConfig,
    FastVLMForConditionalGeneration,
)
from lerobot.policies.dynamicvla.modeling_vlm_with_expert import VLMWithExpertModel

# Matches ".soNNN", optionally followed by "-something", up to the "_buffer_" marker
_VARIANT_RE = re.compile(r"\.so\d+(?:-[\w]+)?_buffer_")


def canonicalise(k: str) -> str:
    """
    Remove dataset-variant markers like '.so100-blue_' or '.so100_' from a
    normalisation-buffer key.
    """
    return _VARIANT_RE.sub(".buffer_", k)


def standardise_state_dict(
    checkpoint: dict[str, torch.Tensor], ref_keys: set[str], *, verbose: bool = True
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    • Re-keys `checkpoint ` so that every entry matches the *reference* key set.
    • If several variant keys collapse to the same canonical name we keep the
      first one and log the collision.
    • Returns the new dict + a list of entries that could not be matched.
    """
    out, collisions, unmatched = {}, {}, []

    for k, v in checkpoint.items():
        canon = canonicalise(k)
        if canon in ref_keys:
            if canon in out:  # duplicate after collapsing
                collisions.setdefault(canon, []).append(k)
            else:
                out[canon] = v
        else:
            unmatched.append(k)

    if verbose:
        for canon, variants in collisions.items():
            logging.info(f"[standardise_state_dict] '{canon}'  ←  {variants}")
        if unmatched:
            logging.info(
                f"[standardise_state_dict] kept {len(unmatched)} unmatched keys"
            )

    out.update({k: checkpoint[k] for k in unmatched})
    return out, unmatched


def rename_checkpoint_keys(checkpoint: dict, rename_str: str):
    """
    Renames keys in a checkpoint dictionary based on the given rename string.

    Args:
        checkpoint (dict): The checkpoint dictionary.
        rename_str (str): A string specifying key mappings in the format "old1//new1,old2//new2".

    Returns:
        dict: The modified checkpoint with renamed keys.
    """

    rename_dict = dict(pair.split("//") for pair in rename_str.split(","))

    new_checkpoint = {}
    for k, v in checkpoint.items():
        for old_key, new_key in rename_dict.items():
            if old_key in k:
                k = k.replace(old_key, new_key)
        new_checkpoint[k] = v
    return new_checkpoint


def load_dynamicvla(
    model: torch.nn.Module,
    filename: str | os.PathLike,
    *,
    device: str = "cpu",
    checkpoint_keys_mapping: str = "",
) -> torch.nn.Module:
    state_dict = safetensors.torch.load_file(filename, device=device)

    # Optional user-supplied renames (e.g. "model._orig_mod.//model.")
    if checkpoint_keys_mapping and "//" in checkpoint_keys_mapping:
        state_dict = rename_checkpoint_keys(state_dict, checkpoint_keys_mapping)

    state_dict, _ = standardise_state_dict(state_dict, set(model.state_dict().keys()))

    # HACK(aliberts): to not overwrite normalization parameters as they should come from the dataset
    norm_keys = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(norm_keys)}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if not all(key.startswith(norm_keys) for key in missing) or unexpected:
        raise RuntimeError(
            "DynamicVLA %d missing / %d unexpected keys"
            % (len(missing), len(unexpected))
        )

    return model


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> torch.Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError(
            "The time torch.Tensor is expected to be of shape `(batch_size, )`."
        )

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with dynamicvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (
            2 * horn_radius * linear_position
        )
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by dynamicvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class DynamicVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = DynamicVLAConfig
    name = "dynamicvla"

    def __init__(
        self,
        config: DynamicVLAConfig,
        ckpt_filename: str | None = None,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            ckpt_filename: The file path of the pretrained model checkpoint. This is only used when
                `config.enable_streaming` is set to True, in which case a separate process
                is spawned to run the VLA model for streaming inference.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        # NOTE: APIs removed in the new lerobot version
        # self.normalize_inputs = Normalize(
        #     config.input_features, config.normalization_mapping, dataset_stats
        # )
        # self.normalize_targets = Normalize(
        #     config.output_features, config.normalization_mapping, dataset_stats
        # )
        # self.unnormalize_outputs = Unnormalize(
        #     config.output_features, config.normalization_mapping, dataset_stats
        # )
        self.model = VLAFlowMatching(config)
        self.language_tokenizer = self.model.vlm_with_expert.tokenizer
        self.reset()

        # ckpt_filename is used to initlialize the streaming process.
        # The variable is only set once in the get_streaming_model function.
        if config.enable_streaming and ckpt_filename is not None:
            # The initialization of the wrapper process of _inference_loop
            ctx = mp.get_context("spawn")
            self.q_in = ctx.Manager().dict()
            self.q_out = ctx.Queue(maxsize=1)
            self.worker = ctx.Process(
                target=self._inference_loop,
                args=(ckpt_filename, config, self.q_in, self.q_out),
            )
            self.worker.daemon = True
            self.worker.start()
            # Wait for the VLA model to be initialized
            _ = self.q_out.get()
            assert "initialized" in _

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        self._obs_index = 0
        self._action_index = 0
        if hasattr(self, "q_in"):
            self.q_in.clear()  # Clear the input queue
        if hasattr(self, "q_out") and not self.q_out.empty():
            self.q_out.get_nowait()  # Clear the output queue

    @staticmethod
    def get_streaming_model(pretrained_model: str, vla_cfg: DynamicVLAConfig):
        wrapper = DynamicVLAPolicy.from_pretrained(
            pretrained_model, config=vla_cfg, ckpt_filename=pretrained_model
        )
        # Remove unnecessary components to reduce VRAM
        del wrapper.language_tokenizer, wrapper.model
        torch.cuda.empty_cache()

        return wrapper

    @staticmethod
    @torch.no_grad()
    def _inference_loop(
        pretrained_model: str, vla_cfg: DynamicVLAConfig, q_in: dict, q_out: mp.Queue
    ):
        logging.basicConfig(
            level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s"
        )
        # Initialization
        vla_model = DynamicVLAPolicy.from_pretrained(pretrained_model, config=vla_cfg)
        vla_model.eval()
        if torch.cuda.is_available():
            vla_model = vla_model.cuda()
        # Warm-up (to accelerate the first inference)
        dummy_batch = {
            k: torch.zeros(
                1,
                vla_cfg.n_obs_steps,
                *vla_cfg.input_features[k].shape,
                dtype=torch.float32,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            for k in vla_cfg.input_features
        }
        dummy_batch["task"] = ["dummy text input"]
        vla_model._get_action_chunk(dummy_batch)
        q_out.put({"initialized": True})

        # The streaming inference loop
        while True:
            try:
                latest_obs = q_in.get("obs")
            except:
                latest_obs = None

            if latest_obs is None:
                continue

            q_in.clear()
            noise = latest_obs[1].cuda() if latest_obs[1] is not None else None
            batch = {}
            for k, v in latest_obs[0].items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda() if torch.cuda.is_available() else v
                else:
                    batch[k] = v

            index = batch["index"]
            latest_state = (
                batch[OBS_STATE][:, -1:, :]
                if batch[OBS_STATE].ndim > 2
                else batch[OBS_STATE]
            )
            batch = vla_model._prepare_batch(batch)

            actions = vla_model._get_action_chunk(batch, noise)
            if vla_model.config.use_delta_action:
                action_dim = actions.shape[-1] - 1
                actions[..., :action_dim] += latest_state[..., :action_dim]

            if q_out.full():
                logging.warning("The output queue is full. Skipping an action.")
                continue

            # NOTE: All torch.Tensors are on CPU if streaming is enabled. Because IPC with
            #       CUDA torch.Tensors is not supported.
            q_out.put_nowait({"actions": actions.transpose(0, 1).cpu(), "index": index})

    @classmethod
    def _load_as_safetensor(
        cls,
        model: "DynamicVLAPolicy",
        model_file: str,
        map_location: str,
        strict: bool,
    ):
        safetensors.torch.load_model(
            model, model_file, strict=strict, device=map_location
        )
        return load_dynamicvla(
            model,
            model_file,
            device=map_location,
            checkpoint_keys_mapping="model._orig_mod.//model.",
        )

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        tick = time.perf_counter()
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        # lang_tokens, lang_masks = self.prepare_language(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        # actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        inference_time = time.perf_counter() - tick
        if "dt_scale" in batch and batch["dt_scale"] > 1.0:
            # IMPORTANT: To align the inference time with the simulation time
            sleep_time = inference_time * (batch["dt_scale"] - 1)
            logging.info(
                "[Step%03d] Inference Time: %.4fs; Sleep Time: %.4fs"
                % (batch["index"], inference_time, sleep_time)
            )
            time.sleep(sleep_time)

        return actions

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        # return self.normalize_inputs(batch)
        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.config.enable_streaming:
            return self._get_streaming_action(batch, noise)
        else:
            return self._get_non_streaming_action(batch, noise)

    @torch.no_grad()
    def _get_streaming_action(
        self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        # NOTE: This function does not do any GPU computation.
        assert "index" in batch
        # Put the latest observation into the input dict
        self.q_in.update({"obs": (batch, noise)})

        actions = None
        if not self.q_out.empty():
            actions = self.q_out.get_nowait()

        # Merge actions into the queue
        if actions is not None:
            assert actions["actions"].size(0) == self.config.n_action_steps
            skip_n_actions = batch["index"] - actions["index"]
            logging.debug(
                "Curr. Step: %03d; Act. Step: %03d; Skip Steps: %03d"
                % (batch["index"], actions["index"], skip_n_actions)
            )

            actions["actions"] = actions["actions"][skip_n_actions:]
            actions["index"] += skip_n_actions

            prev_action_chunk = list(self._queues[ACTION])
            curr_action_chunk = [
                {"index": actions["index"] + i, "action": a}
                for i, a in enumerate(actions["actions"])
            ]
            if not prev_action_chunk:
                # The action queue is empty
                self._queues[ACTION].extend(
                    [a for a in curr_action_chunk if a["index"] > self._action_index]
                )
            else:
                self._queues[ACTION].clear()
                prev_index_start = prev_action_chunk[0]["index"]
                prev_index_end = prev_action_chunk[-1]["index"]
                curr_index_start = curr_action_chunk[0]["index"]
                if curr_index_start > prev_index_end:
                    self._queues[ACTION].extend(curr_action_chunk)
                elif curr_index_start > prev_index_start:
                    keeplen = curr_index_start - prev_index_end
                    self._queues[ACTION].extend(
                        prev_action_chunk[:keeplen] + curr_action_chunk
                    )
                else:  # if curr_index_start <= prev_index_start:
                    droplen = prev_index_start - curr_index_start
                    self._queues[ACTION].extend(curr_action_chunk[droplen:])

        if len(self._queues[ACTION]) == 0:
            return None
        else:
            action = self._queues[ACTION].popleft()
            self._action_index = action["index"]
            return action["action"]

    @torch.no_grad()
    def _get_non_streaming_action(
        self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.eval()
        # Save the state before normalization
        latest_state = (
            batch[OBS_STATE][:, -1:, :]
            if batch[OBS_STATE].ndim > 2
            else batch[OBS_STATE]
        )

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted,
        # populate it by querying the policy.
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)
            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim)
            # torch.Tensor, but the queue effectively has shape (n_action_steps, batch_size, *),
            # hence the transpose.
            if self.config.use_delta_action:
                action_dim = actions.shape[-1] - 1
                actions[..., :action_dim] += latest_state[..., :action_dim]

            self._queues[ACTION].extend(
                actions.transpose(0, 1)[: self.config.n_action_steps]
            )

        return self._queues[ACTION].popleft()

    def forward(
        self, batch: dict[str, torch.Tensor], noise=None, time=None
    ) -> dict[str, torch.Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        # batch = self.normalize_inputs(batch)
        # batch = self.normalize_targets(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        # lang_tokens, lang_masks = self.prepare_language(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        loss_dict["losses_after_forward"] = losses.clone()
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For backward pass
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply DynamicVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [
            key for key in self.config.image_features if key not in batch
        ]

        if len(present_img_keys) == 0:
            raise ValueError(
                "All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            imgs = batch[key][:, None, :, :, :] if batch[key].ndim == 4 else batch[key]
            b, n, c, h, w = imgs.shape
            assert n == self.config.n_obs_steps
            img = imgs.view(b, n * c, h, w)
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(
                    img, *self.config.resize_imgs_with_padding, pad_value=0
                )
            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(img.shape[0], dtype=torch.bool, device=img.device)

            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break

            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])

        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(
                actions[:, :, motor_idx]
            )
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(
                actions[:, :, motor_idx]
            )
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = (
            batch[OBS_STATE][:, -1, :]
            if batch[OBS_STATE].ndim > 2
            else batch[OBS_STATE]
        )
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a torch.Tensor along sequence dimension to match max_len.

    Args:
        torch.Tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = torch.Tensor.shape[:2]

    # Create a padded torch.Tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(torch.nn.Module):
    """
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: DynamicVLAConfig):
        super().__init__()
        self.config = config

        if config.temporal_fusion == "conv":
            self.mults_proj = torch.nn.Sequential(
                torch.nn.Conv2d(3 * config.n_obs_steps, 3, kernel_size=7, padding=3),
                torch.nn.GELU(),
            )
            vlm_input_channels = 3
        elif config.temporal_fusion == "attn":
            vlm_input_channels = 3 * config.n_obs_steps
        elif config.temporal_fusion == "flat":
            vlm_input_channels = 3
        else:
            raise ValueError(f"Unknown temporal_fusion: {config.temporal_fusion}")

        self.vlm_with_expert: VLMWithExpertModel = self._get_vlm_with_expert(
            config, config.vlm_model_name, vlm_input_channels
        )
        # Cached Image downscaling facto
        self.state_proj = torch.nn.Linear(
            config.max_state_dim,
            self.vlm_with_expert.vlm_config.text_config.hidden_size,
        )
        self.action_in_proj = torch.nn.Linear(
            config.max_action_dim, self.vlm_with_expert.expert_hidden_size
        )
        self.action_out_proj = torch.nn.Linear(
            self.vlm_with_expert.expert_hidden_size, config.max_action_dim
        )
        self.action_time_mlp_in = torch.nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2,
            self.vlm_with_expert.expert_hidden_size,
        )
        self.action_time_mlp_out = torch.nn.Linear(
            self.vlm_with_expert.expert_hidden_size,
            self.vlm_with_expert.expert_hidden_size,
        )
        self._set_requires_grad()

    def _get_vlm_with_expert(
        self,
        config: DynamicVLAConfig,
        vlm_model_name: str,
        vlm_input_channels: int,
    ):
        if vlm_model_name.startswith("HuggingFaceTB/SmolVLM2"):
            vlm_config = AutoConfig.from_pretrained(vlm_model_name)
            vlm_config.vision_config.num_channels = vlm_input_channels
            vlm_config.vision_config.patch_size = config.smolvlm_patch_size
            vlm_config.vision_config.num_attention_heads = (
                config.smolvlm_attention_heads
            )
            vlm_config.vision_config.hidden_size = config.smolvlm_hidden_size
            vlm_config.vision_config.intermediate_size = (
                config.smolvlm_intermediate_size
            )
            vlm = SmolVLMForConditionalGeneration(config=vlm_config)
        elif vlm_model_name.startswith("HuggingFaceTB/SmolLM2"):
            text_config = AutoConfig.from_pretrained(vlm_model_name)
            vision_config = FastViTConfig(
                in_channels=vlm_input_channels,
                position_embeddings=[
                    None,
                    None,
                    None,
                    {"name": "RepCPE", "spatial_shape": (7, 7)},
                    {"name": "RepCPE", "spatial_shape": (7, 7)},
                ],
                inference_mode=config.fastvlm_inference_mode,
            )
            vlm = FastVLMForConditionalGeneration(
                config=FastVLMConfig(
                    text_config=text_config,
                    vision_config=vision_config,
                )
            )
        else:
            raise ValueError(f"Unknown VLM: {vlm_model_name}")

        return VLMWithExpertModel(
            model_id=config.vlm_model_name,
            vlm=vlm,
            freeze_vision_model=config.freeze_vision_model,
            freeze_connector=config.freeze_connector,
            freeze_text_model=config.freeze_text_model,
            num_vlm_layers=self.config.num_vlm_layers,
            num_expert_layers=config.num_expert_layers,
            num_expert_skip_layers=config.num_expert_skip_layers,
            attention_mode=config.attention_mode,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
        )

    def _set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def _sample_noise(self, shape, device, dtype=torch.float32):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=dtype,
            device=device,
        )
        return noise

    def _sample_time(self, bsize, device, dtype=torch.float32):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=dtype)
        time = time_beta * 0.999 + 0.001
        return time

    def _embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []
        for img, img_mask in zip(images, img_masks, strict=False):
            bsize = img.size(0)
            if self.config.temporal_fusion == "conv":
                # Temporal fusion of image frames with Conv2d
                img = self.mults_proj(img)
            elif self.config.temporal_fusion == "flat":
                # Flatten temporal dimension into batch dimension
                img = img.view(-1, 3, img.shape[2], img.shape[3])

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb_dim = img_emb.size(-1)
            if self.config.temporal_fusion == "flat":
                # Reshape back to (batch_size, n_obs_steps * 3, emb_dim)
                img_emb = img_emb.view(bsize, -1, img_emb_dim)

            # Normalize image embeddings
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
            )
            num_img_embs = img_emb.size(1)
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * (num_img_embs)

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        if state is not None:
            state_emb = self.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            embs.append(state_emb)
            states_seq_len = state_emb.shape[1]
            state_mask = torch.ones(
                state_emb.shape[0],
                states_seq_len,
                dtype=torch.bool,
                device=state_emb.device,
            )
            pad_masks.append(state_mask)
            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1] * (states_seq_len)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.config.prefix_length:
            embs = pad_tensor(embs, self.config.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.config.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.config.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks

    def _embed_suffix(self, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device
        )
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def _get_position_ids(
        self, prefix_offsets: torch.Tensor | None, pad_masks: torch.Tensor
    ) -> torch.Tensor:
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        if prefix_offsets is not None:
            position_ids += prefix_offsets

        return position_ids

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
    ) -> torch.Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self._sample_noise(actions.shape, actions.device)

        if time is None:
            time = self._sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self._embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = self._get_position_ids(None, pad_masks)

        (_, suffix_out), _ = self.vlm_with_expert(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_vlm_embedding(
        self, images, img_masks, lang_tokens, lang_masks, state
    ) -> torch.Tensor:
        """Do a half inference forward and compute the VLM embedding"""

        prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = self._get_position_ids(None, prefix_pad_masks)
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        return prefix_pad_masks, past_key_values

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self._sample_noise(actions_shape, device)

        prefix_pad_masks, past_key_values = self.sample_vlm_embedding(
            images, img_masks, lang_tokens, lang_masks, state
        )
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            # Euler step
            x_t += dt * v_t
            time += dt

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self._embed_suffix(
            x_t, timestep
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = self._get_position_ids(prefix_offsets, suffix_pad_masks)

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
