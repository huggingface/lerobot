#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""
π0: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi0.pdf)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.

Install pi0 extra dependencies:
```bash
pip install -e ".[pi0]"
```

Example of finetuning the pi0 pretrained model (`pi0_base` in `openpi`):
```bash
python lerobot/scripts/train.py \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of finetuning the pi0 neural network with PaliGemma and expert Gemma
pretrained with VLM default parameters before pi0 finetuning:
```bash
python lerobot/scripts/train.py \
--policy.type=pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of using the pi0 pretrained model outside LeRobot training framework:
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

"""

import math
from collections import deque
import os
import re
import safetensors
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoProcessor

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.normalize import (
    Normalize,
    NormalizePerRobotType,
    Unnormalize,
    UnnormalizePerRobotType,
)
from lerobot.policies.smolpi0.configuration_smolpi0 import SMOLPI0Config
from lerobot.policies.smolpi0.smolvlm_with_expert import (
    SmolVLMWithExpertModel
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import get_safe_dtype
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
ACTION = "action"
OBS_IMAGE_2 = "observation.image2"
OBS_IMAGE_3 = "observation.image3"
OBS_IMAGE_4 = "observation.image4"
TASK = "task"
ROBOT = "robot_type"
IMAGES_ORDER = {
    OBS_IMAGE: 0,
    OBS_IMAGE_2: 1,
    OBS_IMAGE_3: 2,
    OBS_IMAGE_4: 3,
}
from lerobot.policies.utils import (
    populate_queues,
)
import random
def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


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
            print(f"[standardise_state_dict] '{canon}'  ←  {variants}")
        if unmatched:
            print(f"[standardise_state_dict] kept {len(unmatched)} unmatched keys")

    out.update({k: checkpoint[k] for k in unmatched})
    return out, unmatched

def load_smolvla(
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
            "SmolVLA %d missing / %d unexpected keys",
            len(missing),
            len(unexpected),
        )

    return model
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
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)

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
def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
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

class SMOLPI0Policy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SMOLPI0Config
    name = "smolpi0"

    def __init__(
        self,
        config: SMOLPI0Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_per_robot_type = getattr(
            config, "normalize_per_robot_type", False
        )  # FIXME(mshukor): assert in case of single dataset
        if self.normalize_per_robot_type:
            if not dataset_stats:
                dataset_stats[config.robot_type] = {}
            self.normalize_inputs = NormalizePerRobotType(
                config.input_features, config.normalization_mapping, dataset_stats
            )
            self.normalize_targets = NormalizePerRobotType(
                config.output_features, config.normalization_mapping, dataset_stats
            )
            self.unnormalize_outputs = UnnormalizePerRobotType(
                config.output_features, config.normalization_mapping, dataset_stats
            )
        else:
            self.normalize_inputs = Normalize(
                config.input_features, config.normalization_mapping, dataset_stats
            )
            self.normalize_targets = Normalize(
                config.output_features, config.normalization_mapping, dataset_stats
            )
            self.unnormalize_outputs = Unnormalize(
                config.output_features, config.normalization_mapping, dataset_stats
            )

        self.language_tokenizer = AutoProcessor.from_pretrained(self.config.vlm_model_name).tokenizer
        self.model = VLAFlowMatching(config)
        self.include_past_states = config.n_obs_steps > 1 and OBS_STATE in self.config.past_obs_keys.split(",")
        self.include_past_images = config.n_obs_steps > 1 and "image" in self.config.past_obs_keys.split(",")
        self.num_past_images = self.config.n_obs_steps if self.include_past_images else 1
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        # self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.n_obs_steps > 1:
            for k in self.config.input_features:
                if any([past_obs_key in k for past_obs_key in self.config.past_obs_keys.split(",")]):
                    self._queues[k] = deque(maxlen=self.config.n_obs_steps)
    
    def get_optim_params(self) -> dict:
        if self.config.optimizer_lr_vlm > 0 and self.config.optimizer_lr_vlm != self.config.optimizer_lr:
            params = [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not ".vlm." in n and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if ".vlm." in n and p.requires_grad
                    ],
                    "lr": self.config.optimizer_lr_vlm,
                },
            ]
            return params
            
        else:
            return self.parameters()
            

    def merge_peft_model_weights(self) -> None:
        if "lora" in self.config.peft_method:
            self.model.vlm_with_expert.merge_lora_weights()

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        batch = self.normalize_inputs(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )
        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({"action": actions, "robot_type": batch["robot_type"]})["action"]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    # HACK(aliberts, danaaubakirova): we overwrite this classmethod here to fix smolVLA-specific issues
    @classmethod
    def _load_as_safetensor(
        cls,
        model: "SmolVLAPolicy",
        model_file: str,
        map_location: str,
        strict: bool,
        **kwargs,
    ):
        safetensors.torch.load_model(model, model_file, strict=strict, device=map_location)
        return load_smolvla(
            model,
            model_file,
            device=map_location,
            checkpoint_keys_mapping="model._orig_mod.//model.",
        )
    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        batch = self.normalize_inputs(batch)

        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            for k in batch:
                if k in self._queues:
                    batch[k] = torch.stack(list(self._queues[k]), dim=1)
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)
            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            )
            if self.config.predict_relative_actions and actions.ndim == 3:
                # If the model predicts relative actions, we need to unpad the actions
                # and then convert them to absolute actions.
                if self.config.relative_actions_mode == "first":
                    actions = torch.cat((actions[:, :1], actions[:, 1:] + actions[:, :1]), dim=1)
                elif self.config.relative_actions_mode == "state":
                    actions = actions + state.unsqueeze(1)
                else:
                    actions = torch.cat((actions[:, :1], actions[:, 1:] + actions[:, :-1]), dim=1)
            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        images, img_masks = self.prepare_images(
            batch
        )  # FIXME(mshukor): adapte it to take into account already padded images in the batch
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch, state=state)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.mean().clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.mean().clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.mean().clone()

        # For backward pass
        loss = losses.mean()
        # For backward pass
        loss_dict["loss"] = loss
        # # For logging
        # loss_dict["l2_loss"] = loss.item() # remove for torch compile
        return loss_dict

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        present_img_keys = sorted(present_img_keys, key=lambda k: IMAGES_ORDER.get(k, float("inf")), reverse=self.config.reverse_images_order)
        if self.config.shuffle_camera_positions and ACTION in batch: # only during training
            present_img_keys = random.sample(present_img_keys, len(present_img_keys))
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        for i in range(self.num_past_images):
            # Preprocess image features present in the batch
            for key in present_img_keys:
                img = batch[key][:, i, :, :, :] if batch[key].ndim == 5 else batch[key]
                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

                # Normalize from range [0,1] to [-1,1] as expacted by siglip
                img = img * 2.0 - 1.0

                bsize = img.shape[0]
                device = img.device
                if f"{key}_padding_mask" in batch:
                    mask = batch[f"{key}_padding_mask"].bool()
                else:
                    mask = torch.ones(bsize, dtype=torch.bool, device=device)
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

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_STATE].device
        tasks = batch["task"]
        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]

        if self.config.add_prompt_template:
            tasks = [f"{self.config.prefix_prompt_template}{task}{self.config.suffix_prompt_template}" for task in tasks]
        else:
            tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True, # FIXME(mshukor)
        )

        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

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
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if (batch[OBS_STATE].ndim > 2 and not self.include_past_states) else batch[OBS_STATE] # FIXME(mshukor): no state history for now
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch, state=None):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        if self.config.predict_relative_actions and actions.ndim == 3:
            if self.config.relative_actions_mode == "first":
                actions = torch.cat((actions[:, :1], actions[:, 1:] - actions[:, :1]), dim=1)
            elif self.config.relative_actions_mode == "state":
                assert batch[ACTION].shape[-1] == batch[OBS_STATE].shape[-1], "Relative action mode 'state' requires the action and state to have the same dimension."
                if state.ndim == 2:
                    state = state.unsqueeze(1)
                actions = actions - state
            else:
                actions = torch.cat((actions[:, :1], actions[:, 1:] - actions[:, :-1]), dim=1)
        return actions

def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    B, L = tensor.shape[:2]
    
    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full((B, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:, :L] = tensor  # Efficient in-place copy

    return padded_tensor

class VLAFlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │   VLM   │   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(model_id=self.config.vlm_model_name, 
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            self_attn_only_actions=self.config.self_attn_only_actions,
            )
        # self.paligemma_with_expert = self.configure_peft(paligemma_with_expert)
        self.vlm_with_expert.configure_peft(config=self.config)
        # Projections are float32
        self.state_to_prefix = self.config.state_to_prefix
        if self.state_to_prefix:
            self.state_proj = nn.Linear(self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size)
        else:
            self.state_proj = nn.Linear(self.config.max_state_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size)

        self.set_requires_grad()
        # SmolVLM2 has: [fake_tok + crop_tok + crop + fake_tok + crop_tok ... + fake_tok + global_tok + global + fake_tok] + [second image] + ...
        if  any([k in self.config.vlm_model_name for k in ["SmolVLM-", "SmolVLA-"]]):
            if "SmolVLM-Instruct" in self.config.vlm_model_name:
                self.fake_image_token =  49152
                self.global_image_token =  [44, 13906, 29, 6266, 46]
                self.global_image_start_token = torch.tensor([self.fake_image_token] + self.global_image_token, dtype=torch.long)
            else:
                self.fake_image_token =  49189
                self.global_image_token =  49152
                self.global_image_start_token = torch.tensor([self.fake_image_token, self.global_image_token], dtype=torch.long)
        else:
            self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
            self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
            self.global_image_start_token = torch.tensor([self.fake_image_token, self.global_image_token], dtype=torch.long)

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.add_local_special_image_tokens = self.config.add_local_special_image_tokens
        self.local_image_tokens = [torch.tensor([self.fake_image_token, tok], dtype=torch.long) for tok in [49153, 49154, 49155, 49159, 49160, 49161, 49165, 49166, 49167]] # assume 3 x 3 grid
        
        self.local_image_start_token = self.global_image_start_token
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.include_past_images = self.config.n_obs_steps > 1 and "image" in self.config.past_obs_keys.split(",")
        self.num_past_images = self.config.n_obs_steps if self.include_past_images else 1
        self.causal_attention_on_history = self.config.causal_attention_on_history
        
        
        

    # def configure_peft(self, model):
    #     # return model
    #     self.peft_method = self.config.peft_method
    #     if "lora" in self.peft_method:
    #         peft_config = self.config.peft_config
    #         target_modules = peft_config.target_modules
    #         if not isinstance(target_modules, list):
    #             target_modules = target_modules.split(",")
    #         lora_config = LoraConfig(
    #             task_type=TaskType.CAUSAL_LM,  # Based on the task type (e.g., language modeling, etc.)
    #             r=peft_config.r,  # The rank of the low-rank adaptation
    #             lora_alpha=peft_config.lora_alpha,  # Scaling factor
    #             lora_dropout=peft_config.lora_dropout,  # Dropout applied to LoRA layers
    #             target_modules=target_modules,  # The components where LoRA is applied
    #             exclude_modules=["gemma_expert", "model.gemma_expert.model.layers"], # FIXME(mshukor): this does not work for now
    #         )
    #         # LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=1, lora_dropout=0, target_modules=["q_proj"], exclude_modules=["gemma_expert"])
    #         self.lora_config = lora_config
    #         # Apply LoRA and ensure only LoRA parameters are trainable

    #         model = get_peft_model(model, lora_config)
    #         assert self.config.train_expert_only, "Backbone should be frozen and only lora parameters are " # FIXME(mshukor): handle this here?
    #         for name, param in model.named_parameters():
    #             if (
    #                 "lora" in name
    #             ):  # lm_head is not a parameter in most LLMs becasue it's tied to the embedding layer
    #                 param.requires_grad = True
    #     return model

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []
        num_images = len(images) // self.num_past_images
        # TODO: remove for loop
        for img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            # FIXME(mshukor):  add special tokens for the history each history_steps or not 
            if self.add_image_special_tokens:
                if self.add_local_special_image_tokens and img_idx % num_images != num_images - 1:
                    local_token_idx = img_idx % num_images
                    image_start_token = self.vlm_with_expert.embed_language_tokens(self.local_image_tokens[local_token_idx].to(device=self.vlm_with_expert.vlm.device)).unsqueeze(0).expand(img.shape[0], -1, -1)
                else:
                    image_start_token = self.vlm_with_expert.embed_language_tokens(self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)).unsqueeze(0).expand(img.shape[0], -1, -1)
                image_start_mask = torch.ones_like(image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device)
                if self.causal_attention_on_history and img_idx % num_images == 0:
                    att_masks += [1] + [0] * (image_start_mask.shape[-1] - 1)
                else:
                    att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb #.to(dtype=self.vlm_with_expert.type)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            # FIXME(mshukor): add special image tokens. Assume no tiling fake global images fake
            # template <|im_start|>User: What actions? image tokens \nAssistant: or processor.apply_chat_template?
            # processor.fake_image_token
            # processor.global_image_token

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                if not self.add_local_special_image_tokens or (self.add_local_special_image_tokens and img_idx % num_images == num_images - 1):
                    image_end_token = self.vlm_with_expert.embed_language_tokens(self.image_end_token.to(device=self.vlm_with_expert.vlm.device)).unsqueeze(0).expand(img.shape[0], -1, -1)
                    image_end_mask = torch.ones_like(image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device)
                    embs.append(image_end_token)
                    pad_masks.append(image_end_mask)
                    att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim) # FIXME(mshukor): is this needed for smolvlm?

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        if state is not None and self.state_to_prefix:
            state_emb = self.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb #.to(dtype=self.vlm_with_expert.type)
            embs.append(state_emb)
            bsize = state_emb.shape[0]
            dtype = state_emb.dtype
            device = state_emb.device

            states_seq_len = state_emb.shape[1]
            state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            # att_masks += [1] + [0]*(states_seq_len - 1)
            att_masks += [1]*(states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        if not self.state_to_prefix:
            state_emb = self.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb #.to(dtype=self.vlm_with_expert.type)
            embs.append(state_emb)
            bsize = state_emb.shape[0]
            dtype = state_emb.dtype
            device = state_emb.device

            states_seq_len = state_emb.shape[1]
            state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1] + [0]*(states_seq_len - 1)


        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.vlm_with_expert.expert_hidden_size, min_period=4e-3, max_period=4.0, device=device
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
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        if self.config.causal_action_attention_mask:
            att_masks += [1] * self.config.chunk_size
        else:
            att_masks += [1] + ([0] * (self.config.chunk_size - 1))
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        if self.config.regression_loss:
            # Hack to compare regression to flow matching
            time = torch.zeros_like(time, dtype=time.dtype, device=time.device)
            x_t = torch.zeros_like(actions, dtype=actions.dtype, device=actions.device)
            u_t = actions
        else:
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)
        
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
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
        if self.config.regression_loss:
            losses = F.l1_loss(u_t, v_t, reduction="none")
        else:
            losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        if self.config.regression_loss:
            x_t = torch.zeros_like(noise, dtype=torch.float32, device=device)
            expanded_time = torch.zeros(bsize, dtype=torch.float32, device=device)
            x_t = self.denoise_step(
                                state,
                                prefix_pad_masks,
                                past_key_values,
                                x_t,
                                expanded_time,
                            )
        else:
            dt = -1.0 / self.config.num_steps
            dt = torch.tensor(dt, dtype=torch.float32, device=device)

            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = self.denoise_step(
                    state,
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
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

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
