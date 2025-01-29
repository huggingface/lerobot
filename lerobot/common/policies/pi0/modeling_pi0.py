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

import json
import math
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0_paligemma import PI0PaliGemmaModel
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_dtype
from lerobot.configs.types import FeatureType


def display(tensor: torch.Tensor):
    """
    Display function for a PyTorch tensor that prints its shape, mean, std, min, and max.
    Args:
        tensor (torch.Tensor): The tensor to analyze and display.
    """
    if tensor.dtype == torch.bool:
        tensor = tensor.float()
    print(f"Shape: {tensor.shape}")
    print(f"Mean: {tensor.mean().item()}")
    print(f"Std: {tensor.std().item()}")
    print(f"Min: {tensor.min().item()}")
    print(f"Max: {tensor.max().item()}")


class PI0Policy(PreTrainedPolicy):
    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
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

        # checkpoint_dir = Path("/home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim")

        # with open(checkpoint_dir / "assets/norm_stats.json") as f:
        #     norm_stats = json.load(f)

        # num_motors = 14
        # dataset_stats = {
        #     "observation.state": {
        #         "mean": torch.tensor(norm_stats["norm_stats"]["state"]["mean"][:num_motors]),
        #         "std": torch.tensor(norm_stats["norm_stats"]["state"]["std"][:num_motors]),
        #         "min": torch.zeros(num_motors),
        #         "max": torch.ones(num_motors),
        #     },
        #     "action": {
        #         "mean": torch.tensor(norm_stats["norm_stats"]["actions"]["mean"][:num_motors]),
        #         "std": torch.tensor(norm_stats["norm_stats"]["actions"]["std"][:num_motors]),
        #         "min": torch.zeros(num_motors),
        #         "max": torch.ones(num_motors),
        #     },
        # }

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = PI0(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def init_ema_modules(self) -> dict[str, torch.optim.swa_utils.AveragedModel]:
        ema_decay = 0.99
        ema_names = [
            "model.pi0_paligemma.gemma_expert",
            "model.pi0_paligemma.action_in_proj",
            "model.pi0_paligemma.action_out_proj",
            "model.pi0_paligemma.action_time_mlp_in",
            "model.pi0_paligemma.action_time_mlp_out",
        ]

        self.ema_module_dict = {}
        for name, module in self.named_modules():
            if name not in ema_names:
                continue
            self.ema_module_dict[name] = torch.optim.swa_utils.AveragedModel(
                module,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
                use_buffers=False,
            )

        if len(self.ema_module_dict) != len(ema_names):
            raise ValueError("Some modules were not found.")

    def update_ema_modules(self):
        updated_module_names = []
        for name, module in self.named_modules():
            if name in self.ema_module_dict:
                self.ema_module_dict[name].update_parameters(module)
                updated_module_names.append(name)

        if len(updated_module_names) != len(self.ema_module_dict):
            raise ValueError(
                'Some EMA modules have not been updated. Possibly a mismatch in names between EMA module names and their original counterpart. EMA module names must have the same name as original module but ends with "_ema" (e.g  "linear_ema" and "linear").'
            )

    def state_dict_ema_modules(self):
        return {name: module.module.state_dict() for name, module in self.ema_module_dict.items()}

    def init_from_state_dict_ema(self):
        path = "outputs/train/2025-01-28/20-06-16_aloha_pi0/checkpoints/080000/pretrained_model/ema.pth"
        ema_state_dict = torch.load(path)
        for name, module in self.ema_module_dict.items():
            module.module.load_state_dict(ema_state_dict[name])

    def use_ema_modules(self):
        # self.init_ema_modules()
        # self.init_from_state_dict_ema()
        self.saved_state_dict = {}
        for name, module in self.named_modules():
            if name in self.ema_module_dict:
                self.saved_state_dict[name] = module.state_dict()
                module.load_state_dict(self.ema_module_dict[name].module.state_dict())
        return self.saved_state_dict

    def use_original_modules(self):
        for name, module in self.named_modules():
            if name in self.ema_module_dict:
                module.load_state_dict(self.saved_state_dict[name])
        del self.saved_state_dict

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise=None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model.inference(batch, noise=noise)[:, : self.config.n_action_steps]

            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        loss_dict = self.model.forward(batch)
        return loss_dict


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `batch_size`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta_gpu(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


class PI0(nn.Module):
    def __init__(self, config: PI0Config):
        super().__init__()
        self.config = config

        # tokenizer_path = "Tinkering/frostpunklab_23012024"
        # pi0_paligemma_path = "Tinkering/frostpunklab_full_float32"

        tokenizer_path = "/home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"
        pi0_paligemma_path = "/home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # self.pi0_paligemma = PI0PaliGemmaModel.from_pretrained(
        #     "Tinkering/frostpunklab_23012024", torch_dtype="bfloat16"
        # )
        self.pi0_paligemma = PI0PaliGemmaModel.from_pretrained(pi0_paligemma_path)
        # change important stuff in bf16
        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.pi0_paligemma.named_parameters():
            if any([selector in name for selector in params_to_change_dtype]):
                param.data = param.data.to(dtype=torch.bfloat16)

        if self.config.train_expert_only:
            self.pi0_paligemma.paligemma.eval()
            for params in self.pi0_paligemma.paligemma.parameters():
                params.requires_grad = False
            for params in self.pi0_paligemma.state_proj.parameters():
                params.requires_grad = False

        # pos_emb = create_sinusoidal_pos_embedding(n_action_steps, width, min_period=4e-3, max_period=4.0)
        # self.register_buffer("pos_emb", pos_emb.unsqueeze(0))
        self.torch_dtype = torch.bfloat16
        self._rng = torch.Generator()
        self._rng.manual_seed(42)  # Set an initial seed

    def train(self, mode: bool = True):
        # Apply train() to all submodules
        super().train(mode)

        if self.config.train_expert_only:
            # Explicitly set paligemma to eval mode
            self.pi0_paligemma.paligemma.eval()

    def prepare_images(self, batch):
        for key in self.config.image_features:
            if key not in batch:
                continue

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(batch[key], *self.config.resize_imgs_with_padding)

            bsize = batch[key].shape[0]
            device = batch[key].device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            batch[key] = img
            batch[f"{key}_mask"] = mask

        # TODO: remove HARDCODE
        for key in self.config.image_features:
            if key in batch:
                continue
            batch[key] = torch.ones_like(img) * -1
            batch[f"{key}_mask"] = torch.zeros_like(mask)
        return batch

    def prepare_language(self, batch):
        bsize = batch[OBS_ROBOT].shape[0]
        device = batch[OBS_ROBOT].device

        # tokenizer works on lists
        # PaliGemma prompt has to end with a new line
        max_length = 48
        if not hasattr(self, "tokenized_prompt"):
            tokenized_prompt = self.tokenizer.__call__(
                "Pick up yellow lego block and put it in the bin\n",
                padding="max_length",
                padding_side="right",
                max_length=max_length,
                return_tensors="pt",
            ).to(device=device)

            tokenized_prompt["attention_mask"] = tokenized_prompt["attention_mask"].type(dtype=torch.bool)

            self.tokenized_prompt = tokenized_prompt["input_ids"].repeat(bsize, 1).to(device=device)
            self.tokenized_prompt_mask = tokenized_prompt["attention_mask"].repeat(bsize, 1).to(device=device)

        batch["tokenized_prompt"] = self.tokenized_prompt
        batch["tokenized_prompt_mask"] = self.tokenized_prompt_mask
        return batch

    def prepare_state(self, batch):
        batch[OBS_ROBOT] = pad_vector(batch[OBS_ROBOT], self.config.state_dim)
        return batch

    def prepare_action(self, batch):
        actions = batch[ACTION]
        actions = pad_vector(actions, self.config.action_dim)
        return actions

    def forward(self, batch: dict[str, Tensor], noise=None):
        batch = self.prepare_images(batch)
        batch = self.prepare_state(batch)
        batch = self.prepare_language(batch)

        actions = self.prepare_action(batch)
        device = actions.device
        bsize = actions.shape[0]

        if noise is None:
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=actions.shape,
                dtype=torch.float32,
                device=device,
            )
        else:
            noise = noise.to(dtype=torch.float32, device=device)

        # time_beta = torch.distributions.Beta(1.5, 1).sample((bsize,))
        # compute directly sampling on GPU
        time_beta = sample_beta_gpu(1.5, 1.0, bsize, device)

        if self.config.fix_noise:
            noise = torch.load("../openpi/data/aloha_sim/noise_train.pth")
            noise = torch.from_numpy(noise).to(dtype=torch.float32, device=device)
            time_beta = torch.load("../openpi/data/aloha_sim/beta_time.pth")
            time_beta = torch.from_numpy(time_beta).to(dtype=torch.float32, device=device)

        time = time_beta * 0.999 + 0.001
        time = time.to(dtype=torch.float32, device=device)
        time_expanded = time[..., None, None]

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(batch)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(batch, x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.pi0_paligemma.forward(
            input_ids=None,
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.pi0_paligemma.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")

        if "action_is_pad" in batch:
            in_episode_bound = ~batch["action_is_pad"]
            losses = losses * in_episode_bound.unsqueeze(-1)

        # pi_loss = torch.from_numpy(torch.load("../openpi/data/aloha_sim/loss.pth"))

        real_num_motors = 6
        losses = losses[:, :, :real_num_motors]

        loss = losses.mean()

        loss_dict = {"l2_loss": loss.item(), "loss": loss}
        return loss_dict

    def inference(
        self, batch: dict[str, Tensor], noise=None
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        batch = self.prepare_images(batch)
        batch = self.prepare_state(batch)
        batch = self.prepare_language(batch)
        actions = self.sample_actions(batch, noise=noise)

        # unpad
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def sample_actions(self, batch, noise=None):
        bsize = batch[OBS_ROBOT].shape[0]
        # dtype = torch.bfloat16
        dtype = self.torch_dtype
        device = batch[OBS_ROBOT].device

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(batch)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        use_cache = True  # TODO: from config

        # fill image text cache
        fill_kv_cache = True
        _, past_key_values = self.pi0_paligemma.forward(
            input_ids=None,
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )
        fill_kv_cache = False

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        if noise is None:
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=(bsize, self.config.n_action_steps, self.config.action_dim),
                dtype=torch.float32,
                device=device,
            )
        else:
            noise = noise.to(dtype=torch.float32, device=device)

        x_t = noise
        time = 1.0
        time = torch.tensor(time, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            # time_batched = x_t[None, ...]
            expanded_time = time.expand(bsize)
            _, suffix_out = self.sample_step(
                batch,
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                past_key_values,
                fill_kv_cache,
                x_t,
                expanded_time,
            )
            suffix_out = suffix_out[:, -self.config.n_action_steps :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.pi0_paligemma.action_out_proj(suffix_out)

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def embed_prefix(self, batch):
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for key in self.config.image_features:
            # no finetuning of siglip
            # TODO: requires_grad = False for paligemma siglip to g
            img = batch[key]
            # normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0
            with torch.no_grad():
                img_emb = self.pi0_paligemma.paligemma.get_image_features(img)
            img_emb = img_emb.to(dtype=self.torch_dtype)

            img_emb_dim = img_emb.shape[-1]
            # img_emb = img_emb * math.sqrt(img_emb_dim)
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
            # img_emb *= img_emb_dim ** 0.5

            bsize, num_img_embs = img_emb.shape[:2]

            # img_mask = batch[f"{key}_mask"].expand(bsize, num_img_embs)
            img_mask = (batch[f"{key}_mask"])[:, None].expand(bsize, num_img_embs)
            # img_mask = torch.ones(bsize, num_img_embs, dtype=torch.bool, device=device)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # image tokens attend to each other
            att_masks += [0] * num_img_embs

        # TODO: if language
        lang_emb = self.pi0_paligemma.paligemma.language_model.model.embed_tokens(batch["tokenized_prompt"])
        # lang_emb = lang_emb.to(dtype=self.torch_dtype)

        # TODO: remove normalization?
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        # lang_emb = lang_emb * torch.tensor(lang_emb_dim**0.5, dtype=lang_emb.dtype)

        embs.append(lang_emb)
        pad_masks.append(batch["tokenized_prompt_mask"])

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, batch, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []

        # add a single state token

        # TODO (molbap): should be moved to the model backbone methods
        state_emb = self.pi0_paligemma.state_proj(batch["observation.state"])
        state_emb = state_emb.to(dtype=self.torch_dtype)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        # image/language inputs do not attend to state or actions
        att_masks += [1]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        width = self.config.action_expert_width
        time_emb = create_sinusoidal_pos_embedding(
            timestep, width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # mix timestep + action information using an MLP
        action_emb = self.pi0_paligemma.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        # TODO (molbap): should be moved to the model backbone methods

        action_time_emb = self.pi0_paligemma.action_time_mlp_in(action_time_emb)
        # action_time_emb = F.swish(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        # TODO (molbap): should be moved to the model backbone methods
        action_time_emb = self.pi0_paligemma.action_time_mlp_out(action_time_emb)

        # add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # image/language/state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def sample_step(
        self,
        batch,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        past_key_values,
        fill_kv_cache,
        x_t,
        timestep,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(batch, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        use_cache = True  # TODO: from config

        outputs_embeds, _ = self.pi0_paligemma.forward(
            input_ids=None,
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )
        return outputs_embeds


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


def resize_with_pad(img, width, height):
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
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=-1)
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


def main():
    import pickle

    # obs_path = "/raid/pablo/alohasim/obs.pkl"
    # action_path = "/raid/pablo/alohasim/action.pkl"
    # checkpoint_dir = Path("/raid/pablo/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim")
    # noise_bsize_2_path = "/raid/pablo/alohasim/noise_bsize_2.pkl"
    # noise_path = "/raid/pablo/alohasim/noise_2.pkl"
    # save_pretrained_path = "outputs/exported/2025-01-27/12-17-01_aloha_pi0/last/pretrained_model"

    obs_path = "../openpi/data/aloha_sim/obs.pkl"
    action_path = "../openpi/data/aloha_sim/action.pkl"
    checkpoint_dir = Path("/home/remi_cadene/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim")
    noise_bsize_2_path = "../openpi/data/aloha_sim/noise_bsize_2.pkl"
    noise_path = "../openpi/data/aloha_sim/noise_2.pkl"
    save_pretrained_path = "outputs/exported/2025-01-27/12-17-01_aloha_pi0/last/pretrained_model"

    with open(obs_path, "rb") as f:
        obs = pickle.load(f)

    with open(action_path, "rb") as f:
        pi_actions = torch.from_numpy(pickle.load(f)["actions"])

    with open(checkpoint_dir / "assets/norm_stats.json") as f:
        norm_stats = json.load(f)

    device = "cuda"
    num_motors = 14

    dataset_stats = {
        "observation.images.top": {
            "mean": torch.zeros(3, 1, 1),
            "std": torch.ones(3, 1, 1),
            "min": torch.zeros(3, 1, 1),
            "max": torch.ones(3, 1, 1),
        },
        "observation.state": {
            "mean": torch.tensor(norm_stats["norm_stats"]["state"]["mean"][:num_motors]),
            "std": torch.tensor(norm_stats["norm_stats"]["state"]["std"][:num_motors]),
            "min": torch.zeros(num_motors),
            "max": torch.ones(num_motors),
        },
        "action": {
            "mean": torch.tensor(norm_stats["norm_stats"]["actions"]["mean"][:num_motors]),
            "std": torch.tensor(norm_stats["norm_stats"]["actions"]["std"][:num_motors]),
            "min": torch.zeros(num_motors),
            "max": torch.ones(num_motors),
        },
    }

    cam_top = torch.from_numpy(obs["images"]["cam_high"]).unsqueeze(0) / 255.0  # * 2.0 - 1.0
    cam_top = cam_top.to(dtype=torch.float32)

    state = torch.from_numpy(obs["state"]).unsqueeze(0)
    state = state.to(dtype=torch.float32)

    gt_action = pi_actions.to(dtype=torch.float32)

    # Add bsize=2
    make_double_bsize = False
    if make_double_bsize:
        cam_top = torch.cat([cam_top, cam_top], dim=0)
        state = torch.cat([state, state], dim=0)
        noise = torch.load(noise_bsize_2_path)
        noise[1] = noise[0]
    else:
        noise = torch.load(noise_path)

    if not isinstance(noise, torch.Tensor):
        noise = torch.from_numpy(noise)

    batch = {
        "observation.images.top": cam_top,
        "observation.state": state,
        "action": gt_action.unsqueeze(0),
    }

    for k in batch:
        batch[k] = batch[k].to(device=device)

    ds_meta = LeRobotDatasetMetadata("lerobot/aloha_transfer_cube_human")
    features = dataset_to_policy_features(ds_meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = PI0Config(
        output_features=output_features,
        input_features=input_features,
        empty_cameras=2,
        fix_noise=True,
        train_expert_only=False,
    )

    policy = PI0Policy(cfg, dataset_stats=dataset_stats)

    policy.save_pretrained(save_pretrained_path)
    policy.to(device=device)

    loss_dict = policy.forward(batch)
    loss_dict["loss"].backward()

    actions = []
    for i in range(50):
        action = policy.select_action(batch, noise=noise)
        actions.append(action)

    actions = torch.stack(actions, dim=1)
    pi_actions = pi_actions.to(dtype=actions.dtype, device=actions.device)
    pi_actions = pi_actions.unsqueeze(0)
    print("actions")
    display(actions)
    print()
    print("pi_actions")
    display(pi_actions)
    print("atol=3e-2", torch.allclose(actions, pi_actions, atol=3e-2))
    print("atol=2e-2", torch.allclose(actions, pi_actions, atol=2e-2))
    print("atol=1e-2", torch.allclose(actions, pi_actions, atol=1e-2))


if __name__ == "__main__":
    main()
