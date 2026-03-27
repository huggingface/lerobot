# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
SmolStar06: Advantage-Conditioned SmolVLA Policy

Wraps SmolVLA with RECAP-style advantage conditioning. A frozen value network
(trained separately) labels training data with per-sample advantages that are
binarized into "Advantage: positive/negative" tokens appended to the language
prompt. At inference, the model conditions on "Advantage: positive" to produce
higher-quality actions.

Two-phase workflow:
  1. Train a value network with RECAPTrainSmolVLANetwork (see RECAP_VALUE_NETWORK_TRAINING.md)
  2. Train SmolStar06Policy using the frozen value network weights
"""

import csv
import logging
from typing import Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.smolstar06.configuration_smolstar06 import SmolStar06Config
from lerobot.policies.smolvla.modeling_smolvla import ActionSelectKwargs, SmolVLAPolicy
from lerobot.rl.algorithms.RECAPSmolVLAValueNetwork import (
    RECAPSmolVLAValueNetwork,
    RECAPSmolVLAValueNetworkConfig,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


class SmolStar06Policy(SmolVLAPolicy):
    """Advantage-conditioned SmolVLA policy.

    During training, a frozen value network computes V(o_t) for each sample.
    The deterministic return R_t is derived from episode success/fail labels.
    Advantage A = R_t - V(o_t) is binarized and injected as language tokens.

    During inference, "Advantage: positive" is always appended to the prompt.
    Optional classifier-free guidance (cfg_beta > 1) sharpens the distribution
    by interpolating conditioned and unconditioned flow vectors.
    """

    config_class = SmolStar06Config
    name = "smolstar06"

    def __init__(
        self,
        config: SmolStar06Config,
        dataset_meta=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        self.value_network: RECAPSmolVLAValueNetwork | None = None
        if config.value_network_checkpoint:
            self._load_value_network(config.value_network_checkpoint)

        self._setup_advantage_tokens()

        self._episode_info: dict | None = None
        self._task_max_len: dict | None = None
        if dataset_meta is not None and config.episode_labels_path is not None:
            self._setup_episode_metadata(dataset_meta, config.episode_labels_path)

    def _load_value_network(self, checkpoint_path: str) -> None:
        """Load a pre-trained RECAP value network and freeze all parameters."""
        logging.info(f"Loading frozen value network from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config_dict = checkpoint["model_config"]
        vn_config = RECAPSmolVLAValueNetworkConfig(**model_config_dict)
        self.value_network = RECAPSmolVLAValueNetwork(vn_config)
        self.value_network.load_state_dict(checkpoint["model_state_dict"])
        self.value_network.eval()
        for param in self.value_network.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.value_network.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.value_network.parameters())
        logging.info(f"Value network loaded: {total:,} params ({trainable:,} trainable)")

    def _setup_advantage_tokens(self) -> None:
        """Pre-tokenize advantage indicator strings as registered buffers."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.vlm_model_name)

        pos_ids = tokenizer.encode(" Advantage: positive", add_special_tokens=False)
        neg_ids = tokenizer.encode(" Advantage: negative", add_special_tokens=False)

        max_len = max(len(pos_ids), len(neg_ids))
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pos_ids += [pad_id] * (max_len - len(pos_ids))
        neg_ids += [pad_id] * (max_len - len(neg_ids))

        self.register_buffer("advantage_positive_tokens", torch.tensor(pos_ids, dtype=torch.long))
        self.register_buffer("advantage_negative_tokens", torch.tensor(neg_ids, dtype=torch.long))
        self.advantage_token_length = max_len
        logging.info(
            f"Advantage tokens: positive={self.advantage_positive_tokens.tolist()}, "
            f"negative={self.advantage_negative_tokens.tolist()} ({max_len} tokens)"
        )

    def _setup_episode_metadata(self, dataset_meta, labels_path: str) -> None:
        """Load episode labels and build per-episode metadata for on-the-fly return computation."""
        success_map: dict[int, bool] = {}
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                success_map[int(row["episode_index"])] = bool(int(row["success"]))

        episodes = dataset_meta.episodes
        episode_info: dict[int, dict] = {}
        task_max_len: dict[str, int] = {}

        for i in range(len(episodes)):
            ep = episodes[i]
            ep_idx = ep["episode_index"]
            ep_len = ep["length"]
            ep_tasks = ep["tasks"] if isinstance(ep["tasks"], list) else [ep["tasks"]]
            task = ep_tasks[0] if ep_tasks else "unknown"

            if ep_idx in success_map:
                episode_info[ep_idx] = {
                    "length": ep_len,
                    "success": success_map[ep_idx],
                    "task": task,
                }
                if task not in task_max_len or ep_len > task_max_len[task]:
                    task_max_len[task] = ep_len

        self._episode_info = episode_info
        self._task_max_len = task_max_len
        logging.info(
            f"Episode metadata loaded: {len(episode_info)} episodes, "
            f"{sum(1 for v in episode_info.values() if v['success'])} successful"
        )

    def _augment_lang_tokens(
        self,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        advantage_indicator: Tensor | None = None,
        dropout_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Insert advantage indicator tokens right after text, before padding.

        Args:
            lang_tokens: [B, T] token IDs (right-padded).
            lang_masks: [B, T] attention mask (1=real, 0=padding).
            advantage_indicator: [B] bool tensor (True=positive). None → always positive.
            dropout_mask: [B] bool tensor (True=drop advantage tokens for this sample).

        Returns:
            Augmented (lang_tokens, lang_masks) with advantage tokens inserted.
        """
        B, T = lang_tokens.shape
        device = lang_tokens.device
        adv_len = self.advantage_token_length

        if advantage_indicator is not None:
            pos = self.advantage_positive_tokens.unsqueeze(0).expand(B, -1)
            neg = self.advantage_negative_tokens.unsqueeze(0).expand(B, -1)
            adv_tokens = torch.where(
                advantage_indicator[:, None].expand(-1, adv_len),
                pos,
                neg,
            )
        else:
            adv_tokens = self.advantage_positive_tokens.unsqueeze(0).expand(B, -1)

        text_lengths = lang_masks.sum(dim=1, keepdim=True).long()
        offsets = torch.arange(adv_len, device=device).unsqueeze(0)
        positions = (text_lengths + offsets).clamp(max=T - 1)

        new_tokens = lang_tokens.clone()
        new_tokens.scatter_(1, positions, adv_tokens)

        adv_mask_vals = torch.ones(B, adv_len, device=device, dtype=lang_masks.dtype)
        if dropout_mask is not None:
            adv_mask_vals = adv_mask_vals * (~dropout_mask).unsqueeze(1).to(adv_mask_vals.dtype)

        new_masks = lang_masks.clone()
        new_masks.scatter_(1, positions, adv_mask_vals)

        return new_tokens, new_masks

    @torch.no_grad()
    def _compute_value(self, batch: dict[str, Tensor]) -> Tensor:
        """Run the frozen value network to get V(o_t) for each sample.

        Returns:
            Tensor of shape [B] with expected values in [-1, 0].
        """
        images = self._prepare_vn_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        outputs = self.value_network(
            images=images,
            state=state,
            input_ids=lang_tokens,
            attention_mask=lang_masks,
        )
        return outputs["expected_value"].squeeze(-1)

    def _prepare_vn_images(self, batch: dict[str, Tensor]) -> Tensor:
        """Extract and resize images for the value network: [B, N_cam, 3, H, W] in [0, 1]."""
        image_list = []
        for key in self.config.image_features:
            if key in batch:
                img = batch[key]
                if img.ndim == 5:
                    img = img[:, -1]
                image_list.append(img)

        if not image_list:
            raise ValueError("No images found in batch for value network")

        stacked = torch.stack(image_list, dim=1)
        B, N, C, H, W = stacked.shape
        target_h, target_w = self.config.resize_imgs_with_padding
        if H != target_h or W != target_w:
            flat = stacked.reshape(B * N, C, H, W)
            flat = F.interpolate(flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            stacked = flat.reshape(B, N, C, target_h, target_w)

        return stacked

    def _compute_return_from_metadata(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute deterministic return R_t from episode labels and frame position.

        Uses the RECAP reward structure:
          - Non-terminal: r = -1
          - Terminal success: r = 0
          - Terminal failure: r = -c_fail

        Returns are normalized by per-task max episode length and clamped to [-1, 0].
        """
        ep_indices = batch["episode_index"]
        frame_indices = batch["frame_index"]
        B = ep_indices.shape[0]
        device = ep_indices.device

        returns = torch.zeros(B, device=device, dtype=torch.float32)
        for i in range(B):
            ep_idx = ep_indices[i].item()
            frame_idx = frame_indices[i].item()

            info = self._episode_info.get(ep_idx)
            if info is None:
                raise ValueError(
                    f"No episode metadata for episode {ep_idx}. "
                    f"Ensure episode_labels_path covers all training episodes."
                )

            length = info["length"]
            remaining = length - 1 - frame_idx
            ret = float(-remaining)
            if not info["success"]:
                ret -= self.config.c_fail

            max_len = float(self._task_max_len.get(info["task"], length))
            ret = ret / max_len
            returns[i] = max(-1.0, min(0.0, ret))

        return returns

    def _compute_advantages(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute per-sample advantages using the most informative available source.

        Priority:
          1. batch["advantage"] — pre-computed advantages (e.g., from offline labeling)
          2. batch["target_value"] + frozen VN — on-the-fly V(o_t)
          3. Episode metadata + frozen VN — fully on-the-fly R_t and V(o_t)
        """
        if "advantage" in batch:
            return batch["advantage"]

        if self.value_network is None:
            raise ValueError(
                "Cannot compute advantages without a value network. "
                "Either provide batch['advantage'] or set value_network_checkpoint in config."
            )

        V_t = self._compute_value(batch)

        if "target_value" in batch:
            R_t = batch["target_value"]
        elif self._episode_info is not None:
            R_t = self._compute_return_from_metadata(batch)
        else:
            raise ValueError(
                "Cannot compute advantages: no target_value in batch and no episode metadata. "
                "Provide batch['target_value'], batch['advantage'], or set episode_labels_path."
            )

        return R_t - V_t

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Training forward pass with advantage-conditioned flow matching.

        Steps:
          1. Compute per-sample advantages (R_t - V(o_t))
          2. Binarize: positive if advantage > threshold
          3. Apply dropout (30% by default): omit indicator for CFG training
          4. Augment language tokens with advantage indicator
          5. Run base SmolVLA flow-matching forward
        """
        advantage = self._compute_advantages(batch)

        advantage_indicator = advantage > self.config.advantage_threshold

        dropout_mask = None
        if self.training and self.config.advantage_dropout > 0:
            dropout_mask = (
                torch.rand(advantage.shape[0], device=advantage.device) < self.config.advantage_dropout
            )

        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        aug_tokens, aug_masks = self._augment_lang_tokens(
            lang_tokens, lang_masks, advantage_indicator, dropout_mask
        )

        batch = dict(batch)
        batch[OBS_LANGUAGE_TOKENS] = aug_tokens
        batch[OBS_LANGUAGE_ATTENTION_MASK] = aug_masks

        return super().forward(batch, noise=noise, time=time, reduction=reduction)

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Inference action sampling: always condition on 'Advantage: positive'.

        For cfg_beta > 1, runs classifier-free guidance by interpolating
        conditioned and unconditioned flow vectors.
        """
        orig_tokens = batch[OBS_LANGUAGE_TOKENS]
        orig_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        aug_tokens, aug_masks = self._augment_lang_tokens(orig_tokens, orig_masks)

        batch = dict(batch)
        batch[OBS_LANGUAGE_TOKENS] = aug_tokens
        batch[OBS_LANGUAGE_ATTENTION_MASK] = aug_masks
        batch["_orig_lang_tokens"] = orig_tokens
        batch["_orig_lang_masks"] = orig_masks

        if self.config.cfg_beta > 1.0:
            return self._get_action_chunk_cfg(batch, noise, **kwargs)

        return super()._get_action_chunk(batch, noise, **kwargs)

    def _get_action_chunk_cfg(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Classifier-free guidance inference.

        Runs two forward passes per denoising step:
          1. Conditioned on "Advantage: positive"
          2. Unconditioned (no advantage tokens)
        Interpolates: v = v_uncond + beta * (v_cond - v_uncond)
        """
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)

        cond_tokens = batch[OBS_LANGUAGE_TOKENS]
        cond_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        uncond_tokens = batch["_orig_lang_tokens"]
        uncond_masks = batch["_orig_lang_masks"]

        bsize = state.shape[0]
        device = state.device
        beta = self.config.cfg_beta

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.model.sample_noise(actions_shape, device)

        cond_prefix_embs, cond_prefix_pad, cond_prefix_att = self.model.embed_prefix(
            images, img_masks, cond_tokens, cond_masks, state=state
        )
        uncond_prefix_embs, uncond_prefix_pad, uncond_prefix_att = self.model.embed_prefix(
            images, img_masks, uncond_tokens, uncond_masks, state=state
        )

        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        cond_att_2d = make_att_2d_masks(cond_prefix_pad, cond_prefix_att)
        cond_pos_ids = torch.cumsum(cond_prefix_pad, dim=1) - 1
        _, cond_kv = self.model.vlm_with_expert.forward(
            attention_mask=cond_att_2d,
            position_ids=cond_pos_ids,
            past_key_values=None,
            inputs_embeds=[cond_prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        uncond_att_2d = make_att_2d_masks(uncond_prefix_pad, uncond_prefix_att)
        uncond_pos_ids = torch.cumsum(uncond_prefix_pad, dim=1) - 1
        _, uncond_kv = self.model.vlm_with_expert.forward(
            attention_mask=uncond_att_2d,
            position_ids=uncond_pos_ids,
            past_key_values=None,
            inputs_embeds=[uncond_prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(bsize)

            v_cond = self.model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=cond_prefix_pad,
                past_key_values=cond_kv,
                timestep=time_tensor,
            )
            v_uncond = self.model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=uncond_prefix_pad,
                past_key_values=uncond_kv,
                timestep=time_tensor,
            )

            v_t = v_uncond + beta * (v_cond - v_uncond)
            x_t = x_t + dt * v_t

        actions = x_t
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def get_optim_params(self) -> dict:
        """Exclude frozen value network parameters from optimization."""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith("value_network."):
                params.append(param)
        return params
