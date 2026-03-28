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
binarized and injected as a learned embedding directly into the action expert's
input pathway (embed_suffix). This bypasses the VLM text processing and gives
the advantage signal a direct gradient path to the flow-matching loss.

At inference, the model conditions on the positive advantage embedding to
produce higher-quality actions.

Two-phase workflow:
  1. Train a value network with RECAPTrainSmolVLANetwork (see RECAP_VALUE_NETWORK_TRAINING.md)
  2. Train SmolStar06Policy using the frozen value network weights
"""

import csv
import logging
from typing import Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.smolstar06.configuration_smolstar06 import SmolStar06Config
from lerobot.policies.smolvla.modeling_smolvla import ActionSelectKwargs, SmolVLAPolicy, make_att_2d_masks
from lerobot.rl.algorithms.RECAPSmolVLAValueNetwork import (
    RECAPSmolVLAValueNetwork,
    RECAPSmolVLAValueNetworkConfig,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


class SmolStar06Policy(SmolVLAPolicy):
    """Advantage-conditioned SmolVLA policy.

    During training, a frozen value network computes V(o_t) for each sample.
    The deterministic return R_t is derived from episode success/fail labels.
    Advantage A = R_t - V(o_t) is binarized and injected as a learned embedding
    directly into the action expert's suffix input (embed_suffix), bypassing
    the VLM text processing entirely.

    During inference, the positive advantage embedding is always applied.
    Optional classifier-free guidance (cfg_beta > 1) sharpens the distribution
    by interpolating conditioned and unconditioned flow vectors.
    """

    config: SmolStar06Config
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

        expert_hidden_size = self.model.vlm_with_expert.expert_hidden_size
        self.advantage_embedding = nn.Embedding(2, expert_hidden_size)
        nn.init.zeros_(self.advantage_embedding.weight)
        logging.info(
            f"Advantage embedding initialized: nn.Embedding(2, {expert_hidden_size}), "
            f"zero-initialized"
        )

        self._episode_info: dict[int, dict] | None = None
        self._task_max_len: dict[str, int] | None = None
        if dataset_meta is not None and config.episode_labels_path is not None:
            self._setup_episode_metadata(dataset_meta, config.episode_labels_path)

        self._train_step_count = 0

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
                    "dataset_from_index": ep["dataset_from_index"],
                }
                if task not in task_max_len or ep_len > task_max_len[task]:
                    task_max_len[task] = ep_len

        self._episode_info = episode_info
        self._task_max_len = task_max_len
        logging.info(
            f"Episode metadata loaded: {len(episode_info)} episodes, "
            f"{sum(1 for v in episode_info.values() if v['success'])} successful"
        )

    def _build_advantage_embedding(
        self,
        advantage_indicator: Tensor,
        dropout_mask: Tensor | None,
        reference_emb: Tensor,
    ) -> Tensor:
        """Build the advantage embedding to add to the suffix.

        Args:
            advantage_indicator: [B] bool tensor (True=positive advantage).
            dropout_mask: [B] bool tensor (True=drop advantage for this sample).
                When dropped, the embedding is zeroed out (unconditional pass).
            reference_emb: [B, chunk_size, expert_hidden] tensor to match
                shape and dtype.

        Returns:
            [B, chunk_size, expert_hidden] advantage embedding tensor.
        """
        indices = advantage_indicator.long()
        adv_emb = self.advantage_embedding(indices)
        adv_emb = adv_emb.unsqueeze(1).expand_as(reference_emb)
        adv_emb = adv_emb.to(dtype=reference_emb.dtype)

        if dropout_mask is not None:
            keep_mask = (~dropout_mask).to(dtype=adv_emb.dtype)
            adv_emb = adv_emb * keep_mask[:, None, None]

        return adv_emb

    @torch.no_grad()
    def _compute_value(self, batch: dict[str, Tensor]) -> Tensor:
        """Run the frozen value network to get V(o_t) for each sample.

        Returns:
            Tensor of shape [B] with expected values in [-1, 0].
        """
        assert self.value_network is not None
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
        if target_h != H or target_w != W:
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
        assert self._episode_info is not None
        assert self._task_max_len is not None
        ep_indices = batch["episode_index"]
        global_indices = batch["index"]
        B = ep_indices.shape[0]
        device = ep_indices.device

        returns = torch.zeros(B, device=device, dtype=torch.float32)
        for i in range(B):
            ep_idx = int(ep_indices[i].item())

            info = self._episode_info.get(ep_idx)
            if info is None:
                raise ValueError(
                    f"No episode metadata for episode {ep_idx}. "
                    f"Ensure episode_labels_path covers all training episodes."
                )

            frame_idx = global_indices[i].item() - info["dataset_from_index"]
            length = info["length"]
            remaining = length - 1 - frame_idx
            ret = float(-remaining)
            if not info["success"]:
                ret -= self.config.c_fail

            max_len = float(self._task_max_len.get(info["task"], length))
            ret = ret / max_len
            returns[i] = max(-1.0, min(0.0, ret))

        return returns

    def _compute_advantages(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Compute per-sample advantages using the most informative available source.

        Priority:
          1. batch["advantage"] -- pre-computed advantages (e.g., from offline labeling)
          2. batch["target_value"] + frozen VN -- on-the-fly V(o_t)
          3. Episode metadata + frozen VN -- fully on-the-fly R_t and V(o_t)

        Returns (advantage, diagnostics) where diagnostics contains V_t, R_t stats.
        """
        if "advantage" in batch:
            adv = batch["advantage"]
            return adv, {"advantage_mean": adv.mean().item(), "advantage_std": adv.std().item()}

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

        adv = R_t - V_t
        diagnostics = {
            "V_t_mean": V_t.mean().item(),
            "V_t_std": V_t.std().item(),
            "R_t_mean": R_t.mean().item(),
            "R_t_std": R_t.std().item(),
            "advantage_mean": adv.mean().item(),
            "advantage_std": adv.std().item(),
        }
        return adv, diagnostics

    def _forward_with_advantage(
        self,
        batch: dict[str, Tensor],
        advantage_indicator: Tensor,
        dropout_mask: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Flow-matching forward with advantage embedding injected into embed_suffix.

        This method replicates VLAFlowMatching.forward() but adds the advantage
        embedding to the suffix embeddings before running the joint VLM+expert pass.

        Returns:
            Per-element MSE losses of shape [B, chunk_size, action_dim].
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)

        if noise is None:
            noise = self.model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.model.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.model.embed_suffix(x_t, time)

        adv_emb = self._build_advantage_embedding(advantage_indicator, dropout_mask, suffix_embs)
        suffix_embs = suffix_embs + adv_emb

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.model.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.model.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def forward(  # ty: ignore[invalid-method-override]
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict[str, float]]:
        """Training forward pass with advantage-conditioned flow matching.

        Steps:
          1. Compute per-sample advantages (R_t - V(o_t))
          2. Binarize: positive if advantage > threshold
          3. Apply dropout (30% by default): zero out advantage embedding for CFG training
          4. Inject advantage embedding into action expert suffix
          5. Run flow-matching forward and compute loss
        """
        if self.config.adapt_to_pi_aloha:
            from lerobot.utils.constants import OBS_STATE
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        advantage, adv_diagnostics = self._compute_advantages(batch)

        advantage_indicator = advantage > self.config.advantage_threshold
        n_positive = advantage_indicator.sum().item()
        n_total = advantage_indicator.shape[0]

        dropout_mask = None
        if self.training and self.config.advantage_dropout > 0:
            dropout_mask = (
                torch.rand(advantage.shape[0], device=advantage.device) < self.config.advantage_dropout
            )

        losses = self._forward_with_advantage(
            batch, advantage_indicator, dropout_mask, noise=noise, time=time
        )

        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict: dict[str, float] = {}
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss = per_sample_loss
            loss_dict["loss"] = per_sample_loss.mean().item()
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()

        output_dict = loss_dict
        output_dict.update(adv_diagnostics)
        output_dict["advantage_threshold"] = self.config.advantage_threshold
        output_dict["advantage_pct_positive"] = n_positive / n_total
        n_dropped = int(dropout_mask.sum().item()) if dropout_mask is not None else 0
        output_dict["advantage_pct_dropped"] = n_dropped / n_total

        if self.training:
            self._train_step_count += 1
            if self._train_step_count % 10 == 1:
                parts = [f"[RECAP step {self._train_step_count}]"]
                if "V_t_mean" in adv_diagnostics:
                    parts.append(f"V(o_t)={adv_diagnostics['V_t_mean']:.4f}±{adv_diagnostics['V_t_std']:.4f}")
                    parts.append(f"R_t={adv_diagnostics['R_t_mean']:.4f}±{adv_diagnostics['R_t_std']:.4f}")
                parts.append(f"adv={adv_diagnostics['advantage_mean']:.4f}±{adv_diagnostics['advantage_std']:.4f}")
                parts.append(f"emb=pos:{n_positive}/{n_total} drop:{n_dropped}/{n_total}")
                parts.append(f"thresh={self.config.advantage_threshold}")
                logging.info("  ".join(parts))

        return loss, output_dict

    def _denoise_step_with_advantage(
        self,
        x_t: Tensor,
        prefix_pad_masks: Tensor,
        past_key_values,
        timestep: Tensor,
        advantage_indicator: Tensor,
    ) -> Tensor:
        """Single denoising step with advantage embedding injected into suffix."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.model.embed_suffix(x_t, timestep)

        adv_emb = self._build_advantage_embedding(advantage_indicator, None, suffix_embs)
        suffix_embs = suffix_embs + adv_emb

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.model.vlm_with_expert.forward(
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
        v_t = self.model.action_out_proj(suffix_out)
        return v_t

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Inference action sampling with positive advantage embedding.

        For cfg_beta > 1, runs classifier-free guidance by interpolating
        conditioned and unconditioned flow vectors.
        """
        if self.config.cfg_beta > 1.0:
            return self._get_action_chunk_cfg(batch, noise, **kwargs)

        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.model.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        advantage_indicator = torch.ones(bsize, dtype=torch.bool, device=device)

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(bsize)

            v_t = self._denoise_step_with_advantage(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
                advantage_indicator=advantage_indicator,
            )
            x_t = x_t + dt * v_t

        actions = x_t
        assert self.config.action_feature is not None
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _get_action_chunk_cfg(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Classifier-free guidance inference.

        Runs two denoising passes per Euler step:
          1. Conditioned: positive advantage embedding applied
          2. Unconditioned: no advantage embedding (zero)
        Interpolates: v = v_uncond + beta * (v_cond - v_uncond)
        """
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        bsize = state.shape[0]
        device = state.device
        beta = self.config.cfg_beta

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.model.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        cond_indicator = torch.ones(bsize, dtype=torch.bool, device=device)
        uncond_dropout = torch.ones(bsize, dtype=torch.bool, device=device)

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(bsize)

            v_cond = self._denoise_step_with_advantage(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
                advantage_indicator=cond_indicator,
            )

            suffix_embs, suffix_pad_masks, suffix_att_masks = self.model.embed_suffix(x_t, time_tensor)

            adv_emb = self._build_advantage_embedding(cond_indicator, uncond_dropout, suffix_embs)
            uncond_suffix_embs = suffix_embs + adv_emb

            suffix_len = suffix_pad_masks.shape[1]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            outputs_embeds, _ = self.model.vlm_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, uncond_suffix_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
            )
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_uncond = self.model.action_out_proj(suffix_out)

            v_t = v_uncond + beta * (v_cond - v_uncond)
            x_t = x_t + dt * v_t

        actions = x_t
        assert self.config.action_feature is not None
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def get_optim_params(self) -> list:  # ty: ignore[invalid-method-override]
        """Exclude frozen value network parameters from optimization."""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith("value_network."):
                params.append(param)
        return params
