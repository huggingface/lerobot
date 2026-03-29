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
PiStar06: Advantage-Conditioned Pi0.5 Policy

Wraps Pi0.5 with RECAP-style advantage conditioning. A frozen SmolVLA value
network (trained separately) labels training data with per-sample advantages
that are binarized and injected as a learned embedding directly into the action
expert's input pathway (embed_suffix). This bypasses the VLM text processing
and gives the advantage signal a direct gradient path to the flow-matching loss.

At inference, the model conditions on the positive advantage embedding to
produce higher-quality actions.

Two-phase workflow:
  1. Train a value network with RECAPTrainSmolVLANetwork (see RECAP_VALUE_NETWORK_TRAINING.md)
  2. Train PiStar06Policy using pre-computed advantages from the frozen value network
"""

import copy
import csv
import logging
from typing import Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PI05Policy,
    get_gemma_config,
    make_att_2d_masks,
    pad_vector,
)
from lerobot.policies.pistar06.configuration_pistar06 import PiStar06Config
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


class PiStar06Policy(PI05Policy):
    """Advantage-conditioned Pi0.5 policy.

    During training, pre-computed advantages (R_t - V(o_t)) are read from the
    batch. The advantage is binarized and injected as a learned embedding
    directly into the action expert's suffix input (embed_suffix), bypassing
    the PaliGemma text processing entirely.

    During inference, the positive advantage embedding is always applied.
    Optional classifier-free guidance (cfg_beta > 1) sharpens the distribution
    by interpolating conditioned and unconditioned flow vectors.
    """

    config: PiStar06Config
    config_class = PiStar06Config
    name = "pistar06"

    def __init__(
        self,
        config: PiStar06Config,
        dataset_meta=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        expert_config = get_gemma_config(config.action_expert_variant)
        expert_hidden_size = expert_config.width
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

    def _setup_episode_metadata(self, dataset_meta, labels_path: str) -> None:
        """Load episode labels and build per-episode metadata for return computation."""
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
            reference_emb: [B, chunk_size, expert_hidden] tensor to match shape and dtype.

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

    def _compute_advantages(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Read pre-computed advantages from the batch.

        Expects batch["advantage"] to be a [B] tensor of advantage values,
        pre-computed by the training script using the frozen SmolVLA value network.
        """
        if "advantage" not in batch:
            raise ValueError(
                "PiStar06 requires pre-computed advantages in batch['advantage']. "
                "The training script should pre-compute these using the SmolVLA value network."
            )
        adv = batch["advantage"]
        diagnostics = {
            "advantage_mean": adv.mean().item(),
            "advantage_std": adv.std().item(),
        }
        if "target_value" in batch:
            diagnostics["R_t_mean"] = batch["target_value"].mean().item()
            diagnostics["R_t_std"] = batch["target_value"].std().item()
        if "predicted_value" in batch:
            diagnostics["V_t_mean"] = batch["predicted_value"].mean().item()
            diagnostics["V_t_std"] = batch["predicted_value"].std().item()
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

        Replicates PI05Pytorch.forward() but adds the advantage embedding to the
        suffix embeddings before running the joint VLM+expert pass.

        Returns:
            Per-element MSE losses of shape [B, chunk_size, action_dim].
        """
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)

        if noise is None:
            noise = self.model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.model.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.model.embed_suffix(x_t, time)

        adv_emb = self._build_advantage_embedding(advantage_indicator, dropout_mask, suffix_embs)
        suffix_embs = suffix_embs + adv_emb

        if (
            self.model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self.model._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = self.model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.model.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def forward(  # ty: ignore[invalid-method-override]
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict[str, float]]:
        """Training forward pass with advantage-conditioned flow matching.

        Steps:
          1. Read pre-computed per-sample advantages from batch
          2. Binarize: positive if advantage > threshold
          3. Apply dropout (30% by default): zero out advantage embedding for CFG training
          4. Inject advantage embedding into action expert suffix
          5. Run flow-matching forward and compute loss
        """
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

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict: dict[str, float] = {}
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

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
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.model.embed_suffix(
            x_t, timestep
        )

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

        full_att_2d_masks_4d = self.model._prepare_attention_masks_4d(full_att_2d_masks)
        self.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.model.action_out_proj(suffix_out)
        return v_t

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Inference action sampling with positive advantage embedding.

        For cfg_beta > 1, runs classifier-free guidance by interpolating
        conditioned and unconditioned flow vectors.
        """
        self.eval()

        if self.config.cfg_beta > 1.0:
            return self._predict_action_chunk_cfg(batch, **kwargs)

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        bsize = tokens.shape[0]
        device = tokens.device

        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.model.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        advantage_indicator = torch.ones(bsize, dtype=torch.bool, device=device)

        num_steps = self.config.num_inference_steps
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
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions

    def _predict_action_chunk_cfg(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Classifier-free guidance inference.

        Runs two denoising passes per Euler step:
          1. Conditioned: positive advantage embedding applied
          2. Unconditioned: no advantage embedding (zero)
        Interpolates: v = v_uncond + beta * (v_cond - v_uncond)
        """
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        bsize = tokens.shape[0]
        device = tokens.device
        beta = self.config.cfg_beta

        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.model.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        cond_indicator = torch.ones(bsize, dtype=torch.bool, device=device)
        uncond_dropout = torch.ones(bsize, dtype=torch.bool, device=device)

        num_steps = self.config.num_inference_steps
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

            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.model.embed_suffix(
                x_t, time_tensor
            )
            adv_emb = self._build_advantage_embedding(cond_indicator, uncond_dropout, suffix_embs)
            uncond_suffix_embs = suffix_embs + adv_emb

            suffix_len = suffix_pad_masks.shape[1]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = self.model._prepare_attention_masks_4d(full_att_2d_masks)
            self.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

            past_kv_copy = copy.deepcopy(past_key_values)
            outputs_embeds, _ = self.model.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_kv_copy,
                inputs_embeds=[None, uncond_suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_uncond = self.model.action_out_proj(suffix_out)

            v_t = v_uncond + beta * (v_cond - v_uncond)
            x_t = x_t + dt * v_t

        actions = x_t
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions

    def get_optim_params(self) -> list:  # ty: ignore[invalid-method-override]
        """Return all trainable parameters (no frozen value network to exclude)."""
        return [p for p in self.parameters() if p.requires_grad]
