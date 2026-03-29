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

Wraps Pi0.5 with RECAP-style advantage conditioning via text tokens.
A frozen SmolVLA value network (trained separately) labels training data with
per-sample advantages that are binarized into "Advantage: positive" or
"Advantage: negative" text, which is appended to the language prompt before the
VLM processes it.

Pi0.5 uses a single-stream architecture where language tokens and action tokens
share the same causal attention sequence.  The advantage text sits at the end of
the language prefix, making it the most recent context when the action expert
begins generating — exactly the mechanism described in the RECAP paper for
pi-0.6.

At inference the model always appends "Advantage: positive".  Optional
classifier-free guidance (cfg_beta > 1) runs two prefix computations (with /
without advantage text) and interpolates the flow vectors.

Two-phase workflow:
  1. Train a value network with RECAPTrainSmolVLANetwork (see docs/source/recap.mdx)
  2. Train PiStar06Policy using pre-computed advantages from the frozen value network
"""

import csv
import logging
from typing import Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PI05Policy,
    make_att_2d_masks,
)
from lerobot.policies.pistar06.configuration_pistar06 import PiStar06Config
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


class PiStar06Policy(PI05Policy):
    """Advantage-conditioned Pi0.5 policy using text-token injection.

    During training, pre-computed advantages (R_t - V(o_t)) are read from the
    batch, binarized, and the corresponding text ("Advantage: positive" or
    "Advantage: negative") is appended to the language tokens.  30 % of the
    time (configurable) the advantage text is omitted entirely, training the
    unconditional path for optional classifier-free guidance at test time.

    During inference, "Advantage: positive" is always appended.
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

        if config.num_expert_layers > 0:
            expert_model = self.model.paligemma_with_expert.gemma_expert.model
            total = len(expert_model.layers)
            if config.num_expert_layers > total:
                raise ValueError(
                    f"num_expert_layers={config.num_expert_layers} exceeds "
                    f"action expert depth {total}"
                )
            expert_model.layers = expert_model.layers[: config.num_expert_layers]
            logging.info(
                f"Truncated action expert to {config.num_expert_layers}/{total} layers"
            )

        self._setup_advantage_tokens()

        self._episode_info: dict[int, dict] | None = None
        self._task_max_len: dict[str, int] | None = None
        if dataset_meta is not None and config.episode_labels_path is not None:
            self._setup_episode_metadata(dataset_meta, config.episode_labels_path)

        self._train_step_count = 0

    # ── Advantage token setup ────────────────────────────────────────────

    def _setup_advantage_tokens(self) -> None:
        """Pre-tokenize advantage strings with the PaliGemma tokenizer."""
        from transformers import AutoTokenizer

        # The tokenizer is identical across all PaliGemma model sizes (300M, 2B, 3B);
        # only tokenizer config files are downloaded, not model weights.
        tokenizer = AutoTokenizer.from_pretrained(self.config.paligemma_tokenizer_name)
        pos_ids = tokenizer.encode(" Advantage: positive", add_special_tokens=False)
        neg_ids = tokenizer.encode(" Advantage: negative", add_special_tokens=False)

        self.register_buffer(
            "_positive_adv_token_ids",
            torch.tensor(pos_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_negative_adv_token_ids",
            torch.tensor(neg_ids, dtype=torch.long),
            persistent=False,
        )
        logging.info(
            f"Advantage tokens: positive={pos_ids} ({len(pos_ids)} tokens), "
            f"negative={neg_ids} ({len(neg_ids)} tokens)"
        )

    def _inject_advantage_text(
        self,
        tokens: Tensor,
        masks: Tensor,
        advantage_indicator: Tensor,
        dropout_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Append advantage text tokens to each sample's language sequence.

        Finds the boundary between real content and right-padding in each row,
        then writes the appropriate advantage token IDs there and extends the
        attention mask accordingly.

        Args:
            tokens: [B, seq_len] token IDs (will be cloned).
            masks:  [B, seq_len] attention mask (will be cloned).
            advantage_indicator: [B] bool — True means positive advantage.
            dropout_mask: [B] bool or None — True means *omit* advantage text
                for that sample (unconditional pass for CFG training).

        Returns:
            Modified (tokens, masks) tensors.
        """
        tokens = tokens.clone()
        masks = masks.clone()
        seq_len = tokens.shape[1]

        for i in range(tokens.shape[0]):
            if dropout_mask is not None and dropout_mask[i]:
                continue

            content_len = int(masks[i].sum().item())
            adv_ids = (
                self._positive_adv_token_ids
                if advantage_indicator[i]
                else self._negative_adv_token_ids
            )
            n_adv = adv_ids.shape[0]

            if content_len + n_adv > seq_len:
                n_adv = seq_len - content_len
                if n_adv <= 0:
                    continue
                adv_ids = adv_ids[:n_adv]

            tokens[i, content_len : content_len + n_adv] = adv_ids
            masks[i, content_len : content_len + n_adv] = 1

        return tokens, masks

    # ── Episode metadata ─────────────────────────────────────────────────

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

    # ── Advantage computation ────────────────────────────────────────────

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
        diagnostics: dict[str, float] = {
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

    # ── Training forward ─────────────────────────────────────────────────

    def _forward_with_advantage(
        self,
        batch: dict[str, Tensor],
        advantage_indicator: Tensor,
        dropout_mask: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Flow-matching forward with advantage text injected into language tokens.

        Appends "Advantage: positive/negative" to the language prompt, then runs
        the standard Pi0.5 forward pass (embed_prefix → embed_suffix → joint
        VLM+expert → action_out_proj → MSE loss).

        Returns:
            Per-element MSE losses of shape [B, chunk_size, action_dim].
        """
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)

        tokens, masks = self._inject_advantage_text(
            tokens, masks, advantage_indicator, dropout_mask
        )

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
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.model.embed_suffix(
            x_t, time
        )

        if (
            self.model.paligemma_with_expert.paligemma.model.language_model
            .layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
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
        return F.mse_loss(u_t, v_t, reduction="none")

    def forward(  # ty: ignore[invalid-method-override]
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict[str, float]]:
        """Training forward pass with advantage-conditioned flow matching.

        Steps:
          1. Read pre-computed per-sample advantages from batch
          2. Binarize: positive if advantage > threshold
          3. Apply dropout (30 % default): omit advantage text for CFG training
          4. Append advantage text tokens to the language prompt
          5. Run flow-matching forward and compute loss
        """
        advantage, adv_diagnostics = self._compute_advantages(batch)

        advantage_indicator = advantage > self.config.advantage_threshold
        n_positive = advantage_indicator.sum().item()
        n_total = advantage_indicator.shape[0]

        dropout_mask = None
        if self.training and self.config.advantage_dropout > 0:
            dropout_mask = (
                torch.rand(advantage.shape[0], device=advantage.device)
                < self.config.advantage_dropout
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
                    parts.append(
                        f"V(o_t)={adv_diagnostics['V_t_mean']:.4f}"
                        f"±{adv_diagnostics['V_t_std']:.4f}"
                    )
                    parts.append(
                        f"R_t={adv_diagnostics['R_t_mean']:.4f}"
                        f"±{adv_diagnostics['R_t_std']:.4f}"
                    )
                parts.append(
                    f"adv={adv_diagnostics['advantage_mean']:.4f}"
                    f"±{adv_diagnostics['advantage_std']:.4f}"
                )
                parts.append(f"text=pos:{n_positive}/{n_total} drop:{n_dropped}/{n_total}")
                parts.append(f"thresh={self.config.advantage_threshold}")
                logging.info("  ".join(parts))

        return loss, output_dict

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Inference with "Advantage: positive" appended to the prompt.

        The advantage text becomes part of the prefix KV-cache, so the standard
        ``PI05Pytorch.denoise_step`` is used for each Euler step — no custom
        denoising method is needed.
        """
        self.eval()

        if self.config.cfg_beta > 1.0:
            return self._predict_action_chunk_cfg(batch, **kwargs)

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        bsize = tokens.shape[0]
        device = tokens.device

        positive_indicator = torch.ones(bsize, dtype=torch.bool, device=device)
        tokens, masks = self._inject_advantage_text(tokens, masks, positive_indicator, None)

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

        num_steps = self.config.num_inference_steps
        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return x_t[:, :, :original_action_dim]

    def _predict_action_chunk_cfg(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Classifier-free guidance inference.

        Two prefix computations:
          1. Conditioned — "Advantage: positive" appended → KV-cache_cond
          2. Unconditional — no advantage text → KV-cache_uncond

        Each Euler step runs two ``denoise_step`` calls (one per cache) and
        interpolates: v = v_uncond + beta * (v_cond - v_uncond).
        """
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        bsize = tokens.shape[0]
        device = tokens.device
        beta = self.config.cfg_beta

        positive_indicator = torch.ones(bsize, dtype=torch.bool, device=device)
        tokens_cond, masks_cond = self._inject_advantage_text(
            tokens, masks, positive_indicator, None
        )

        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.model.sample_noise(actions_shape, device)

        self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Conditioned prefix (with advantage text)
        prefix_embs_c, prefix_pad_masks_c, prefix_att_masks_c = self.model.embed_prefix(
            images, img_masks, tokens_cond, masks_cond
        )
        att_2d_c = make_att_2d_masks(prefix_pad_masks_c, prefix_att_masks_c)
        pos_ids_c = torch.cumsum(prefix_pad_masks_c, dim=1) - 1
        _, past_kv_cond = self.model.paligemma_with_expert.forward(
            attention_mask=self.model._prepare_attention_masks_4d(att_2d_c),
            position_ids=pos_ids_c,
            past_key_values=None,
            inputs_embeds=[prefix_embs_c, None],
            use_cache=True,
        )

        # Unconditional prefix (no advantage text — original tokens)
        prefix_embs_u, prefix_pad_masks_u, prefix_att_masks_u = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        att_2d_u = make_att_2d_masks(prefix_pad_masks_u, prefix_att_masks_u)
        pos_ids_u = torch.cumsum(prefix_pad_masks_u, dim=1) - 1
        _, past_kv_uncond = self.model.paligemma_with_expert.forward(
            attention_mask=self.model._prepare_attention_masks_4d(att_2d_u),
            position_ids=pos_ids_u,
            past_key_values=None,
            inputs_embeds=[prefix_embs_u, None],
            use_cache=True,
        )

        num_steps = self.config.num_inference_steps
        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(bsize)

            v_cond = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks_c,
                past_key_values=past_kv_cond,
                x_t=x_t,
                timestep=time_tensor,
            )
            v_uncond = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks_u,
                past_key_values=past_kv_uncond,
                x_t=x_t,
                timestep=time_tensor,
            )

            v_t = v_uncond + beta * (v_cond - v_uncond)
            x_t = x_t + dt * v_t

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return x_t[:, :, :original_action_dim]

    # ── Optimizer params ─────────────────────────────────────────────────

    def get_optim_params(self) -> list:  # ty: ignore[invalid-method-override]
        """Return all trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
