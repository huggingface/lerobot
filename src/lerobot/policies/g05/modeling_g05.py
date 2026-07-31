# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

"""Complete G0.5 model and LeRobot policy entry points."""

from __future__ import annotations

import copy
import logging
import re
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.device_utils import resolve_safetensors_device
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_g05 import G05Config

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5VisionModel

    from .modular_g05 import (
        ActionExpert,
        G05GatedDeltaNet,
        G05VisionPatchEmbed,
        G05VisionPatchMerger,
    )
else:
    DynamicCache = None
    Qwen3_5TextConfig = None
    Qwen3_5VisionConfig = None
    Qwen3_5TextModel = None
    Qwen3_5VisionModel = None
    ActionExpert = None
    G05GatedDeltaNet = None
    G05VisionPatchEmbed = None
    G05VisionPatchMerger = None


_COT_SPECIAL_TOKEN = re.compile(r"<\|[a-z_]+\|>")
_COT_RULE_WIDTH = 78


def format_chain_of_thought(text: str) -> str:
    """Render a generated chain of thought as a readable block.

    G0.5 emits reasoning as ``|``-separated ``Label: value`` segments with
    grounding markup interleaved, which is unreadable on a single log line. Each
    segment gets its own row with labels aligned. The block has no right border,
    so wide characters can never break the layout.
    """
    cleaned = _COT_SPECIAL_TOKEN.sub("", text or "").strip()
    lines = ["╭─ G0.5 Chain of Thought " + "─" * (_COT_RULE_WIDTH - 24)]
    segments = [part.strip() for part in cleaned.split("|") if part.strip()]
    if not segments:
        lines.append("│ (empty)")
    else:
        split = [segment.split(":", 1) for segment in segments]
        width = max((len(pair[0]) for pair in split if len(pair) == 2), default=0)
        for pair in split:
            if len(pair) == 2:
                lines.append(f"│ {pair[0].strip():<{width}} : {pair[1].strip()}".rstrip())
            else:
                lines.append(f"│ {pair[0].strip()}")
    lines.append("╰" + "─" * (_COT_RULE_WIDTH - 1))
    return "\n".join(lines)


class CotGeneration(NamedTuple):
    """Per-row state carried from the CoT stage into the AR action stage.

    ``stop_tokens`` is the token that actually terminated each row. The AR stage
    must commit that exact token rather than re-decoding the frozen hidden state,
    which is not reproducible once ``ar_do_sample`` is enabled. ``history`` keeps
    the CoT tokens visible to the repetition penalties during the action stage;
    ``history_mask`` excludes padding added after shorter rows finish.
    """

    stop_tokens: Tensor
    history: Tensor | None
    history_mask: Tensor | None


class G05Model(nn.Module):
    """Qwen3.5 VLM plus the flow-matching G0.5 action expert."""

    def __init__(self, config: G05Config) -> None:
        require_package("transformers", extra="g05")
        super().__init__()
        from transformers import utils as transformers_utils

        for backend in {config.attn_implementation, config.vision_attn_implementation}:
            if backend.startswith("flash_attention"):
                version = backend.rsplit("_", 1)[-1]
                available = getattr(transformers_utils, f"is_flash_attn_{version}_available")()
                if not available:
                    raise ImportError(f"{backend} was selected but its local package is unavailable")
        text_config = Qwen3_5TextConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.text_hidden_size,
            intermediate_size=config.text_intermediate_size,
            num_hidden_layers=config.text_num_layers,
            num_attention_heads=config.text_num_heads,
            num_key_value_heads=config.text_num_kv_heads,
            head_dim=config.text_head_dim,
            max_position_embeddings=262_144,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": 0.25,
                "mrope_section": list(config.mrope_section),
                "mrope_interleaved": True,
            },
            layer_types=config.text_layer_types,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
        )
        vision_config = Qwen3_5VisionConfig(
            depth=config.vision_depth,
            hidden_size=config.vision_hidden_size,
            intermediate_size=config.vision_intermediate_size,
            num_heads=config.vision_num_heads,
            patch_size=config.vision_patch_size,
            temporal_patch_size=config.vision_temporal_patch_size,
            spatial_merge_size=config.vision_spatial_merge_size,
            out_hidden_size=config.text_hidden_size,
        )
        text_config._attn_implementation = config.attn_implementation
        vision_config._attn_implementation = config.vision_attn_implementation
        self.config = config
        self.vision_tower = Qwen3_5VisionModel(vision_config)
        # The released checkpoint keeps only patch projection and merger in
        # fp32. Tiny forward overrides preserve those autocast boundaries while
        # the complete vision tower remains the native Transformers model.
        self.vision_tower.patch_embed = G05VisionPatchEmbed(vision_config)
        self.vision_tower.merger = G05VisionPatchMerger(vision_config, use_postshuffle_norm=False)
        self.vlm = Qwen3_5TextModel(text_config)
        # Transformers' generic Qwen3.5 linear attention uses different chunk
        # and precision defaults. Replace only those layers while retaining the
        # same checkpoint parameter paths and DynamicCache contract.
        for index, layer in enumerate(self.vlm.layers):
            if layer.layer_type == "linear_attention":
                layer.linear_attn = G05GatedDeltaNet(text_config, index)
        self.output_proj = nn.Linear(config.text_hidden_size, config.vocab_size, bias=False)
        self.output_proj.weight = self.vlm.embed_tokens.weight
        self.proprio_embedder = nn.Sequential(
            nn.Linear(config.internal_state_dim, config.text_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.text_hidden_size),
            nn.Linear(config.text_hidden_size, config.text_hidden_size),
        )
        self.action_expert = ActionExpert(config)
        action_dimension_is_pad = torch.ones(config.internal_action_dim, dtype=torch.bool)
        physical_action_dim = config.output_features[ACTION].shape[-1]
        action_indices = config.action_indices or list(range(physical_action_dim))
        action_dimension_is_pad[action_indices] = False
        self.register_buffer("action_dimension_is_pad", action_dimension_is_pad, persistent=False)

    def materialize_runtime_buffers(self, device: torch.device | str) -> None:
        """Recreate buffers omitted from safetensors after meta construction."""
        action_dimension_is_pad = torch.ones(self.config.internal_action_dim, dtype=torch.bool, device=device)
        action_indices = self.config.action_indices or list(
            range(self.config.output_features[ACTION].shape[-1])
        )
        action_dimension_is_pad[action_indices] = False
        self.action_dimension_is_pad = action_dimension_is_pad

        vision_rotary = self.vision_tower.rotary_pos_emb
        vision_rotary.inv_freq = 1.0 / (
            vision_rotary.theta
            ** (torch.arange(0, vision_rotary.dim, 2, dtype=torch.float32, device=device) / vision_rotary.dim)
        )

        for rotary in (self.vlm.rotary_emb, self.action_expert.rotary_emb):
            # The Transformers helper creates an intermediate arange without an
            # explicit device before moving it. Override the surrounding meta
            # context so that intermediate has real storage.
            with torch.device(device):
                inv_freq, rotary.attention_scaling = rotary.compute_default_rope_parameters(
                    rotary.config, device=device
                )
            rotary.inv_freq = inv_freq
            rotary.original_inv_freq = inv_freq.clone()

        self.action_expert.time_embedding.materialize_buffer(device)

    def encode_images(self, images: Tensor) -> Tensor:
        """Patchify all cameras/timesteps and encode them as one packed vision batch.

        G0.5 repeats each single frame over the temporal patch axis. The einops
        layout mirrors the official token order: image, merged-grid row/column,
        merge row/column, then flattened channel/temporal/patch content.
        """
        if images.ndim == 5:
            images = rearrange(
                images, "batch camera channel height width -> batch camera 1 channel height width"
            )
        if images.ndim != 6:
            raise ValueError("g05_images must have shape [B,Cam,Time,C,H,W]")

        batch_size, num_cameras, num_frames, _, height, width = images.shape
        patch_size = self.config.vision_patch_size
        merge_size = self.config.vision_spatial_merge_size
        temporal_patch = self.config.vision_temporal_patch_size
        grid_height, grid_width = height // patch_size, width // patch_size
        frames = repeat(
            images,
            "batch camera time channel height width -> (batch camera time) channel temporal height width",
            temporal=temporal_patch,
        )
        # The official checkpoint was trained with a legacy channel/temporal
        # reinterpretation before patch extraction. These two named layouts
        # preserve that exact order without an opaque reshape/permute chain.
        legacy_frames = rearrange(
            frames, "frames channel temporal height width -> frames (channel temporal) height width"
        )
        patches = rearrange(
            legacy_frames,
            "frames (temporal channel) "
            "(grid_h merge_h patch_h) (grid_w merge_w patch_w) -> "
            "(frames grid_h grid_w merge_h merge_w) (channel temporal patch_h patch_w)",
            temporal=temporal_patch,
            channel=images.shape[3],
            merge_h=merge_size,
            merge_w=merge_size,
            patch_h=patch_size,
            patch_w=patch_size,
        )

        num_images = batch_size * num_cameras * num_frames
        grid_thw = torch.tensor((1, grid_height, grid_width), device=images.device, dtype=torch.long).expand(
            num_images, -1
        )
        encoded = self.vision_tower(patches, grid_thw).pooler_output
        return rearrange(encoded, "(batch tokens) hidden -> batch tokens hidden", batch=batch_size)

    def build_mrope_position_ids(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Build vectorized Qwen3.5 multimodal rotary positions.

        Text advances all three axes by one. An image block uses a constant
        temporal coordinate and a 2-D spatial grid, then advances the following
        text by ``max(grid_height, grid_width)``. Left-padding stays at position 0.
        Processor-generated prompts guarantee contiguous, fixed-size image blocks.
        """
        merged_height = self.config.image_size[0] // (
            self.config.vision_patch_size * self.config.vision_spatial_merge_size
        )
        merged_width = self.config.image_size[1] // (
            self.config.vision_patch_size * self.config.vision_spatial_merge_size
        )
        image_length = merged_height * merged_width
        image_span = max(merged_height, merged_width)
        image_mask = input_ids.eq(self.config.image_token_id) & attention_mask
        if torch.any(image_mask.sum(dim=-1).remainder(image_length)):
            raise ValueError(f"every image must occupy exactly {image_length} prompt tokens")

        valid_rank = attention_mask.long().cumsum(dim=-1) - 1
        image_token_rank = image_mask.long().cumsum(dim=-1) - 1
        within_image = image_token_rank.remainder(image_length)
        image_number = torch.div(image_token_rank, image_length, rounding_mode="floor")
        completed_images = torch.div(image_mask.long().cumsum(dim=-1), image_length, rounding_mode="floor")

        text_position = valid_rank - completed_images * (image_length - image_span)
        image_position = valid_rank - within_image - image_number * (image_length - image_span)
        temporal = torch.where(image_mask, image_position, text_position)
        height = torch.where(
            image_mask,
            image_position + torch.div(within_image, merged_width, rounding_mode="floor"),
            text_position,
        )
        width = torch.where(
            image_mask,
            image_position + within_image.remainder(merged_width),
            text_position,
        )
        position_ids = torch.stack((temporal, height, width))
        return position_ids.masked_fill(~attention_mask.unsqueeze(0), 0)

    def vlm_forward(self, batch: dict[str, Tensor], *, use_cache: bool):
        """Run the multimodal language model for either prefix caching or AR loss."""
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
        inputs_embeds = self.vlm.embed_tokens(input_ids)

        image_features = self.encode_images(batch["pixel_values"]).to(inputs_embeds.dtype)
        image_mask = input_ids.eq(self.config.image_token_id)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1), image_features)

        if self.config.state_token_id is not None:
            state_mask = input_ids.eq(self.config.state_token_id)
            # The released model keeps the proprio encoder in fp32. A single
            # bf16-rounded state token changes GatedDeltaNet's recurrent state
            # and consequently every token that follows it.
            with torch.autocast(inputs_embeds.device.type, enabled=False):
                state_features = self.proprio_embedder(batch[OBS_STATE].float())
            state_features = state_features.to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(state_mask.unsqueeze(-1), state_features)

        position_ids = self.build_mrope_position_ids(input_ids, attention_mask)
        outputs = self.vlm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            return_dict=True,
        )
        return outputs, position_ids, attention_mask

    def prefill(self, batch: dict[str, Tensor]) -> tuple[DynamicCache, Tensor, Tensor, Tensor, Tensor]:
        """Encode the multimodal prefix once for flow and token objectives."""
        outputs, position_ids, attention_mask = self.vlm_forward(batch, use_cache=True)
        return (
            outputs.past_key_values,
            position_ids,
            attention_mask,
            outputs.last_hidden_state[:, -1:],
            batch[OBS_LANGUAGE_TOKENS],
        )

    def copy_vlm_cache(self, source: DynamicCache) -> DynamicCache:
        """Clone mutable linear-attention state while sharing immutable prefix KV."""
        destination = DynamicCache(config=self.vlm.config)
        for index, source_layer in enumerate(source.layers):
            destination_layer = copy.copy(source_layer)
            if not hasattr(source_layer, "is_initialized"):
                destination_layer.conv_states = source_layer.conv_states.clone()
                destination_layer.recurrent_states = source_layer.recurrent_states.clone()
            destination.layers[index] = destination_layer
        return destination

    def sample_next_token(
        self,
        logits: Tensor,
        history: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply the AR sampling contract serialized from the source checkpoint.

        Greedy decoding deliberately ignores the repetition penalty and the
        n-gram ban: the source decoder forces temperature to zero when
        ``do_sample`` is false and returns a plain argmax before either
        constraint is applied. Honouring the penalties here instead would
        suppress the legitimately repeated ActionCodec tokens that released
        checkpoints emit, and the AR chunk then diverges from the reference.
        """
        if not self.config.ar_do_sample or self.config.ar_temperature == 0:
            return logits.float().argmax(dim=-1)

        scores = logits.float().clone()
        if history is not None and history.numel():
            if history_mask is not None and history_mask.shape != history.shape:
                raise ValueError("history_mask must have the same shape as history")
            if self.config.ar_repetition_penalty != 1:
                if history_mask is None:
                    previous_scores = scores.gather(1, history)
                    previous_scores = torch.where(
                        previous_scores < 0,
                        previous_scores * self.config.ar_repetition_penalty,
                        previous_scores / self.config.ar_repetition_penalty,
                    )
                    scores.scatter_(1, history, previous_scores)
                else:
                    for batch_index in range(history.shape[0]):
                        row = history[batch_index, history_mask[batch_index]]
                        if not row.numel():
                            continue
                        previous_scores = scores[batch_index, row]
                        previous_scores = torch.where(
                            previous_scores < 0,
                            previous_scores * self.config.ar_repetition_penalty,
                            previous_scores / self.config.ar_repetition_penalty,
                        )
                        scores[batch_index].scatter_(0, row, previous_scores)

            ngram_size = self.config.ar_no_repeat_ngram_size
            if ngram_size > 0:
                histories = (
                    history.tolist()
                    if history_mask is None
                    else [history[index, history_mask[index]].tolist() for index in range(history.shape[0])]
                )
                for batch_index, row in enumerate(histories):
                    if len(row) < ngram_size - 1:
                        continue
                    prefix = tuple(row[-(ngram_size - 1) :]) if ngram_size > 1 else ()
                    banned = {
                        row[index + ngram_size - 1]
                        for index in range(len(row) - ngram_size + 1)
                        if tuple(row[index : index + ngram_size - 1]) == prefix
                    }
                    if banned:
                        scores[batch_index, list(banned)] = -torch.inf

        scores /= self.config.ar_temperature
        top_k = min(self.config.ar_top_k, scores.shape[-1])
        if top_k > 0:
            threshold = torch.topk(scores, top_k, dim=-1).values[:, -1:]
            scores.masked_fill_(scores < threshold, -torch.inf)
        if self.config.ar_top_p < 1:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative > self.config.ar_top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            scores.scatter_(1, sorted_indices, sorted_scores.masked_fill(remove, -torch.inf))
        return torch.multinomial(scores.softmax(dim=-1), num_samples=1).squeeze(1)

    def generate_cot(
        self,
        batch: dict[str, Tensor],
        prefix: tuple[DynamicCache, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[tuple[DynamicCache, Tensor, Tensor, Tensor, Tensor], CotGeneration]:
        """Extend the VLM prefix until every sample emits EOV or EOS.

        The released SO101 checkpoint conditions its flow expert on this generated
        context. Stop tokens are returned by the language head but are not committed
        to the cache, matching the source inference pipeline. Rows that finish early
        receive masked cache slots while other rows continue; their recurrent linear
        attention state is restored after each padded forward.

        The returned :class:`CotGeneration` carries each row's stop token so the AR
        action stage can commit it verbatim, and the CoT tokens so the repetition
        penalties stay continuous across both stages. Rows that exhaust the token
        budget without stopping report ``-1`` and are simply sampled onwards.
        """
        cache, position_ids, attention_mask, last_hidden, input_ids = prefix
        prompt_length = input_ids.shape[1]
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        stop_tokens = torch.full_like(input_ids[:, 0], -1)
        stop_ids = {self.config.eos_token_id}
        if self.config.eov_token_id is not None:
            stop_ids.add(self.config.eov_token_id)
        generated_history = None
        generated_history_mask = None
        # The hardware-verified SO100 path predates per-row history masking.
        # Preserve that exact sampling contract for SO100 artifacts; changing
        # which prior tokens receive repetition penalties can change AR output.
        track_history_mask = input_ids.shape[0] > 1 and self.config.embodiment not in {
            "so100",
            "so101",
        }

        for _ in range(self.config.max_cot_tokens):
            if generated_history_mask is None:
                next_token = self.sample_next_token(self.output_proj(last_hidden[:, -1]), generated_history)
            else:
                next_token = self.sample_next_token(
                    self.output_proj(last_hidden[:, -1]),
                    generated_history,
                    generated_history_mask,
                )
            is_stop = torch.zeros_like(finished)
            for token_id in stop_ids:
                is_stop |= next_token.eq(token_id)
            stop_tokens = torch.where(is_stop & (~finished), next_token, stop_tokens)
            finished |= is_stop
            if bool(finished.all()):
                break
            history_token = next_token
            if track_history_mask:
                valid_history_token = ~finished
                history_token = history_token.masked_fill(~valid_history_token, self.config.pad_token_id)
                generated_history_mask = (
                    valid_history_token.unsqueeze(1)
                    if generated_history_mask is None
                    else torch.cat((generated_history_mask, valid_history_token.unsqueeze(1)), dim=1)
                )
            generated_history = (
                history_token.unsqueeze(1)
                if generated_history is None
                else torch.cat((generated_history, history_token.unsqueeze(1)), dim=1)
            )

            # Stop tokens are not part of the action-conditioning prefix. Finished
            # rows use a masked pad slot so batched cache lengths remain rectangular.
            input_token = next_token.masked_fill(finished, self.config.pad_token_id).unsqueeze(1)
            input_ids = torch.cat((input_ids, input_token), dim=1)
            attention_mask = torch.cat((attention_mask, (~finished).unsqueeze(1)), dim=1)
            position_ids = self.build_mrope_position_ids(input_ids, attention_mask)

            frozen_linear_states = []
            for layer in cache.layers:
                if hasattr(layer, "keys"):
                    continue
                frozen_linear_states.append(
                    (
                        layer,
                        layer.conv_states[finished].clone(),
                        layer.recurrent_states[finished].clone(),
                    )
                )

            outputs = self.vlm(
                inputs_embeds=self.vlm.embed_tokens(input_token),
                attention_mask=attention_mask,
                position_ids=position_ids[..., -1:],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            for layer, conv_states, recurrent_states in frozen_linear_states:
                layer.conv_states[finished] = conv_states
                layer.recurrent_states[finished] = recurrent_states
            last_hidden = torch.where(
                finished[:, None, None],
                last_hidden,
                outputs.last_hidden_state,
            )
        # Kept so deployments can surface the reasoning the chunk was conditioned
        # on; the stop token is excluded because it is never committed.
        self.last_cot_tokens = input_ids[:, prompt_length:].detach()
        return (
            (cache, position_ids, attention_mask, last_hidden, input_ids),
            CotGeneration(
                stop_tokens=stop_tokens,
                history=generated_history,
                history_mask=generated_history_mask,
            ),
        )

    @torch.no_grad()
    def sample_action_tokens(self, batch: dict[str, Tensor]) -> list[Tensor]:
        """Generate one serialized ActionCodec sequence per observation.

        This path only runs the vision/VLM prefix and language head. It never calls
        the flow-matching action expert, allowing AR and FM deployment costs to be
        measured independently.
        """
        prefix = self.prefill(batch)
        cot: CotGeneration | None = None
        if self.config.predict_cot:
            prefix, cot = self.generate_cot(batch, prefix)
        cache, _, attention_mask, last_hidden, input_ids = prefix
        start = self.config.action_token_start_id
        end = self.config.action_token_end_id
        if start is None or end is None:
            raise ValueError("AR inference requires the converted action-token range")

        batch_size = input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        action_started = torch.zeros_like(finished)
        generated: list[list[Tensor]] = [[] for _ in range(batch_size)]
        full_ids = input_ids
        # Repetition penalties stay continuous across the CoT and action stages.
        generated_history = cot.history if cot is not None else None
        generated_history_mask = cot.history_mask if cot is not None else None
        eov_token_id = self.config.eov_token_id

        for step in range(self.config.max_action_tokens):
            if generated_history_mask is None:
                next_token = self.sample_next_token(self.output_proj(last_hidden[:, -1]), generated_history)
            else:
                next_token = self.sample_next_token(
                    self.output_proj(last_hidden[:, -1]),
                    generated_history,
                    generated_history_mask,
                )
            if step == 0 and cot is not None:
                # CoT stopped before committing its stop token. Re-decoding the
                # frozen hidden state would draw a fresh sample under
                # ar_do_sample, so commit the token the row actually emitted.
                next_token = torch.where(cot.stop_tokens.ge(0), cot.stop_tokens, next_token)
            is_action = next_token.ge(start) & next_token.lt(end)
            live_action = (~finished) & is_action
            for batch_index in live_action.nonzero(as_tuple=False).flatten().tolist():
                generated[batch_index].append(next_token[batch_index])

            # CoT generation stops before committing EOV to the cache, exactly as
            # the source pipeline does. The AR action stage consumes that one EOV
            # transition token, then collects the contiguous ActionCodec payload.
            transition_eov = (
                torch.zeros_like(finished)
                if eov_token_id is None
                else self.config.predict_cot & (~action_started) & next_token.eq(eov_token_id)
            )
            finished |= ~(is_action | transition_eov)
            action_started |= live_action
            if bool(finished.all()):
                break
            history_token = next_token
            if generated_history_mask is not None:
                valid_history_token = ~finished
                history_token = history_token.masked_fill(~valid_history_token, self.config.pad_token_id)
                generated_history_mask = torch.cat(
                    (generated_history_mask, valid_history_token.unsqueeze(1)), dim=1
                )
            generated_history = (
                history_token.unsqueeze(1)
                if generated_history is None
                else torch.cat((generated_history, history_token.unsqueeze(1)), dim=1)
            )

            input_token = next_token.masked_fill(finished, self.config.pad_token_id).unsqueeze(1)
            full_ids = torch.cat((full_ids, input_token), dim=1)
            attention_mask = torch.cat((attention_mask, (~finished).unsqueeze(1)), dim=1)
            position_ids = self.build_mrope_position_ids(full_ids, attention_mask)

            frozen_linear_states = []
            for layer in cache.layers:
                if hasattr(layer, "keys"):
                    continue
                frozen_linear_states.append(
                    (
                        layer,
                        layer.conv_states[finished].clone(),
                        layer.recurrent_states[finished].clone(),
                    )
                )
            outputs = self.vlm(
                inputs_embeds=self.vlm.embed_tokens(input_token),
                attention_mask=attention_mask,
                position_ids=position_ids[..., -1:],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            for layer, conv_states, recurrent_states in frozen_linear_states:
                layer.conv_states[finished] = conv_states
                layer.recurrent_states[finished] = recurrent_states
            last_hidden = torch.where(finished[:, None, None], last_hidden, outputs.last_hidden_state)

        return [torch.stack(row) if row else input_ids.new_empty(0) for row in generated]

    def autoregressive_loss(
        self,
        batch: dict[str, Tensor],
        prefill: tuple[DynamicCache, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute shifted CE and action/CoT accuracies over labeled suffix tokens."""
        if ACTION_TOKENS not in batch:
            raise ValueError("discrete G0.5 training requires action.tokens from the processor")
        prefix_cache, _, _, prefix_last_hidden, _ = prefill
        suffix_cache = self.copy_vlm_cache(prefix_cache)
        prefix_ids = batch[OBS_LANGUAGE_TOKENS]
        prefix_mask = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
        suffix_ids = batch[ACTION_TOKENS]
        full_ids = torch.cat((prefix_ids, suffix_ids), dim=-1)
        full_mask = torch.cat((prefix_mask, torch.ones_like(suffix_ids, dtype=torch.bool)), dim=-1)
        suffix_positions = self.build_mrope_position_ids(full_ids, full_mask)[:, :, prefix_ids.shape[-1] :]
        outputs = self.vlm(
            inputs_embeds=self.vlm.embed_tokens(suffix_ids),
            attention_mask=full_mask,
            position_ids=suffix_positions,
            past_key_values=suffix_cache,
            use_cache=False,
            return_dict=True,
        )
        # The final prefix state predicts the first action token; each suffix
        # state then predicts the following action token or EOS.
        hidden_states = torch.cat((prefix_last_hidden, outputs.last_hidden_state[:, :-1]), dim=1)
        labels = suffix_ids
        valid = labels.ne(-100)
        if not valid.any():
            zero = hidden_states.sum() * 0
            return zero, zero.detach(), zero.detach()
        hidden_states = hidden_states[valid]
        targets = labels[valid]
        logits = self.output_proj(hidden_states)
        loss = torch.nn.functional.cross_entropy(logits.float(), targets)
        correct = logits.detach().argmax(-1).eq(targets)
        start, end = self.config.action_token_start_id, self.config.action_token_end_id
        if start is None or end is None:
            raise ValueError("converted discrete G0.5 config is missing its action-token range")
        action_mask = targets.ge(start) & targets.lt(end)
        action_accuracy = correct[action_mask].float().mean() if action_mask.any() else loss.new_zeros(())
        cot_mask = ~action_mask
        cot_accuracy = correct[cot_mask].float().mean() if cot_mask.any() else loss.new_zeros(())
        return loss, action_accuracy, cot_accuracy

    def flow_loss(
        self,
        batch: dict[str, Tensor],
        prefill: tuple[DynamicCache, Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Compute the masked conditional-flow velocity objective."""
        actions = batch[ACTION]
        prefix_cache, prefix_position_ids, prefix_attention_mask, _, _ = prefill or self.prefill(batch)
        if not self.config.flow_joint_training:
            # Post-training variants optimize only the expert. Full-attention
            # layers cache KV tensors, while linear-attention layers cache
            # convolutional and recurrent state; detach both cache layouts so
            # the flow objective cannot retain or backpropagate through the VLM.
            for layer in prefix_cache.layers:
                if hasattr(layer, "keys") and layer.is_initialized:
                    layer.keys = layer.keys.detach()
                    layer.values = layer.values.detach()
                elif not hasattr(layer, "keys"):
                    layer.conv_states = layer.conv_states.detach()
                    layer.recurrent_states = layer.recurrent_states.detach()

        num_samples = self.config.num_flow_samples
        batch_size = actions.shape[0]
        if self.config.flow_sampling == "beta":
            # Official training samples Beta times on CPU. One batched transfer
            # preserves its seeded RNG sequence without per-sample transfers.
            distribution = torch.distributions.Beta(self.config.flow_beta_alpha, self.config.flow_beta_beta)
            time = distribution.sample((num_samples, batch_size)).to(
                device=actions.device, dtype=actions.dtype
            )
            time = 1 - (1 - self.config.flow_sig_min) * (1 - time)
        else:
            offsets = torch.arange(batch_size, device=actions.device, dtype=actions.dtype)
            time = (
                torch.rand(num_samples, 1, device=actions.device, dtype=actions.dtype) + offsets / batch_size
            ).remainder(1 - 1e-5)

        time = rearrange(time, "samples batch -> (samples batch)")
        noise = torch.randn(
            num_samples,
            *actions.shape,
            device=actions.device,
            dtype=actions.dtype,
        )
        noise = rearrange(noise, "samples batch horizon dim -> (samples batch) horizon dim")
        actions = repeat(actions, "batch horizon dim -> (samples batch) horizon dim", samples=num_samples)
        noised_actions = (1 - time[:, None, None]) * actions + time[:, None, None] * noise

        if num_samples > 1:
            prefix_cache = self.action_expert.copy_prefix_cache(prefix_cache)
            prefix_cache.batch_repeat_interleave(num_samples)
            prefix_position_ids = repeat(
                prefix_position_ids,
                "axes batch tokens -> axes (samples batch) tokens",
                samples=num_samples,
            )
            prefix_attention_mask = repeat(
                prefix_attention_mask,
                "batch tokens -> (samples batch) tokens",
                samples=num_samples,
            )

        velocity = self.action_expert(
            noised_actions,
            time,
            prefix_cache,
            prefix_position_ids,
            prefix_attention_mask,
        )
        squared_error = (velocity - (noise - actions)).square()
        weights = torch.ones_like(squared_error)
        if "action_is_pad" in batch:
            action_pad_mask = repeat(
                batch["action_is_pad"],
                "batch horizon -> (samples batch) horizon",
                samples=num_samples,
            )
            weights.masked_fill_(action_pad_mask[..., None], 0)
        weights.masked_fill_(self.action_dimension_is_pad[None, None, :], 0)
        return (weights * squared_error).sum() / weights.sum().clamp_min(1)

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Integrate the learned velocity field from Gaussian noise to actions."""
        prefix = self.prefill(batch)
        if self.config.predict_cot:
            prefix, _ = self.generate_cot(batch, prefix)
        prefix_cache, prefix_position_ids, prefix_attention_mask, _, _ = prefix
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.internal_action_dim,
                device=input_ids.device,
                dtype=batch[OBS_STATE].dtype,
            )
        dimension_pad_mask = self.action_dimension_is_pad[None, None, :]
        noise = noise.masked_fill(dimension_pad_mask, 0)
        actions = noise
        step_size = 1 / self.config.num_inference_steps
        time = torch.ones(batch_size, device=actions.device, dtype=actions.dtype)
        for _ in range(self.config.num_inference_steps):
            velocity = self.action_expert(
                actions,
                time,
                prefix_cache,
                prefix_position_ids,
                prefix_attention_mask,
            )
            actions = actions - step_size * velocity
            actions.masked_fill_(dimension_pad_mask, 0)
            time = time - step_size
        return actions


class G05Policy(PreTrainedPolicy):
    """LeRobot policy wrapper for training and chunked G0.5 inference."""

    config_class = G05Config
    name = "g05"

    def __init__(
        self,
        config: G05Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs,
    ) -> None:
        """Build the policy from a converted G0.5 configuration.

        Args:
            config: Architecture and data-boundary contract of the checkpoint.
            dataset_stats: Unused. G0.5 normalization lives in the processor pipeline,
                which either carries the checkpoint's fixed statistics or receives
                dataset statistics directly. Accepted so the policy factory can pass
                it uniformly.
            **kwargs: Remaining factory arguments (such as ``dataset_meta``), unused.
        """
        require_package("transformers", extra="g05")
        super().__init__(config)
        config.validate_features()
        self.model = G05Model(config)
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)
        self._physical_action_dim = config.output_features[ACTION].shape[-1]
        self._action_tokenizer = None
        self._text_tokenizer = None

    @property
    def last_cot_text(self) -> list[str] | None:
        """Chain of thought generated for the most recent chunk, one per batch row.

        ``None`` until a CoT-enabled inference has run, or when the artifact's
        text tokenizer was not loaded.
        """
        tokens = getattr(self.model, "last_cot_tokens", None)
        if tokens is None or self._text_tokenizer is None:
            return None
        return [
            self._text_tokenizer.decode(row, skip_special_tokens=False).strip() for row in tokens.tolist()
        ]

    @classmethod
    def _load_as_safetensor(
        cls, model: G05Policy, model_file: str, map_location: str, strict: bool
    ) -> G05Policy:
        # The policy is constructed on meta, then mmap-backed tensors are loaded
        # directly onto the final device and assigned without an intermediate
        # initialized model or state-dict copy.
        from safetensors.torch import load_file

        device = resolve_safetensors_device(map_location)
        # pread avoids mmap page-fault serialization on remote filesystems,
        # while still keeping host memory bounded by the largest tensor.
        state_dict = load_file(model_file, device=device, backend="pread")
        embedding_key = "model.vlm.embed_tokens.weight"
        output_projection_key = "model.output_proj.weight"
        if embedding_key not in state_dict and output_projection_key in state_dict:
            state_dict[embedding_key] = state_dict[output_projection_key]
        elif output_projection_key not in state_dict and embedding_key in state_dict:
            state_dict[output_projection_key] = state_dict[embedding_key]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
        model.model.materialize_runtime_buffers(device)

        # save_model may omit either alias of the shared table. Both keys were
        # supplied above for strict validation; restore the Parameter identity.
        model.model.output_proj.weight = model.model.vlm.embed_tokens.weight
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Error(s) loading G0.5 safetensors: missing={missing_keys}, unexpected={unexpected_keys}"
            )
        return model

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, **kwargs) -> G05Policy:
        # G0.5 is roughly 11 GB. Constructing on meta avoids allocating and
        # randomly initializing a throwaway full-size CPU model before loading.
        with torch.device("meta"):
            policy = super().from_pretrained(pretrained_name_or_path, **kwargs)
        # The text tokenizer is also needed to render generated chain of thought,
        # so it is loaded whenever either feature is enabled.
        if not policy.config.discrete_action and not policy.config.predict_cot:
            return policy

        artifact_root = Path(pretrained_name_or_path)
        if not artifact_root.is_dir():
            from huggingface_hub import snapshot_download

            artifact_root = Path(
                snapshot_download(
                    str(pretrained_name_or_path),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                    allow_patterns=[
                        f"{policy.config.tokenizer_subdir}/*",
                        f"{policy.config.action_tokenizer_subdir}/*",
                    ],
                )
            )
        from transformers import AutoTokenizer

        text_tokenizer = AutoTokenizer.from_pretrained(
            artifact_root / policy.config.tokenizer_subdir,
            local_files_only=True,
        )
        policy._text_tokenizer = text_tokenizer
        if not policy.config.discrete_action:
            return policy

        from .action_tokenizer import G05ActionCodecModel, G05ActionTokenizer

        codec = G05ActionCodecModel.from_pretrained(
            artifact_root / policy.config.action_tokenizer_subdir,
            local_files_only=True,
        ).to(next(policy.parameters()).device)
        policy._action_tokenizer = G05ActionTokenizer(codec, text_tokenizer)
        return policy

    def reset(self) -> None:
        self._action_queue.clear()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Return the configured loss and logging-friendly scalar metrics."""
        device_type = batch[OBS_LANGUAGE_TOKENS].device.type
        with torch.autocast(
            device_type,
            dtype=torch.bfloat16,
            enabled=self.config.dtype == "bfloat16" and device_type == "cuda",
        ):
            prefill = self.model.prefill(batch)
            if self.config.discrete_action:
                ar_loss, action_accuracy, cot_accuracy = self.model.autoregressive_loss(batch, prefill)
                ar_loss = ar_loss * self.config.action_token_loss_weight
            else:
                ar_loss = prefill[3].new_zeros(())
                action_accuracy = cot_accuracy = ar_loss.detach()
            fm_loss = (
                self.model.flow_loss(batch, prefill=prefill) * self.config.fm_loss_weight
                if self.config.continuous_action
                else prefill[3].new_zeros(())
            )
        loss = fm_loss + ar_loss
        return loss, {
            "fm_loss": fm_loss.item(),
            "action_token_loss": ar_loss.item(),
            "cot_loss": 0.0,
            "action_token_accuracy": action_accuracy.item(),
            "cot_accuracy": cot_accuracy.item() if self.config.predict_cot else 0.0,
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict actions through the single enabled continuous or discrete path."""
        if self.config.discrete_action == self.config.continuous_action:
            raise ValueError(
                "inference requires exactly one of discrete_action or continuous_action to be enabled"
            )

        device_type = batch[OBS_LANGUAGE_TOKENS].device.type
        with torch.autocast(
            device_type,
            dtype=torch.bfloat16,
            enabled=self.config.dtype == "bfloat16" and device_type == "cuda",
        ):
            if self.config.continuous_action:
                actions = self.model.sample_actions(batch, noise=noise)
            else:
                if self._action_tokenizer is None:
                    raise RuntimeError("AR inference requires the artifact's ActionCodec tokenizer")
                token_rows = self.model.sample_action_tokens(batch)
                decoded_rows = []
                for index, row in enumerate(token_rows):
                    if row.numel() == 0:
                        # A zero chunk still commands the arm, so make the cause
                        # visible instead of letting it look like a real action.
                        logging.warning(
                            "G0.5 AR inference produced no ActionCodec tokens for batch row %d; "
                            "falling back to a zero action chunk. Check predict_cot, the "
                            "action-token range, and the AR sampling configuration.",
                            index,
                        )
                        decoded = batch[OBS_STATE].new_zeros(
                            self.config.chunk_size, self.config.internal_action_dim
                        )
                    else:
                        decoded = self._action_tokenizer.decode(row.unsqueeze(0))[0]
                    decoded_rows.append(decoded)
                actions = torch.stack(decoded_rows)
        self._log_chain_of_thought()
        indices = self.config.action_indices or list(range(self._physical_action_dim))
        return actions[..., indices]

    def _log_chain_of_thought(self) -> None:
        """Print the reasoning this chunk was conditioned on.

        Lives here rather than in a runner so rollout, eval, and direct policy
        calls show it without each caller having to know about G0.5.
        """
        for text in self.last_cot_text or []:
            if text.strip():
                logging.info("\n%s", format_chain_of_thought(text))

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Serve one action while amortizing model inference over a predicted chunk."""
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, noise=noise)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()
