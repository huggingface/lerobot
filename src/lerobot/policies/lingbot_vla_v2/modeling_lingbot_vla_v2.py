from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager
from typing import Any

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_lingbot_vla_v2 import LingbotVLAV2Config as LeRobotLingbotVLAV2Config
from .model_core.modeling_lingbot_vla_v2_base import (
    FlowMatching as FlowMatchingV1,
    replace_lnorm_with_adanorm,
)
from .model_core.moe_loss import sequence_wise_balance_loss
from .model_core.qwen2_action_expert import (
    Qwen2ForCausalLM,
    Qwen2TokenMoeBlock,
)
from .model_core.qwen3vl_in_vla import Qwen3VLForConditionalGeneration
from .model_core.utils import (
    make_att_2d_masks,
    our_eager_attention_forward,
    our_sdpa_attention_forward,
)

logger = logging.getLogger(__name__)


@contextmanager
def _frozen_params(module: nn.Module):
    """Temporarily freeze trainable params so RTC guidance only builds the x_t graph."""
    params = [p for p in module.parameters() if p.requires_grad]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p in params:
            p.requires_grad_(True)


class QwenvlWithExpertV2Config(PretrainedConfig):
    model_type = "QwenvlWithExpertV2Model"

    def __init__(
        self,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        vocab_size: int = 0,
        use_lm_head: bool = False,
        attention_implementation: str = "sdpa",
        tokenizer_path: str | None = None,
        use_cache: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        action_num_attention_heads: int = 32,
        action_num_key_value_heads: int = 8,
        action_head_dim: int = 128,
        num_layers: int = 36,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        self.action_num_attention_heads = action_num_attention_heads
        self.action_num_key_value_heads = action_num_key_value_heads
        self.action_head_dim = action_head_dim

        self.qwen_expert_config = CONFIG_MAPPING["qwen2"](
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            hidden_act="silu",
            hidden_size=expert_hidden_size,
            head_dim=action_head_dim,
            initializer_range=0.02,
            intermediate_size=expert_intermediate_size,
            max_position_embeddings=32768,
            max_window_layers=21,
            model_type="qwen2",
            num_attention_heads=action_num_attention_heads,
            num_hidden_layers=num_layers,
            num_key_value_heads=action_num_key_value_heads,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=32768,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            transformers_version="4.57.3",
            use_cache=use_cache,
            use_sliding_window=False,
            vocab_size=151936,
        )
        super().__init__(**kwargs)


class QwenvlWithExpertV2Model(PreTrainedModel):
    config_class = QwenvlWithExpertV2Config

    def __init__(self, config: QwenvlWithExpertV2Config, vlm_config: PretrainedConfig):
        super().__init__(config=config)
        self.config = config
        # The LLM attention is computed by the custom dual-stream forward, so the HF
        # model is built with "eager"; the vision tower reads its value straight through.
        hf_attn = "eager"
        hf_vit_attn = self.config.vit_attn_implementation
        if self.config.vocab_size:
            vlm_config.text_config.vocab_size = self.config.vocab_size
        vlm_config._attn_implementation = hf_attn
        vlm_config.text_config._attn_implementation = hf_attn
        vlm_config.vision_config._attn_implementation = hf_vit_attn
        self.qwenvl = Qwen3VLForConditionalGeneration._from_config(vlm_config)
        if self.config.use_lm_head:
            self.qwenvl.tie_weights()

        self.config.qwen_expert_config._attn_implementation = hf_attn
        self.qwen_expert = Qwen2ForCausalLM._from_config(self.config.qwen_expert_config)

        if self.config.adanorm_time:
            replace_lnorm_with_adanorm(
                self.qwen_expert,
                self.config.qwen_expert_config.hidden_size,
                self.config.qwen_expert_config.hidden_size,
                config.final_norm_adanorm,
            )

        self._install_moe_blocks()
        self.pos_embeds = None
        self.position_embeddings = None
        self.cu_seqlens = None
        self.visual_split_sizes = None
        self.visual_max_seqlen = None
        # Capture-context flag: set by the full-prefix CUDA-graph wrapper so the vision
        # grid metadata (pos_embeds / cu_seqlens / split_sizes / max_seqlen) is hoisted
        # out of the per-call host-sync path and cached once. See ``_prefix_graphed``.
        self._capture_grid_cache = False

        del self.qwen_expert.model.embed_tokens
        self.attention_interface = self.get_attention_interface()
        self.set_requires_grad()

    def _install_moe_blocks(self):
        if not self.config.use_moe:
            return
        layers = self.qwen_expert.model.layers
        # Layers past the expert depth (e.g. the 36-layer default on a smaller VLM) are skipped.
        token_moe_layers = [idx for idx in self.config.token_moe_layers if idx < len(layers)]

        if token_moe_layers:
            token_config = CONFIG_MAPPING["qwen2_moe"](
                num_experts=self.config.token_num_experts,
                num_experts_per_tok=self.config.token_top_k,
                norm_topk_prob=True,
                hidden_size=self.config.qwen_expert_config.hidden_size,
                moe_intermediate_size=self.config.token_moe_intermediate_size,
                shared_expert_intermediate_size=self.config.token_shared_intermediate_size,
                output_router_logits=False,
            )
            token_config.router_activation = self.config.router_activation
            token_config.routed_scaling_factor = self.config.routed_scaling_factor
            token_config.use_shared_expert_gate = self.config.use_shared_expert_gate
            for idx in token_moe_layers:
                layers[idx].mlp = Qwen2TokenMoeBlock(token_config)

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.qwenvl.model.visual.eval()
            for params in self.qwenvl.model.visual.parameters():
                params.requires_grad = False
        if self.config.train_expert_only:
            self.qwenvl.eval()
            for params in self.qwenvl.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.qwenvl.model.visual.eval()
        if self.config.train_expert_only:
            self.qwenvl.eval()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ):
        precompute_grid_thw = self.config.precompute_grid_thw
        # Hoist the host-syncing grid preprocess when (a) the precompute flag wants it
        # cached and it is not yet, or (b) the capture grid cache is armed but empty
        # (first warm-up pass of a vision-graph capture). Once populated, subsequent
        # calls — including the CUDA-graph capture itself — skip preprcess_grid_thw
        # entirely, which is what makes the tower capturable (its .item()/.tolist()
        # host syncs cannot be captured).
        trainable_position = (
            torch.is_grad_enabled() and self.qwenvl.model.visual.pos_embed.weight.requires_grad
        )
        if trainable_position:
            # An inference cache becomes stale as soon as the next update can
            # change pos_embed. Keep only parameter-independent grid metadata.
            self.pos_embeds = None
        grid_cache_ready = self.position_embeddings is not None and (
            trainable_position or self.pos_embeds is not None
        )
        if (precompute_grid_thw and not grid_cache_ready) or (
            self._capture_grid_cache and not grid_cache_ready
        ):
            (
                self.pos_embeds,
                self.position_embeddings,
                self.cu_seqlens,
                self.visual_split_sizes,
                self.visual_max_seqlen,
            ) = self.qwenvl.model.visual.preprcess_grid_thw(grid_thw=image_grid_thw)
        image_embeds, deepstack_image_embeds = self.qwenvl.model.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            pos_embeds=self.pos_embeds,
            position_embeddings=self.position_embeddings,
            cu_seqlens=self.cu_seqlens,
            max_seqlen=self.visual_max_seqlen,
        )
        split_sizes = self.visual_split_sizes
        if split_sizes is None:
            split_sizes = (image_grid_thw.prod(-1) // self.qwenvl.model.visual.spatial_merge_size**2).tolist()
        image_chunks = list(torch.split(image_embeds, split_sizes))
        deepstack_chunks = [
            list(torch.split(deepstack_embeds, split_sizes)) for deepstack_embeds in deepstack_image_embeds
        ]
        image_embeds = torch.stack(image_chunks, dim=0)
        deepstack_image_embeds = [torch.stack(chunks, dim=0) for chunks in deepstack_chunks]
        return image_embeds, deepstack_image_embeds

    def embed_image(self, image: torch.Tensor, image_grid_thw: torch.LongTensor):
        return self.get_image_features(
            image,
            image_grid_thw=image_grid_thw,
        )

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwenvl.model.language_model.embed_tokens(tokens)

    def embed_special_token(self, token_id: int, batch: int, count: int, device, dtype):
        token = torch.tensor([token_id], device=device, dtype=torch.long)
        emb = self.embed_language_tokens(token).to(dtype=dtype)
        return emb.view(1, 1, 1, -1).expand(batch, count, 1, -1)

    def build_prefix_position_ids(self, input_ids, attention_mask, image_grid_thw=None, video_grid_thw=None):
        # transformers>=5.5 externalized modality detection: get_rope_index now takes an
        # explicit ``mm_token_type_ids`` (0=text, 1=image, 2=video) instead of matching
        # the placeholder token ids internally. Reconstruct it from the vision token ids.
        vlm_cfg = self.qwenvl.config
        image_token_id = getattr(vlm_cfg, "image_token_id", None)
        video_token_id = getattr(vlm_cfg, "video_token_id", None)
        mm_token_type_ids = torch.zeros_like(input_ids)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2
        position_ids, _ = self.qwenvl.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        # transformers 4.57 (which the 6B checkpoint was trained under) filled masked
        # (padding) rope positions with 1; transformers 5.5 fills them with 0. Restore the
        # 4.57 convention so cached-prefix rope matches the trained weights exactly.
        if attention_mask is not None:
            pad = attention_mask == 0
            while pad.dim() < position_ids.dim():
                pad = pad.unsqueeze(0)
            position_ids = position_ids.masked_fill(pad.expand_as(position_ids), 1)
        return position_ids

    def apply_mrope(self, query_states, key_states, position_ids=None, position_embeddings=None):
        if position_embeddings is None:
            position_embeddings = self.qwenvl.model.language_model.rotary_emb(query_states, position_ids)
        return apply_rotary_pos_emb(query_states, key_states, *position_embeddings, unsqueeze_dim=2)

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}
            if fill_kv_cache:
                past_key_values[layer_idx] = {"key_states": key_states, "value_states": value_states}
            else:
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)
        return key_states, value_states, past_key_values

    def _apply_deepstack(self, hidden_states, layer_idx, visual_pos_masks, deepstack_visual_embeds):
        """Add the level's dense deepstack delta.

        The embeds arrive pre-laid-out over the full prefix length (zeros at
        non-visual positions; see :meth:`embed_prefix`), so the injection is a
        plain shape-static add — no bool indexing, hence no nonzero()/host sync
        and CUDA-graph capturable. ``visual_pos_masks`` is kept for signature
        compatibility; its nonzero rows agree with the dense layout by
        construction. Numerically identical to the transformers
        ``_deepstack_process`` scatter it replaces (x + 0.0 == x).
        """
        if (
            deepstack_visual_embeds is not None
            and visual_pos_masks is not None
            and layer_idx < len(deepstack_visual_embeds)
        ):
            hidden_states = hidden_states + deepstack_visual_embeds[layer_idx].to(hidden_states.dtype)
        return hidden_states

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        ada_cond: list[torch.FloatTensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        # Every call site passes inputs_embeds explicitly (list of per-stream
        # embeds, entries possibly None); the default only satisfies the signature.
        assert inputs_embeds is not None
        models = [self.qwenvl.model.language_model, self.qwen_expert.model]
        num_layers = self.qwenvl.config.text_config.num_hidden_layers
        router_logits_list = []

        # mrope cos/sin depend only on position_ids (and dtype/device) — compute once
        # per forward instead of once per layer. Callers with a loop-invariant
        # position_ids (e.g. the flow-matching denoise loop) can pass a precomputed
        # ``position_embeddings`` to skip this entirely.
        if position_embeddings is None:
            rep = next(h for h in inputs_embeds if h is not None)
            position_embeddings = self.qwenvl.model.language_model.rotary_emb(rep, position_ids)

        use_gradient_checkpointing = (
            self.config.gradient_checkpointing and self.training and torch.is_grad_enabled() and not use_cache
        )

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                inputs_embeds, layer_router_logits = self._checkpointed_layer(
                    layer_idx,
                    inputs_embeds,
                    attention_mask,
                    position_embeddings,
                    ada_cond,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
                router_logits_list.extend(layer_router_logits)
                continue
            inputs_embeds, layer_router_logits, past_key_values = self._layer_forward(
                layer_idx,
                inputs_embeds,
                attention_mask,
                position_embeddings,
                past_key_values,
                use_cache,
                fill_kv_cache,
                ada_cond,
                visual_pos_masks,
                deepstack_visual_embeds,
            )
            router_logits_list.extend(layer_router_logits)

        outputs_embeds: list[torch.FloatTensor | None] = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
            elif self.config.final_norm_adanorm and i == 1:
                out_emb, _ = models[i].norm(hidden_states, ada_cond)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(models[i].norm(hidden_states))
        return outputs_embeds, past_key_values, router_logits_list

    def _layer_forward(
        self,
        layer_idx,
        inputs_embeds,
        attention_mask,
        position_embeddings,
        past_key_values,
        use_cache,
        fill_kv_cache,
        ada_cond,
        visual_pos_masks,
        deepstack_visual_embeds,
    ):
        """One dual-stream layer: per-stream QKV -> joint attention -> per-stream out/MLP."""
        models = [self.qwenvl.model.language_model, self.qwen_expert.model]
        router_logits_list = []
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                continue
            if i == 1:
                q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True, ada_cond=ada_cond)
            else:
                q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True)
            query_states.append(q)
            key_states.append(k)
            value_states.append(v)

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        query_states, key_states = self.apply_mrope(
            query_states, key_states, position_embeddings=position_embeddings
        )
        key_states, value_states, past_key_values = self.handle_kv_cache(
            key_states,
            value_states,
            layer_idx,
            past_key_values=past_key_values,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )
        att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)

        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
                continue
            end = start + hidden_states.shape[1]
            if i == 1:
                out_emb, router_logits = models[i].layers[layer_idx](
                    hidden_states,
                    att_output,
                    start,
                    end,
                    output_atten=True,
                    ada_cond=ada_cond,
                )
                if router_logits is not None:
                    router_logits_list.append(router_logits)
            else:
                out_emb = models[i].layers[layer_idx](
                    hidden_states, att_output, start, end, output_atten=True
                )
                out_emb = self._apply_deepstack(out_emb, layer_idx, visual_pos_masks, deepstack_visual_embeds)
            outputs_embeds.append(out_emb)
            start = end
        return outputs_embeds, router_logits_list, past_key_values

    @torch.compiler.disable
    def _checkpoint_layer_eager(self, *args):
        """Keep checkpoint forward/replay on the same autograd implementation.

        Regional compilation wraps decoder modules independently. If Dynamo's
        recompile limit is reached between forward and backward, replay can
        switch from AOTAutograd to eager and save a different tensor sequence.
        Disable compilation recursively for BOTH calls of this checkpointed
        layer. Vision and auxiliary heads can still be regionally compiled;
        the non-checkpointed decoder path is unchanged. Do not disable the
        checkpoint determinism check or unwrap/bypass FSDP module hooks.
        """
        return self._layer_forward(*args)

    def _checkpointed_layer(
        self,
        layer_idx,
        inputs_embeds,
        attention_mask,
        position_embeddings,
        ada_cond,
        visual_pos_masks,
        deepstack_visual_embeds,
    ):
        """Gradient-checkpointed layer step (training only, KV cache disabled)."""
        outputs_embeds, router_logits_list, _ = torch_checkpoint(
            self._checkpoint_layer_eager,
            layer_idx,
            inputs_embeds,
            attention_mask,
            position_embeddings,
            None,  # past_key_values
            False,  # use_cache
            False,  # fill_kv_cache
            ada_cond,
            visual_pos_masks,
            deepstack_visual_embeds,
            use_reentrant=False,
        )
        return outputs_embeds, router_logits_list

    def get_attention_interface(self):
        if self.config.attention_implementation == "sdpa":
            return our_sdpa_attention_forward
        if self.config.attention_implementation == "eager":
            logger.debug("Using Eager attention")
            return our_eager_attention_forward
        raise ValueError(f"Invalid attention implementation: {self.config.attention_implementation}")


class FlowMatchingV2(FlowMatchingV1):
    def __init__(self, config, rtc_processor: RTCProcessor | None = None):
        nn.Module.__init__(self)
        self.config = config
        self.rtc_processor = rtc_processor
        vlm_config = AutoConfig.from_pretrained(config.tokenizer_path)
        qwenvl_with_export_config = QwenvlWithExpertV2Config(
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            vocab_size=config.vocab_size,
            use_lm_head=config.use_lm_head,
            attention_implementation=config.attention_implementation,
            tokenizer_path=config.tokenizer_path,
            use_cache=config.use_cache,
            expert_hidden_size=config.expert_hidden_size,
            expert_intermediate_size=config.expert_intermediate_size,
            action_num_attention_heads=config.action_num_attention_heads,
            action_num_key_value_heads=config.action_num_key_value_heads,
            action_head_dim=config.action_head_dim,
            num_layers=vlm_config.text_config.num_hidden_layers,
        )
        for name in [
            "adanorm_time",
            "final_norm_adanorm",
            "precompute_grid_thw",
            "vit_attn_implementation",
            "gradient_checkpointing",
            "use_moe",
            "token_moe_layers",
            "token_num_experts",
            "token_top_k",
            "token_moe_intermediate_size",
            "token_shared_intermediate_size",
            "router_activation",
            "routed_scaling_factor",
            "use_shared_expert_gate",
        ]:
            setattr(qwenvl_with_export_config, name, getattr(config, name))
        self.qwenvl_with_expert = QwenvlWithExpertV2Model(qwenvl_with_export_config, vlm_config)
        self.proj_width = width = config.expert_hidden_size

        self.state_proj = nn.Linear(config.max_state_dim, width)
        self.action_in_proj = nn.Linear(config.max_action_dim, width)
        self.action_out_proj = nn.Linear(width, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(width * 2, width)
        self.action_time_mlp_out = nn.Linear(width, width)

        self.set_requires_grad()

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        image_grid_thw=None,
        vision_outputs=None,
    ):
        if image_grid_thw is None:
            raise ValueError("LingbotVLAV2Policy requires image_grid_thw from the Qwen3-VL image processor.")
        bsize = images.shape[0]
        device = images.device
        if images.ndim == 3:
            bsize = 1
            num_images = images.shape[0]
        else:
            num_images = images.shape[1] if images.ndim >= 4 else 1
        if images.ndim == 4:
            images = einops.rearrange(images, "b n l d -> (b n) l d")
        elif images.ndim == 5:
            images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        if image_grid_thw.ndim == 3:
            flat_grid_thw = einops.rearrange(image_grid_thw, "b n d -> (b n) d")
        else:
            flat_grid_thw = image_grid_thw

        if vision_outputs is None:
            img_emb, deepstack_embs = self.qwenvl_with_expert.embed_image(
                images,
                flat_grid_thw,
            )
        else:
            # Pre-computed vision tower outputs (e.g. from the captured vision graph in
            # the use_cudagraph_prefix_full path) — skip the eager ViT pass. Same
            # ``(b n) l d`` flat layout as embed_image returns.
            img_emb, deepstack_embs = vision_outputs
        embed_dtype = img_emb.dtype
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b n l d", b=bsize, n=num_images)
        deepstack_embs = [
            einops.rearrange(x, "(b n) l d -> b n l d", b=bsize, n=num_images) for x in deepstack_embs
        ]
        if img_masks.ndim == 1:
            img_masks = img_masks.unsqueeze(0)

        cfg = self.qwenvl_with_expert.qwenvl.config
        visual_token_id = cfg.image_token_id

        if self.config.qwen3vl_use_vision_boundaries:
            start_emb = self.qwenvl_with_expert.embed_special_token(
                cfg.vision_start_token_id, bsize, num_images, device, embed_dtype
            )
            end_emb = self.qwenvl_with_expert.embed_special_token(
                cfg.vision_end_token_id, bsize, num_images, device, embed_dtype
            )
            img_chunks = torch.cat([start_emb, img_emb, end_emb], dim=2)
            image_token_len = num_patch + 2
            image_pad_masks = einops.repeat(img_masks, "b n -> b n l", l=image_token_len)
            image_visual_masks = torch.zeros_like(image_pad_masks)
            image_visual_masks[:, :, 1 : 1 + num_patch] = einops.repeat(
                img_masks, "b n -> b n l", l=num_patch
            )
            fake_image_ids = torch.full(
                (bsize, num_images, image_token_len),
                visual_token_id,
                dtype=torch.long,
                device=device,
            )
            fake_image_ids[:, :, 0] = cfg.vision_start_token_id
            fake_image_ids[:, :, -1] = cfg.vision_end_token_id
        else:
            img_chunks = img_emb
            image_token_len = num_patch
            image_pad_masks = einops.repeat(img_masks, "b n -> b n l", l=image_token_len)
            image_visual_masks = image_pad_masks
            fake_image_ids = torch.full(
                (bsize, num_images, image_token_len),
                visual_token_id,
                dtype=torch.long,
                device=device,
            )

        img_emb = einops.rearrange(img_chunks, "b n l d -> b (n l) d")
        image_pad_masks = einops.rearrange(image_pad_masks, "b n l -> b (n l)")
        visual_pos_masks = einops.rearrange(image_visual_masks, "b n l -> b (n l)")
        fake_image_ids = einops.rearrange(fake_image_ids, "b n l -> b (n l)")

        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens).to(dtype=embed_dtype)

        embs = torch.cat([img_emb, lang_emb], dim=1)
        pad_masks = torch.cat([image_pad_masks, lang_masks], dim=1)
        prefix_input_ids = torch.cat([fake_image_ids, lang_tokens.to(device)], dim=1)
        full_visual_pos_masks = torch.cat([visual_pos_masks, torch.zeros_like(lang_masks)], dim=1)

        if self.config.vlm_causal:
            att_masks = torch.ones((bsize, embs.shape[1]), device=device, dtype=torch.bool)
        else:
            att_masks = torch.zeros((bsize, embs.shape[1]), device=device, dtype=torch.bool)

        flat_img_masks = einops.rearrange(img_masks, "b n -> (b n)")
        rope_grid_thw = flat_grid_thw[flat_img_masks]
        if rope_grid_thw.numel() == 0:
            rope_grid_thw = flat_grid_thw[:1]
        prefix_position_ids = self.qwenvl_with_expert.build_prefix_position_ids(
            prefix_input_ids,
            pad_masks.long(),
            image_grid_thw=rope_grid_thw,
            video_grid_thw=None,
        )
        # Dense (capture-safe) deepstack layout: zero-masked per-camera embeds
        # laid out over the full prefix length instead of bool-filtered rows.
        # _apply_deepstack then does a plain shape-static add; the old filtered
        # form fed transformers' _deepstack_process, whose bool indexing
        # (nonzero -> device/host sync) breaks CUDA-graph capture and graph-
        # breaks torch.compile. x + 0.0 == x keeps the injection numerically
        # identical (only -0.0 can flip to +0.0), verified bitwise in
        # bench/deepstack_dense_parity.py.
        img_visual_only = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
        img_len = img_emb.shape[1]  # boundary-wrapped image part, [b, n*l, d]
        tail_len = embs.shape[1] - img_len
        dense_deepstack = []
        for deepstack in deepstack_embs:
            level = deepstack * img_visual_only.unsqueeze(-1).to(deepstack.dtype)
            if image_token_len != num_patch:  # vision boundaries: patches sit at [1 : 1+num_patch]
                wrapped = level.new_zeros(bsize, num_images, image_token_len, level.shape[-1])
                wrapped[:, :, 1 : 1 + num_patch] = level
                level = wrapped
            level = einops.rearrange(level, "b n l d -> b (n l) d")
            if tail_len > 0:  # language tail stays zero
                level = torch.cat([level, level.new_zeros(bsize, tail_len, level.shape[-1])], dim=1)
            dense_deepstack.append(level.to(embed_dtype))

        result = (
            embs,
            pad_masks,
            att_masks,
            prefix_position_ids,
            full_visual_pos_masks,
            dense_deepstack,
        )
        return result

    def _build_full_position_ids(self, prefix_position_ids, prefix_pad_masks, suffix_pad_masks):
        valid_prefix_pos = prefix_position_ids.masked_fill(~prefix_pad_masks.unsqueeze(0), 0)
        prefix_offsets = valid_prefix_pos.amax(dim=(0, 2)) + 1
        suffix_1d = prefix_offsets[:, None] + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        suffix_1d = suffix_1d.masked_fill(~suffix_pad_masks, 1)
        suffix_position_ids = suffix_1d.unsqueeze(0).expand(3, -1, -1)
        return torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)

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
        image_grid_thw=None,
        collect_metrics=True,
    ) -> Tensor:
        dtype = state.dtype
        device = state.device
        if noise is None:
            noise = torch.randn(actions.shape, device=device, dtype=dtype)
        if time is None:
            time = self.sample_time(actions.size(0), device).to(dtype)

        x_t, u_t, time = make_flow_matching_inputs(
            actions, noise, time, convention=FlowConvention.NOISE_AT_ONE
        )

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
        )
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = self._build_full_position_ids(prefix_position_ids, prefix_pad_masks, suffix_pad_masks)

        (_, suffix_out), _, router_logits_list = self.qwenvl_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            # Training never reuses a KV cache — filling one here only wastes memory
            # (a full-sequence fp32/bf16 K/V copy per layer, discarded immediately).
            use_cache=False,
            fill_kv_cache=False,
            ada_cond=time_embs if self.config.adanorm_time else None,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        if self.config.action_fp32:
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)

        if self.config.loss_type == "fm":
            losses = F.mse_loss(u_t, v_t, reduction="none")
        elif self.config.loss_type == "L1_fm":
            losses = F.l1_loss(u_t, v_t, reduction="none")
        else:
            raise ValueError(f"Unsupported loss_type: {self.config.loss_type!r} (expected 'fm' or 'L1_fm').")

        seq_wise_loss, router_z_loss, moe_metrics = self._moe_losses_and_metrics(
            router_logits_list, losses, collect_metrics=collect_metrics
        )
        return losses, seq_wise_loss, router_z_loss, moe_metrics

    def _embed_and_fill_prefix(self, images, img_masks, lang_tokens, lang_masks, image_grid_thw):
        """Prefix half of sample_actions: embed_prefix
        (vision tower + language embedding + mrope position ids) followed by the
        36-layer KV fill. Returns exactly what the denoise loop consumes."""
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        _, past_key_values, _ = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return prefix_pad_masks, prefix_position_ids, past_key_values

    def _compile_with_mode(self, fn):
        """torch.compile with the shared mode. The default mode keeps
        triton.cudagraphs off via options (torch.compile forbids mode+options
        together; the *-no-cudagraphs modes already keep CUDA graphs off)."""
        mode = self.config.compile_predict_velocity_mode
        if mode == "default":
            return torch.compile(fn, fullgraph=False, dynamic=False, options={"triton.cudagraphs": False})
        return torch.compile(fn, fullgraph=False, dynamic=False, mode=mode)

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        image_grid_thw=None,
        inference_delay: int = 0,
        prev_chunk_left_over: Tensor | None = None,
    ) -> Tensor:
        """Do a full Qwen3-VL inference forward and compute the action."""
        if not self.config.use_cache:
            raise ValueError(
                "sample_actions requires config.use_cache=True: the denoise loop reuses "
                "the prefix KV cache, and with use_cache=False the prefix fill returns "
                "past_key_values=None. (Training forward does not go through "
                "sample_actions and is unaffected.)"
            )
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        if self.config.use_cudagraph_prefix or self.config.use_cudagraph_prefix_full:
            # CUDA-graphed prefix (see config docs). Falls back to the
            # eager prefix only for the non-CUDA / use_cache
            # guards; capture failures finish eagerly from the already-run
            # embed_prefix (no second vision pass).
            got = self._prefix_graphed(images, img_masks, lang_tokens, lang_masks, image_grid_thw)
            if got is not None:
                prefix_pad_masks, prefix_position_ids, past_key_values = got
            else:
                prefix_pad_masks, prefix_position_ids, past_key_values = self._embed_and_fill_prefix(
                    images, img_masks, lang_tokens, lang_masks, image_grid_thw
                )
        else:
            prefix_pad_masks, prefix_position_ids, past_key_values = self._embed_and_fill_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                image_grid_thw,
            )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        # Precompute the timestep schedule without any host read-back: a
        # `while time >= -dt / 2` condition forces a GPU->CPU sync every
        # denoise step, draining the pipeline and exposing host launch
        # overhead. The values below come from the same iterative `time + dt`
        # accumulation, so the schedule is bit-identical to the while form
        # (exactly num_steps entries; accumulation error stays far below the
        # old -dt/2 threshold).
        time = torch.tensor(1.0, dtype=dtype, device=device)
        time_values = []
        for _ in range(self.config.num_steps):
            time_values.append(time)
            time = time + dt
        count = 0
        predict_velocity_fn = self.predict_velocity
        if self.config.compile_predict_velocity:
            compiled = getattr(self, "_compiled_predict_velocity", None)
            if compiled is None:
                compiled = self._compile_with_mode(self.predict_velocity)
                self._compiled_predict_velocity = compiled
            predict_velocity_fn = compiled

        guided = self._rtc_enabled() and prev_chunk_left_over is not None
        if self.config.use_cudagraph_denoise and not guided:
            # The captured graph has no guidance hook; replay it only unguided.
            graphed = self._denoise_loop_graphed(
                predict_velocity_fn,
                state,
                prefix_pad_masks,
                past_key_values,
                noise,
                prefix_position_ids,
                time_values,
                dt,
            )
            if graphed is not None:
                logger.debug("Denoised %s steps (single CUDA graph replay)", len(time_values))
                return graphed
            # Shape change or capture failure — fall through to the plain loop.

        # Loop-invariant tensors (suffix 2D masks / position ids / mrope cos-sin)
        # are computed on the first predict_velocity call and reused
        # for the remaining denoise steps — they depend on the prefix masks only,
        # not on x_t or the timestep.
        denoise_cache: dict = {}
        for step_time in time_values:
            count += 1
            expanded_time = step_time.expand(bsize)

            if guided:
                assert self.rtc_processor is not None

                # No shared denoise cache: it would alias step k's tensors into step k+1's graph.
                def _pv_step(input_x_t, _et=expanded_time):
                    with _frozen_params(self):
                        return predict_velocity_fn(
                            state,
                            prefix_pad_masks,
                            past_key_values,
                            input_x_t,
                            _et,
                            prefix_position_ids=prefix_position_ids,
                        )

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=step_time,
                    original_denoise_step_partial=_pv_step,
                )
                v_t = v_t.to(x_t.dtype)  # guidance math upcasts to f32; keep x_t dtype
            else:
                v_t = predict_velocity_fn(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                    prefix_position_ids=prefix_position_ids,
                    _denoise_cache=denoise_cache,
                )

            x_t += dt * v_t
        logger.debug("Denoised %s steps%s", count, " (RTC guided)" if guided else "")
        return x_t

    @torch.no_grad()
    def _denoise_loop_graphed(
        self,
        predict_velocity_fn,
        state,
        prefix_pad_masks,
        past_key_values,
        noise,
        prefix_position_ids,
        time_values,
        dt,
    ):
        """Run the denoise loop as a single captured CUDA graph.

        Returns the denoised action chunk, or None when the graph is
        unavailable — non-CUDA input, `use_cache=False`, or a warm-up/capture
        failure — so the caller falls back to the plain loop. An
        observation-shape change drops the stale graph and re-captures.
        """
        if not state.is_cuda:
            return None
        if getattr(self, "_denoise_graph_disabled", False):
            return None
        if past_key_values is None:
            # use_cache=False: there is no KV cache to freeze into the graph.
            return None
        kv_items = tuple(sorted(past_key_values.items()))
        sig = (
            tuple(noise.shape),
            noise.dtype,
            str(noise.device),
            tuple(state.shape),
            state.dtype,
            tuple(prefix_pad_masks.shape),
            tuple(prefix_position_ids.shape),
            tuple(
                (idx, tuple(kv["key_states"].shape), tuple(kv["value_states"].shape)) for idx, kv in kv_items
            ),
            len(time_values),
            # Alias guard: when the static KV are the prefix graph's pool
            # outputs, their generation must match too — any prefix-graph
            # transition (re-capture/drop/disable) changes this term and
            # forces a re-capture here instead of a stale-alias replay.
            self._prefix_kv_gen(past_key_values),
        )
        gs = getattr(self, "_denoise_graph_state", None)
        if gs is not None and gs["sig"] != sig:
            warned = getattr(self, "_denoise_graph_warned", None)
            if warned is None:
                warned = self._denoise_graph_warned = set()
            if sig not in warned:
                logger.warning(
                    "use_cudagraph_denoise: observation shapes or prefix-KV alias changed; "
                    "re-capturing the denoise graph"
                )
                warned.add(sig)
            gs = None  # drop the stale graph (frees its private pool) and re-capture below
        if gs is None:
            gs = self._capture_denoise_graph(
                predict_velocity_fn,
                state,
                prefix_pad_masks,
                past_key_values,
                noise,
                prefix_position_ids,
                time_values,
                dt,
                sig,
            )
            if gs is None:
                return None
            self._denoise_graph_state = gs

        # Replay: copy the live prefix outputs into the static buffers the
        # graph reads, then re-execute the recorded kernel sequence. One
        # _foreach_copy_ for the whole set (76 tensors for a 36-layer prefix):
        # per-tensor copy_ launches would cost ~7ms of host time per chunk.
        # The aliased case skips the KV copies: the signature above matched
        # the identity-derived generation, so these KV *are* the prefix
        # graph's pool outputs the graph was captured reading.
        dsts = [gs["state"], gs["prefix_pad_masks"], gs["prefix_position_ids"], gs["x_t"]]
        srcs = [state, prefix_pad_masks, prefix_position_ids, noise]
        if not gs.get("kv_aliased", False):
            for idx, kv in kv_items:
                dsts.append(gs["kv"][idx]["key_states"])
                srcs.append(kv["key_states"])
                dsts.append(gs["kv"][idx]["value_states"])
                srcs.append(kv["value_states"])
        torch._foreach_copy_(dsts, srcs)
        gs["graph"].replay()
        return gs["out"].clone()

    @torch.no_grad()
    def _capture_denoise_graph(
        self,
        predict_velocity_fn,
        state,
        prefix_pad_masks,
        past_key_values,
        noise,
        prefix_position_ids,
        time_values,
        dt,
        sig,
    ):
        # When this chunk's KV are exactly the prefix graph's pool outputs,
        # alias them as the static buffers instead of cloning: the denoise
        # graph then reads the storage the prefix graph replays into, and the
        # per-chunk KV copy below disappears. The signature's generation term
        # (see _denoise_loop_graphed) keeps the alias valid.
        kv_aliased = self._prefix_kv_gen(past_key_values) is not None
        static = {
            "state": state.clone(),
            "prefix_pad_masks": prefix_pad_masks.clone(),
            "prefix_position_ids": prefix_position_ids.clone(),
            "kv": {
                idx: {
                    "key_states": kv["key_states"] if kv_aliased else kv["key_states"].clone(),
                    "value_states": kv["value_states"] if kv_aliased else kv["value_states"].clone(),
                }
                for idx, kv in past_key_values.items()
            },
            "x_t": noise.clone(),
            "dt": dt.clone(),
            "time_values": [t.clone() for t in time_values],
        }
        bsize = state.shape[0]

        def run_loop():
            # A fresh cache per call: step 1 takes the fill branch, later steps
            # the cached branch — exactly matching the plain loop. Capturing
            # with an already-populated cache would record the all-cached graph,
            # a different compiled artifact whose bf16 fusion differences
            # integrate over the denoise steps (measured as visible drift).
            cache: dict = {}
            x = static["x_t"]
            for step_time in static["time_values"]:
                v_t = predict_velocity_fn(
                    static["state"],
                    static["prefix_pad_masks"],
                    static["kv"],
                    x,
                    step_time.expand(bsize),
                    prefix_position_ids=static["prefix_position_ids"],
                    _denoise_cache=cache,
                )
                x = x + static["dt"] * v_t
            return x

        # Warm on the default stream so both compiled branches (fill +
        # cached) exist, then once on a side stream so the allocator sees
        # the loop's allocations outside the graph pool. A warm-up failure is
        # not a capture failure: disable the graph and let the plain loop
        # surface the real error.
        try:
            run_loop()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run_loop()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_denoise: warm-up pass failed (%s: %s); using the plain denoise loop",
                type(exc).__name__,
                exc,
            )
            self._denoise_graph_disabled = True  # don't retry on every call
            return None

        try:
            graph = torch.cuda.CUDAGraph()
            # A recompile mid-capture would enqueue autotuning work on the
            # capture stream; refuse it instead.
            with torch.compiler.set_stance("fail_on_recompile"), torch.cuda.graph(graph):
                static_out = run_loop()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_denoise: capture failed (%s: %s); using the plain denoise loop",
                type(exc).__name__,
                exc,
            )
            self._denoise_graph_disabled = True  # don't retry on every call
            return None

        logger.info(
            "use_cudagraph_denoise: captured the %s-step denoise loop as one CUDA graph%s",
            len(time_values),
            " (prefix-KV aliased)" if kv_aliased else "",
        )
        return {"sig": sig, "graph": graph, "out": static_out, "kv_aliased": kv_aliased, **static}

    def _prefix_llm_forward(
        self,
        prefix_embs,
        prefix_att_2d_masks,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
    ):
        """The capture-scope unit: the 36-layer KV fill only (no vision tower,
        no embed glue — their host syncs forbid capture). Sync-free since the
        dense-deepstack refactor; compilable and CUDA-graph capturable."""
        _, past_kv, _ = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return past_kv

    def _prefix_kv_gen(self, past_key_values):
        """Generation id of the prefix graph whose pool outputs are exactly
        these KV tensors, else ``None`` (fresh eager tensors).

        This identity check is the alias-safety guard: any prefix-graph state
        transition — re-capture (new pool tensors, new gen), drop, or disable
        with an eager fallback — yields tensors that fail the ``is`` test, so
        the caller's shape signature changes and the stale denoise graph is
        re-captured or re-copied instead of replaying against old KV.
        """
        pgs = getattr(self, "_prefix_graph_state", None)
        if pgs is None:
            return None
        try:
            if all(
                pgs["out"][idx][part] is kv[part]
                for idx, kv in past_key_values.items()
                for part in ("key_states", "value_states")
            ):
                return pgs["gen"]
        except KeyError:
            pass
        return None

    @torch.no_grad()
    def _vision_tower_graphed(self, images, flat_grid_thw):
        """Run the vision tower (``embed_image``) as a captured CUDA graph.

        The grid-derived metadata (pos_embeds / cu_seqlens / split_sizes / max_seqlen)
        is hoisted once into ``core``'s capture cache, so the per-call host syncs in
        ``preprcess_grid_thw`` / ``get_image_features`` become replay-time constants and
        the tower is capture-safe. Returns ``(img_emb, deepstack_embs)`` — the graph
        pool's outputs — or ``None`` on failure (caller falls back to eager).
        """
        if not images.is_cuda or getattr(self, "_vision_graph_disabled", False):
            return None
        sig = (tuple(images.shape), images.dtype, str(images.device), tuple(flat_grid_thw.shape))
        gs = getattr(self, "_vision_graph_state", None)
        if gs is not None and gs["sig"] != sig:
            self._vision_graph_gen = getattr(self, "_vision_graph_gen", 0) + 1
            gs = None
        if gs is None:
            if getattr(self, "_vision_recaptures", 0) >= 4:
                self._vision_graph_disabled = True
                logger.warning("use_cudagraph_prefix_full: vision re-capture limit reached; eager ViT")
                return None
            gs = self._capture_vision_graph(images, flat_grid_thw, sig)
            if gs is None:
                return None
            self._vision_recaptures = getattr(self, "_vision_recaptures", 0) + 1
            self._vision_graph_state = gs
        gs["images"].copy_(images)
        gs["graph"].replay()
        return gs["img_emb"], gs["deepstack"]

    @torch.no_grad()
    def _capture_vision_graph(self, images, flat_grid_thw, sig):
        core = self.qwenvl_with_expert
        prev_flag = core._capture_grid_cache
        prev_precompute = core.config.precompute_grid_thw
        prev_grid = (
            core.pos_embeds,
            core.position_embeddings,
            core.cu_seqlens,
            core.visual_split_sizes,
            core.visual_max_seqlen,
        )
        static_in = images.clone()
        try:
            # Arm + seed the capture grid cache: populate pos_embeds/cu_seqlens/
            # split_sizes/max_seqlen once so the vision tower is sync-free. The cache
            # stays live on `core` afterwards — the captured graph closes over it.
            core._capture_grid_cache = True
            core.config.precompute_grid_thw = True
            core.pos_embeds = None
            core.position_embeddings = None
            core.cu_seqlens = None
            core.visual_split_sizes = None
            core.visual_max_seqlen = None

            def run_vit():
                return core.embed_image(static_in, flat_grid_thw)

            run_vit()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run_vit()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_prefix_full: vision warm-up failed (%s: %s); eager ViT",
                type(exc).__name__,
                exc,
            )
            core._capture_grid_cache = prev_flag
            core.config.precompute_grid_thw = prev_precompute
            (
                core.pos_embeds,
                core.position_embeddings,
                core.cu_seqlens,
                core.visual_split_sizes,
                core.visual_max_seqlen,
            ) = prev_grid
            self._vision_graph_disabled = True
            return None
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.compiler.set_stance("fail_on_recompile"), torch.cuda.graph(graph):
                img_emb, deepstack = run_vit()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_prefix_full: vision capture failed (%s: %s); eager ViT",
                type(exc).__name__,
                exc,
            )
            core._capture_grid_cache = prev_flag
            core.config.precompute_grid_thw = prev_precompute
            (
                core.pos_embeds,
                core.position_embeddings,
                core.cu_seqlens,
                core.visual_split_sizes,
                core.visual_max_seqlen,
            ) = prev_grid
            self._vision_graph_disabled = True
            return None
        finally:
            core._capture_grid_cache = prev_flag
            core.config.precompute_grid_thw = prev_precompute
            # grid cache stays live for replay.
        logger.info("use_cudagraph_prefix_full: captured the vision tower as one CUDA graph")
        return {"sig": sig, "graph": graph, "images": static_in, "img_emb": img_emb, "deepstack": deepstack}

    @torch.no_grad()
    def _prefix_graphed(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        image_grid_thw,
    ):
        """Run the prefix 36-layer KV fill as one captured CUDA graph.

        Returns ``(prefix_pad_masks, prefix_position_ids, past_key_values)``
        — the same contract as :meth:`_embed_and_fill_prefix` — or ``None``
        when the graph path is unavailable (non-CUDA input, ``use_cache``
        off, or disabled after earlier failures) so the caller falls back.
        The vision tower and embed glue stay eager (their host syncs forbid
        capture); only ``qwenvl_with_expert.forward(..., fill_kv_cache=True)``
        is captured. Capture failures finish eagerly from the already-computed
        embeds (no second vision pass). The returned KV are the graph pool's
        output tensors; the denoise graph aliases them (see ``_prefix_kv_gen``).
        """
        if not images.is_cuda or getattr(self, "_prefix_graph_disabled", False):
            return None
        if not self.config.use_cache:
            return None

        if self.config.use_cudagraph_prefix_full:
            # Vision tower as its own graph (grid metadata cached on `core`); the embed
            # glue (language embed / masks / mrope ids / dense deepstack) stays eager —
            # it is light and get_rope_index's data-dependent host syncs cannot be
            # captured. The 36-layer KV fill graph below is unchanged.
            flat_grid_thw = (
                einops.rearrange(image_grid_thw, "b n d -> (b n) d")
                if image_grid_thw.ndim == 3
                else image_grid_thw
            )
            vision_outputs = self._vision_tower_graphed(images, flat_grid_thw)
            if vision_outputs is None:
                return None  # vision graph unavailable — caller falls back to eager prefix
        else:
            vision_outputs = None

        # Capture target: the thin 36-layer fill. Post-dense-deepstack this
        # region is sync-free, so it captures.
        prefix_llm_fn = self._prefix_llm_forward

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
            vision_outputs=vision_outputs,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

        def _eager_finish():
            # ``_prefix_llm_forward`` returns the KV cache alone (the 3-tuple
            # unpack belongs to the ``qwenvl_with_expert.forward`` call inside it).
            past_kv = self._prefix_llm_forward(
                prefix_embs,
                prefix_att_2d_masks,
                prefix_position_ids,
                visual_pos_masks,
                deepstack_visual_embeds,
            )
            return prefix_pad_masks, prefix_position_ids, past_kv

        sig = (
            tuple(prefix_embs.shape),
            prefix_embs.dtype,
            str(prefix_embs.device),
            tuple(prefix_att_2d_masks.shape),
            prefix_att_2d_masks.dtype,
            tuple(prefix_position_ids.shape),
            prefix_position_ids.dtype,
            tuple(visual_pos_masks.shape),
            visual_pos_masks.dtype,
            tuple(tuple(d.shape) for d in deepstack_visual_embeds),
            tuple(d.dtype for d in deepstack_visual_embeds),
        )
        gs = getattr(self, "_prefix_graph_state", None)
        if gs is not None and gs["sig"] != sig:
            warned = getattr(self, "_prefix_graph_warned", None)
            if warned is None:
                warned = self._prefix_graph_warned = set()
            if sig not in warned:
                logger.warning("use_cudagraph_prefix: prefix shapes changed; re-capturing the prefix graph")
                warned.add(sig)
            # Drop the stale graph. The aliased denoise graph still holds
            # references to the old pool tensors until its own signature
            # check (which includes the generation below) drops them.
            self._prefix_graph_gen = getattr(self, "_prefix_graph_gen", 0) + 1
            gs = None
        if gs is None:
            if getattr(self, "_prefix_recaptures", 0) >= 4:
                # Circuit breaker: shape flicker must not re-capture forever.
                self._prefix_graph_disabled = True
                logger.warning("use_cudagraph_prefix: re-capture limit reached; using the eager prefix")
                return _eager_finish()
            gs = self._capture_prefix_graph(
                prefix_llm_fn,
                prefix_embs,
                prefix_att_2d_masks,
                prefix_position_ids,
                visual_pos_masks,
                deepstack_visual_embeds,
                sig,
            )
            if gs is None:
                # Warm-up/capture failure: already warned and disabled.
                return _eager_finish()
            self._prefix_recaptures = getattr(self, "_prefix_recaptures", 0) + 1
            self._prefix_graph_state = gs

        # Replay: copy the live values into the static inputs, then replay.
        # Always executed — including right after capture, whose pool outputs
        # are uninitialized until the first replay.
        dsts = [
            gs["prefix_embs"],
            gs["att_2d"],
            gs["position_ids"],
            gs["visual_pos_masks"],
            *gs["deepstack"],
        ]
        srcs = [
            prefix_embs,
            prefix_att_2d_masks,
            prefix_position_ids,
            visual_pos_masks,
            *deepstack_visual_embeds,
        ]
        torch._foreach_copy_(dsts, srcs)
        gs["graph"].replay()
        return prefix_pad_masks, prefix_position_ids, gs["out"]

    @torch.no_grad()
    def _capture_prefix_graph(
        self,
        prefix_llm_fn,
        prefix_embs,
        prefix_att_2d_masks,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
        sig,
    ):
        static = {
            "prefix_embs": prefix_embs.clone(),
            "att_2d": prefix_att_2d_masks.clone(),
            "position_ids": prefix_position_ids.clone(),
            "visual_pos_masks": visual_pos_masks.clone(),
            "deepstack": [d.clone() for d in deepstack_visual_embeds],
        }

        def run_prefix():
            return prefix_llm_fn(
                static["prefix_embs"],
                static["att_2d"],
                static["position_ids"],
                static["visual_pos_masks"],
                static["deepstack"],
            )

        # Warm-up discipline mirrors _capture_denoise_graph: default stream,
        # side stream, synchronize — so the allocator sees the allocations
        # outside the graph pool and any failure surfaces as a fallback, not
        # a broken capture.
        try:
            run_prefix()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run_prefix()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_prefix: warm-up pass failed (%s: %s); using the eager prefix",
                type(exc).__name__,
                exc,
            )
            self._prefix_graph_disabled = True  # don't retry on every call
            return None

        try:
            graph = torch.cuda.CUDAGraph()
            with torch.compiler.set_stance("fail_on_recompile"), torch.cuda.graph(graph):
                static_out = run_prefix()
        except Exception as exc:
            logger.warning(
                "use_cudagraph_prefix: capture failed (%s: %s); using the eager prefix",
                type(exc).__name__,
                exc,
            )
            self._prefix_graph_disabled = True  # don't retry on every call
            return None

        self._prefix_graph_gen = getattr(self, "_prefix_graph_gen", 0) + 1
        logger.info(
            "use_cudagraph_prefix: captured the %s-layer prefix KV fill as one CUDA graph",
            len(static_out),
        )
        return {"sig": sig, "gen": self._prefix_graph_gen, "graph": graph, "out": static_out, **static}

    def predict_velocity(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        prefix_position_ids=None,
        _denoise_cache: dict | None = None,
    ):
        """Predict velocity at time t using cached Qwen3-VL prefix states.

        ``_denoise_cache`` (optional) is a dict that persists across the denoise
        loop: the suffix attention mask, position ids and mrope cos/sin are
        loop-invariant, so they are computed on the first step and reused afterwards.
        """
        if prefix_position_ids is None:
            raise ValueError("FlowMatchingV2.predict_velocity requires Qwen3-VL prefix_position_ids.")

        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state,
            x_t,
            timestep,
        )

        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        cache = _denoise_cache if _denoise_cache is not None else {}
        if "full_att_2d_masks" not in cache:
            batch_size = prefix_pad_masks.shape[0]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size,
                suffix_len,
                prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

            full_position_ids = self._build_full_position_ids(
                prefix_position_ids,
                prefix_pad_masks,
                suffix_pad_masks,
            )
            position_ids = full_position_ids[:, :, -suffix_len:]
            core = self.qwenvl_with_expert
            position_embeddings = core.qwenvl.model.language_model.rotary_emb(suffix_embs, position_ids)
            cache["full_att_2d_masks"] = full_att_2d_masks
            cache["position_ids"] = position_ids
            cache["position_embeddings"] = position_embeddings

        outputs_embeds, _, _ = self.qwenvl_with_expert.forward(
            attention_mask=cache["full_att_2d_masks"],
            position_ids=cache["position_ids"],
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            ada_cond=time_embs if self.config.adanorm_time else None,
            position_embeddings=cache["position_embeddings"],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        if self.config.action_fp32:
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)
        return v_t

    def _moe_losses_and_metrics(self, router_logits_list, losses, collect_metrics=True):
        router_z_loss_coeff = self.config.router_z_loss_coeff
        router_z_loss = losses.new_zeros(())
        if router_z_loss_coeff > 0 and router_logits_list:
            router_z_layer_losses = [
                torch.logsumexp(logits.float(), dim=-1).pow(2).mean() for logits in router_logits_list
            ]
            router_z_loss = router_z_loss_coeff * torch.stack(router_z_layer_losses).mean()

        # MoE blocks in layer order, matching router_logits_list.
        moe_blocks = [
            layer.mlp
            for layer in self.qwenvl_with_expert.qwen_expert.model.layers
            if isinstance(layer.mlp, Qwen2TokenMoeBlock)
        ]
        seq_wise_loss_coeff = self.config.sequence_wise_loss_coeff
        seq_wise_loss = 0
        if seq_wise_loss_coeff > 0 and router_logits_list:
            # router_logits are [B*T, E] (action-expert tokens, fixed length T per sample).
            # per_sequence -> balance experts within each sample's T tokens (DeepSeek-V3 intent);
            # global -> treat the whole B*T batch as one sequence.
            if self.config.sequence_wise_mode == "global":
                seq_lengths = None
            else:
                batch = losses.shape[0]
                tokens = router_logits_list[0].shape[0]
                seq_lengths = [tokens // batch] * batch
            seqwise_layer_losses = sequence_wise_balance_loss(
                router_logits_list=tuple(router_logits_list),
                top_k=self.config.token_top_k,
                seq_lengths=seq_lengths,
                padding_len=0,
                score_func=self.config.router_activation,
                # The loss's top-k must match the router's bias-corrected selection.
                e_score_correction_bias_list=tuple(block.e_score_correction_bias for block in moe_blocks),
            )
            if seqwise_layer_losses:
                seq_wise_loss = seq_wise_loss_coeff * torch.stack(seqwise_layer_losses).mean()

        # Monitoring-only cross-layer aggregates, computed on logging steps only.
        moe_metrics = {}
        if collect_metrics and router_logits_list:
            maxvio, minvio, minload, entropy, bias = [], [], [], [], []
            with torch.no_grad():
                for logits, block in zip(router_logits_list, moe_blocks, strict=True):
                    routing_probs = F.softmax(logits, dim=1, dtype=torch.float)
                    selected = routing_probs.argmax(dim=-1)
                    counts = F.one_hot(selected, num_classes=logits.shape[-1]).float().sum(dim=0)
                    avg_load = counts.mean().clamp(min=1e-9)
                    maxvio.append((counts.max() - avg_load) / avg_load)
                    minvio.append((avg_load - counts.min()) / avg_load)
                    minload.append(counts.min() / avg_load)
                    entropy.append(-(routing_probs * routing_probs.clamp(min=1e-9).log()).sum(dim=-1).mean())
                    bias.append(block.e_score_correction_bias.abs().max().float())
            moe_metrics = {
                "moe_summary/maxvio_avg": torch.stack(maxvio).mean(),
                "moe_summary/maxvio_max": torch.stack(maxvio).max(),
                "moe_summary/minvio_avg": torch.stack(minvio).mean(),
                "moe_summary/minvio_max": torch.stack(minvio).max(),
                "moe_summary/min_load_ratio": torch.stack(minload).min(),
                "moe_summary/has_dead_expert": (torch.stack(minload).min() == 0).float(),
                "moe_summary/entropy_avg_rank0": torch.stack(entropy).mean(),
                "moe_summary/bias_absmax": torch.stack(bias).max(),
            }
        return seq_wise_loss, router_z_loss, moe_metrics


# ============================================================================
# LeRobot policy wrapper
# ============================================================================
# The classes above are vendored/adapted from the upstream LingBot-VLA 2.0 repo
# (Robbyant/lingbot-vla-v2). The wrapper below exposes them through LeRobot's
# ``PreTrainedPolicy`` interface (train ``forward`` + rolling ``select_action``),
# mirroring the v1 ``lingbot_vla`` policy. The LeRobot dataclass config carries
# every field ``FlowMatchingV2`` reads, so it is passed straight through.


class LingbotVLAV2Policy(PreTrainedPolicy):
    """LingBot-VLA 2.0 policy for cross-embodiment robotic control.

    Couples a Qwen3-VL-4B vision-language backbone with a sparse-MoE action
    expert (pi0-style dual-stream) and predicts action chunks via flow matching.
    Native-resolution image tokens are described by ``image_grid_thw``.

    The model expects already model-ready tensors in the batch (produced by the
    lingbot_vla_v2 processor pipeline):
        - ``images``: patchified pixels for Qwen3-VL
        - ``img_masks``: per-view validity mask
        - ``observation.language.tokens`` / ``observation.language.attention_mask``:
          tokenized instruction + mask (standard TokenizerProcessorStep keys)
        - ``image_grid_thw``: (num_images, 3) temporal/height/width patch grid
        - ``observation.state``: (B, max_state_dim) padded state
        - ``action``: (B, chunk_size, max_action_dim) padded action (training)
        - optional ``joint_mask``: (B, chunk_size, max_action_dim) valid-slot mask
    """

    config_class = LeRobotLingbotVLAV2Config
    name = "lingbot_vla_v2"
    _fsdp_wrap_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock", "Qwen2DecoderLayer"]

    def __init__(self, config: LeRobotLingbotVLAV2Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = FlowMatchingV2(config, rtc_processor=self.rtc_processor)

        if not self.config.use_lm_head:
            del self.model.qwenvl_with_expert.qwenvl.lm_head
        del self.model.qwenvl_with_expert.qwen_expert.lm_head

        # The Qwen3-VL backbone builds in bfloat16 while our added projection/AdaRMSNorm
        # heads build in float32. Cast the whole model to one dtype so the dual streams
        # stay consistent (mixed dtypes raise "mat1 and mat2 must have the same dtype").
        model_dtype = self.config.dtype if isinstance(self.config.dtype, torch.dtype) else torch.bfloat16
        if model_dtype.is_floating_point:
            self.model.to(model_dtype)

        self.reset()

    def init_rtc_processor(self):
        """Build the RTC processor from ``config.rtc_config`` (called again by the rollout RTC engine)."""
        self.rtc_processor = (
            RTCProcessor(self.config.rtc_config) if self.config.rtc_config is not None else None
        )
        model = getattr(self, "model", None)
        if model is not None:
            model.rtc_processor = self.rtc_processor

    def reset(self):
        """Reset the rolling action queue used by select_action."""
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> list[dict]:
        """Param groups with a separate LR for the MoE experts (standard AdamW).

        Returns two groups: the MoE expert params (scaled LR when use_moe_expert_lr)
        and everything else at the base LR. Frozen params are excluded — with PEFT that
        is ~0.2B adapter params instead of 6B.
        """
        cfg = self.config
        expert_lr_scale = 1.0
        if cfg.use_moe and cfg.use_moe_expert_lr and cfg.token_top_k > 0:
            expert_lr_scale = (cfg.token_num_experts / cfg.token_top_k) ** 0.5

        expert_params = []
        base_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if ".mlp.experts." in name:
                expert_params.append(p)
            else:
                base_params.append(p)

        groups = []
        if base_params:
            groups.append({"params": base_params})
        if expert_params:
            groups.append({"params": expert_params, "lr": cfg.optimizer_lr * expert_lr_scale})
        if not groups:
            raise ValueError("No trainable LingBot parameters")
        return groups

    # ==================== PEFT (LoRA) integration ====================
    # The community PEFT path (lerobot `--peft.*` CLI → wrap_with_peft →
    # save/resume via adapter checkpoints) works with this policy through the two
    # subclass hooks below. Targeting notes specific to this architecture:
    # - Both the Qwen3-VL LLM and the action expert name their attention
    #   projections q/k/v/o_proj, so one suffix list covers both streams.
    # - The vision tower uses a fused `qkv` projection and is not matched (it is
    #   frozen via freeze_vision_encoder regardless).
    # - The MoE router (`...mlp.gate`, a hidden×num_experts Linear) stays fully
    #   trainable via modules_to_save: freezing the routing distribution hurts
    #   fine-tuning on new robot data, and it is tiny.
    # - The routed experts are stored as fused grouped GEMMs (Qwen2FusedExperts,
    #   plain Parameters — not nn.Linear), so stock LoRA cannot target them.

    def _get_default_peft_targets(self) -> dict[str, Any] | None:
        return {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "modules_to_save": ["gate"],
        }

    def _validate_peft_config(self, peft_config) -> None:
        super()._validate_peft_config(peft_config)
        targets = getattr(peft_config, "target_modules", None) or []
        if isinstance(targets, str):
            targets = [targets]
        mlp_targets = {"gate_proj", "up_proj", "down_proj"} & set(targets)
        if mlp_targets:
            logger.warning(
                "PEFT target_modules %s only match the shared-expert MLP (nn.Linear); the routed "
                "experts use fused grouped-GEMM storage (Qwen2FusedExperts) and will NOT be adapted.",
                sorted(mlp_targets),
            )
        if not self.config.gradient_checkpointing:
            logger.warning(
                "LoRA adapters are inside every decoder layer, so backward still traverses the "
                "frozen backbone's activations. Consider --policy.gradient_checkpointing=true to "
                "cut activation memory."
            )

    def _extract_model_inputs(self, batch: dict):
        dtype = next(self.parameters()).dtype
        images = batch["images"].to(dtype=dtype)
        img_masks = batch["img_masks"]
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = batch[OBS_STATE].to(dtype=dtype)
        state = F.pad(state, (0, self.config.max_state_dim - state.shape[-1]))
        image_grid_thw = batch.get("image_grid_thw")
        return images, img_masks, lang_tokens, lang_masks, state, image_grid_thw

    def forward(self, batch: dict, reduction: str = "mean") -> tuple[Tensor, dict]:
        """Training forward pass returning the flow-matching loss (per sample with ``reduction="none"``)."""
        images, img_masks, lang_tokens, lang_masks, state, image_grid_thw = self._extract_model_inputs(batch)
        actions = batch[ACTION].to(dtype=state.dtype)
        action_dim = actions.shape[-1]
        actions = F.pad(actions, (0, self.config.max_action_dim - action_dim))

        # MoE monitoring metrics (.item() syncs) only on logging steps.
        self._train_step_count = getattr(self, "_train_step_count", 0) + 1
        collect_metrics = self._train_step_count % max(1, self.config.moe_metrics_interval) == 0

        losses, seq_wise_loss, router_z_loss, moe_metrics = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise=batch.get("noise"),
            time=batch.get("time"),
            image_grid_thw=image_grid_thw,
            collect_metrics=collect_metrics,
        )

        dims = (1, 2) if reduction == "none" else None
        joint_mask = batch.get("joint_mask")
        if joint_mask is not None:
            loss_vla = (losses * joint_mask).sum(dim=dims) / joint_mask.sum(dim=dims).clamp(min=1)
        else:
            loss_vla = losses[:, :, :action_dim].mean(dim=dims)

        loss_dict: dict = {
            "l1_loss" if self.config.loss_type == "L1_fm" else "l2_loss": loss_vla.mean().item()
        }
        total_loss = loss_vla
        for loss_name, term in (
            ("seq_wise_loss", seq_wise_loss),
            ("router_z_loss", router_z_loss),
        ):
            if torch.is_tensor(term):
                loss_dict[loss_name] = term.item()
                total_loss = total_loss + term
        loss_dict.update({k: v.item() for k, v in moe_metrics.items()})
        loss_dict["loss"] = total_loss.mean().item()
        return total_loss, loss_dict

    def supports_rtc(self) -> bool:
        return True

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict,
        noise: Tensor | None = None,
        inference_delay: int = 0,
        prev_chunk_left_over: Tensor | None = None,
    ) -> Tensor:
        """Run flow-matching denoising and return the canonical action chunk (B, chunk, max_action_dim).

        The output stays in the normalized canonical space; the policy postprocessor
        inverts the slot mapping, unnormalizes, and re-absolutizes the actions (see
        ``processor_lingbot_vla_v2``).

        RTC args (from the rollout RTC engine): ``prev_chunk_left_over`` is the
        normalized leftover prefix of the previous chunk (already truncated to the
        execution horizon); ``inference_delay`` is the chunk's reaction lag in action
        steps. Passing ``None`` (default) reproduces the plain unguided sampling.
        """
        self.eval()
        images, img_masks, lang_tokens, lang_masks, state, image_grid_thw = self._extract_model_inputs(batch)
        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            image_grid_thw=image_grid_thw,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
        )
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict, noise: Tensor | None = None) -> Tensor:
        """Select a single action for environment execution, buffering chunks in a queue."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()
