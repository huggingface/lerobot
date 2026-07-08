import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.models.auto import CONFIG_MAPPING
from transformers.cache_utils import Cache
from transformers.utils import logging

from .configuration_lingbot_vla import LingbotVLAV2Config
from .qwen3vl_in_vla import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextModel,
    Qwen3VLPreTrainedModel,
    apply_rotary_pos_emb,
)
from .modeling_lingbot_vla import (
    AdaRMSNorm,
    FixAdaRMSNorm,
    replace_lnorm_with_adanorm,
    FlowMatching as FlowMatchingV1,
)
from .utils import (
    block_suffix_to_fv_,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    our_eager_attention_forward,
    our_sdpa_attention_forward,
    prefix_query_segments,
    prefix_query_token_spans,
    sample_beta,
)
from .flex_attention import build_block_mask, flex_attention_forward, flex_attention_with_block_mask

LingBotVLAWeightLoader = None  # noqa: N816  # lerobot PreTrainedPolicy handles weight loading
from .moe_loss import sequence_wise_balance_loss as triton_sequence_wise_balance_loss
from .qwen2_action_expert import (
    Qwen2ForCausalLM,
    Qwen2TokenMoeBlock,
    Qwen2FusedExperts,
    FixQwen2RMSNorm,
)

try:
    from dinov3.hub.backbones import dinov3_vitb16
except Exception:
    dinov3_vitb16 = None


logger = logging.get_logger(__name__)


class QwenvlWithExpertV2Config(PretrainedConfig):
    model_type = "QwenvlWithExpertV2Model"

    def __init__(
        self,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        vocab_size: int = 0,
        use_lm_head: bool = False,
        attention_implementation: str = "flex_cached",
        tokenizer_path: str | None = None,
        enable_expert_vision: bool = False,
        expert_vision_type: str | None = None,
        use_cache: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        action_num_attention_heads: int = 32,
        action_num_key_value_heads: int = 8,
        action_head_dim: int = 128,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.tokenizer_path = tokenizer_path
        self.enable_expert_vision = enable_expert_vision
        self.expert_vision_type = expert_vision_type
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        self.action_num_attention_heads = action_num_attention_heads
        self.action_num_key_value_heads = action_num_key_value_heads
        self.action_head_dim = action_head_dim
        num_layers = 36

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
        print(
            "=====Action Expert V2 init "
            f"{num_layers} Layers, hidden={expert_hidden_size}, "
            f"q_heads={action_num_attention_heads}, kv_heads={action_num_key_value_heads}, "
            f"head_dim={action_head_dim}.====="
        )
        super().__init__(**kwargs)


class QwenvlWithExpertV2Model(PreTrainedModel):
    config_class = QwenvlWithExpertV2Config

    def __init__(self, config: QwenvlWithExpertV2Config, eval=False):
        super().__init__(config=config)
        self.config = config
        # Map our attention_implementation to a transformers-valid attn class for the
        # HF model instantiation. "fa2" -> flash_attention_2; everything else (eager /
        # flex / flex_cached) builds with "eager" — the flex paths override attention in
        # the custom forward, and eager is required where flash-attn is absent (Jetson,
        # CPU, this A100 box without the flash_attn package).
        hf_attn = "flash_attention_2" if self.config.attention_implementation == "fa2" else "eager"
        vlm_config = AutoConfig.from_pretrained(self.config.tokenizer_path)
        if self.config.vocab_size not in (0, 257152):
            vlm_config.text_config.vocab_size = self.config.vocab_size
        vlm_config._attn_implementation = hf_attn
        vlm_config.text_config._attn_implementation = hf_attn
        vlm_config.vision_config._attn_implementation = self.config.vit_attn_implementation
        self.qwenvl = Qwen3VLForConditionalGeneration._from_config(vlm_config)
        if self.config.use_lm_head:
            self.qwenvl.tie_weights()

        self.config.qwen_expert_config._attn_implementation = hf_attn
        self.qwen_expert = Qwen2ForCausalLM._from_config(self.config.qwen_expert_config, eval=eval)

        if getattr(self.config, "adanorm_time", False):
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

        del self.qwen_expert.model.embed_tokens
        if self.config.enable_expert_vision:
            if dinov3_vitb16 is None:
                raise ImportError("dinov3 is required when enable_expert_vision=True")
            if "dinov3_vitb16" in self.config.expert_vision_type:
                self.expert_visual = dinov3_vitb16(pretrained=False)
            self.expert_visual_mlp = nn.Sequential(
                nn.Linear(self.expert_visual.embed_dim, self.expert_visual.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.expert_visual.embed_dim * 2, self.config.qwen_expert_config.hidden_size),
            )

        self.attention_interface = self.get_attention_interface()
        self.set_requires_grad()

    def _install_moe_blocks(self):
        if not getattr(self.config, "use_moe", False):
            return
        bias_update_speed = getattr(self.config, "bias_update_speed", 0.001)
        hidden_size = self.config.qwen_expert_config.hidden_size
        token_moe_layers = getattr(self.config, "token_moe_layers", None) or []

        _moe_impl = getattr(self.config, "_moe_implementation", None)

        if token_moe_layers:
            token_config = CONFIG_MAPPING["qwen2_moe"](
                num_experts=getattr(self.config, "token_num_experts", 32),
                num_experts_per_tok=getattr(self.config, "token_top_k", 1),
                norm_topk_prob=True,
                hidden_size=hidden_size,
                moe_intermediate_size=getattr(self.config, "token_moe_intermediate_size", 256),
                shared_expert_intermediate_size=getattr(self.config, "token_shared_intermediate_size", 256),
                output_router_logits=False,
            )
            token_config.bias_update_speed = bias_update_speed
            token_config._moe_implementation = _moe_impl
            token_config.router_activation = getattr(self.config, "router_activation", "softmax")
            token_config.routed_scaling_factor = getattr(self.config, "routed_scaling_factor", 1.0)
            token_config.use_shared_expert_gate = getattr(self.config, "use_shared_expert_gate", True)
            for idx in token_moe_layers:
                self.qwen_expert.model.layers[idx].mlp = Qwen2TokenMoeBlock(token_config)

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
        precompute_grid_thw = getattr(self.config, "precompute_grid_thw", False)
        if precompute_grid_thw and self.position_embeddings is None:
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

    def apply_mrope(self, query_states, key_states, position_ids):
        position_embeddings = self.qwenvl.model.language_model.rotary_emb(query_states, position_ids)
        return apply_rotary_pos_emb(query_states, key_states, *position_embeddings, unsqueeze_dim=2)

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
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
        if (
            deepstack_visual_embeds is not None
            and visual_pos_masks is not None
            and layer_idx < len(deepstack_visual_embeds)
        ):
            hidden_states = self.qwenvl.model.language_model._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )
        return hidden_states

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        vlm_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        ada_cond: List[torch.FloatTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ):
        models = [self.qwenvl.model.language_model, self.qwen_expert.model]
        num_layers = self.qwenvl.config.text_config.num_hidden_layers
        action_num_layers = self.config.qwen_expert_config.num_hidden_layers
        router_logits_list = []

        assert action_num_layers == num_layers, (
            "Action expert and VLM must have the same number of layers "
            f"(got action={action_num_layers}, vlm={num_layers})."
        )

        for layer_idx in range(num_layers):
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
                query_states.append(q.float())
                key_states.append(k.float())
                value_states.append(v.float())

            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)
            query_states, key_states = self.apply_mrope(query_states, key_states, position_ids)
            key_states, value_states, past_key_values = self.handle_kv_cache(
                key_states,
                value_states,
                layer_idx,
                past_key_values=past_key_values,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
            )
            if self.config.attention_implementation == "flex_cached":
                if layer_idx == 0:
                    _full_len = query_states.shape[1]
                    _full_block_mask = build_block_mask(
                        attention_mask,
                        self.qwenvl.config.text_config.num_attention_heads,
                        _full_len,
                        _full_len,
                    )
                att_output = flex_attention_with_block_mask(
                    query_states, key_states, value_states, _full_block_mask, query_states.shape[1]
                )
            else:
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
                    out_emb = self._apply_deepstack(
                        out_emb, layer_idx, visual_pos_masks, deepstack_visual_embeds
                    )
                outputs_embeds.append(out_emb)
                start = end
            inputs_embeds = outputs_embeds

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
            elif self.config.final_norm_adanorm and i == 1:
                out_emb, _ = models[i].norm(hidden_states, ada_cond)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(models[i].norm(hidden_states))
        return outputs_embeds, past_key_values, router_logits_list

    def get_attention_interface(self):
        if self.config.attention_implementation == "flex":
            print("=====Using Flex Attn=====")
            return flex_attention_forward
        if self.config.attention_implementation == "flex_cached":
            print("=====Using Flex Cached (prebuilt BlockMask) Attn=====")
            return flex_attention_forward
        if self.config.attention_implementation == "sdpa":
            return our_sdpa_attention_forward
        if self.config.attention_implementation == "eager":
            print("=====Using Eager Attn=====")
            return our_eager_attention_forward
        raise ValueError(f"Invalid attention implementation: {self.config.attention_implementation}")


class FlowMatchingV2(FlowMatchingV1):
    def __init__(self, config, eval):
        nn.Module.__init__(self)
        self.config = config
        qwenvl_with_export_config = QwenvlWithExpertV2Config(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            vocab_size=getattr(self.config, "vocab_size", 0),
            use_lm_head=getattr(self.config, "use_lm_head", False),
            attention_implementation=self.config.attention_implementation,
            tokenizer_path=self.config.tokenizer_path,
            enable_expert_vision=self.config.enable_expert_vision,
            expert_vision_type=self.config.expert_vision_type,
            use_cache=getattr(self.config, "use_cache", True),
            expert_hidden_size=getattr(self.config, "expert_hidden_size", 768),
            expert_intermediate_size=getattr(self.config, "expert_intermediate_size", 2752),
            action_num_attention_heads=getattr(self.config, "action_num_attention_heads", 32),
            action_num_key_value_heads=getattr(self.config, "action_num_key_value_heads", 8),
            action_head_dim=getattr(self.config, "action_head_dim", 128),
        )
        for name in [
            "adanorm_time",
            "final_norm_adanorm",
            "precompute_grid_thw",
            "vit_attn_implementation",
            "use_moe",
            "bias_update_speed",
            "token_moe_layers",
            "token_num_experts",
            "token_top_k",
            "token_moe_intermediate_size",
            "token_shared_intermediate_size",
            "router_activation",
            "routed_scaling_factor",
            "use_shared_expert_gate",
            "_moe_implementation",
        ]:
            if hasattr(config, name):
                setattr(qwenvl_with_export_config, name, getattr(config, name))
        self.qwenvl_with_expert = QwenvlWithExpertV2Model(qwenvl_with_export_config, eval)
        self.config.proj_width = qwenvl_with_export_config.qwen_expert_config.hidden_size
        self.config.initializer_range = getattr(
            qwenvl_with_export_config.qwen_expert_config, "initializer_range", None
        )

        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.config.align_params = getattr(self.config, "align_params", None) or {}
        if self.config.align_params != {}:
            self.steps = 0
            self.use_depth_align = True
            self.init_depth_heads(self.config.align_params)
            self.use_future_video = self.config.align_params.get("use_future_video", False)
            if self.use_future_video:
                self.init_video_heads(self.config.align_params)
        else:
            self.use_depth_align = False
            self.use_future_video = False
            self.use_future_video_patch = False
            self.use_current_video_patch = False
            self.use_current_shared_task_proj = False
            self.use_future_video_cls = False
            self.use_shared_future_task_proj = False
            self.future_video_share_future_depth_query = False
            self.block_future_depth_to_action = False

        self.set_requires_grad()

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        image_grid_thw=None,
    ):
        if image_grid_thw is None:
            raise ValueError("LingbotVlaV2Policy requires image_grid_thw from the Qwen3-VL image processor.")
        bsize = images.shape[0]
        device = images.device
        dtype = images.dtype
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

        img_emb, deepstack_embs = self.qwenvl_with_expert.embed_image(
            images,
            flat_grid_thw,
        )
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

        if getattr(self.config, "qwen3vl_use_vision_boundaries", True):
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

        if self.use_depth_align and self.align_type == "query":

            def _get_align_tokens(tokens):
                tk_weights = tokens.view(
                    self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1]
                )
                tk_weights = tk_weights.mean(dim=1)
                return tk_weights

            align_pad_masks = torch.ones(bsize, self.num_task_tokens, device=device, dtype=lang_masks.dtype)
            fake_align_ids = torch.full(
                (bsize, self.num_task_tokens), cfg.text_config.eos_token_id, dtype=torch.long, device=device
            )

            current_task = _get_align_tokens(self.depth_align_embs)
            if (
                getattr(self, "use_future_video", False)
                and getattr(self, "use_current_video_patch", False)
                and getattr(self, "use_current_shared_task_proj", False)
            ):
                current_video_task = _get_align_tokens(self.current_video_align_embs)
                current_task = self.current_shared_task_proj(
                    torch.cat([current_task, current_video_task], dim=-1)
                )
            align_embs = current_task.repeat(img_emb.size(0), 1, 1).to(img_emb.device, img_emb.dtype)
            parts = [img_emb]
            masks = [image_pad_masks]
            input_ids = [fake_image_ids]
            visual_masks = [visual_pos_masks]

            def _append(
                tokens,
                token_masks,
                token_ids,
                token_visual_masks=None,
            ):
                parts.append(tokens)
                masks.append(token_masks)
                input_ids.append(token_ids)
                if token_visual_masks is None:
                    token_visual_masks = torch.zeros_like(token_masks)
                visual_masks.append(token_visual_masks)

            future_align_embs = None
            if self.use_future_depth:
                future_task = _get_align_tokens(self.future_depth_align_embs)
                if (
                    getattr(self, "use_future_video", False)
                    and getattr(self, "use_future_video_patch", True)
                    and getattr(self, "future_video_share_future_depth_query", False)
                    and getattr(self, "use_shared_future_task_proj", False)
                ):
                    future_video_task = _get_align_tokens(self.future_video_align_embs)
                    future_task = self.future_shared_task_proj(
                        torch.cat([future_task, future_video_task], dim=-1)
                    )
                future_align_embs = future_task.repeat(img_emb.size(0), 1, 1).to(
                    img_emb.device, img_emb.dtype
                )

            if (
                not self.use_future_depth
                and getattr(self, "use_future_video", False)
                and getattr(self, "future_video_share_future_depth_query", False)
            ):
                raise ValueError("share_future_depth_query=True requires depth.use_future_depth=True.")

            for segment_name in prefix_query_segments(
                use_depth_align=True,
                use_future_depth=self.use_future_depth,
                use_future_video=getattr(self, "use_future_video", False),
                use_future_video_cls=getattr(self, "use_future_video_cls", False),
                use_future_video_patch=getattr(self, "use_future_video_patch", True),
                future_video_share_future_depth_query=getattr(
                    self,
                    "future_video_share_future_depth_query",
                    False,
                ),
            ):
                if segment_name == "language":
                    _append(
                        lang_emb,
                        lang_masks,
                        lang_tokens.to(device),
                    )
                elif segment_name == "current_depth":
                    _append(align_embs, align_pad_masks, fake_align_ids)
                elif segment_name == "future_video_cls":
                    future_video_cls_align_emb = self.future_video_cls_align_emb.weight.repeat(
                        img_emb.size(0), 1, 1
                    ).to(img_emb.device, img_emb.dtype)
                    cls_align_pad_masks = torch.ones(
                        bsize,
                        1,
                        device=device,
                        dtype=lang_masks.dtype,
                    )
                    fake_cls_align_ids = torch.full(
                        (bsize, 1),
                        cfg.text_config.eos_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    _append(future_video_cls_align_emb, cls_align_pad_masks, fake_cls_align_ids)
                elif segment_name == "future_video":
                    future_video_align_embs = (
                        _get_align_tokens(self.future_video_align_embs)
                        .repeat(img_emb.size(0), 1, 1)
                        .to(img_emb.device, img_emb.dtype)
                    )
                    _append(future_video_align_embs, align_pad_masks, fake_align_ids)
                elif segment_name == "future_depth":
                    _append(future_align_embs, align_pad_masks, fake_align_ids)
                else:
                    raise ValueError(f"Unsupported prefix query segment: {segment_name}")

            embs = torch.cat(parts, dim=1)
            pad_masks = torch.cat(masks, dim=1)
            prefix_input_ids = torch.cat(input_ids, dim=1)
            full_visual_pos_masks = torch.cat(visual_masks, dim=1)
        else:
            embs = torch.cat([img_emb, lang_emb], dim=1)
            pad_masks = torch.cat([image_pad_masks, lang_masks], dim=1)
            prefix_input_ids = torch.cat([fake_image_ids, lang_tokens.to(device)], dim=1)
            full_visual_pos_masks = torch.cat([visual_pos_masks, torch.zeros_like(lang_masks)], dim=1)

        if getattr(self.config, "vlm_causal", False):
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
        filtered_deepstack = []
        img_visual_only = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
        for deepstack in deepstack_embs:
            filtered_deepstack.append(deepstack[img_visual_only])

        result = (
            embs,
            pad_masks,
            att_masks,
            prefix_position_ids,
            full_visual_pos_masks,
            filtered_deepstack,
        )
        return result

    def _build_full_position_ids(self, prefix_position_ids, prefix_pad_masks, suffix_pad_masks):
        valid_prefix_pos = prefix_position_ids.masked_fill(~prefix_pad_masks.unsqueeze(0), 0)
        prefix_offsets = valid_prefix_pos.amax(dim=(0, 2)) + 1
        suffix_1d = prefix_offsets[:, None] + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        suffix_1d = suffix_1d.masked_fill(~suffix_pad_masks, 1)
        suffix_position_ids = suffix_1d.unsqueeze(0).expand(3, -1, -1)
        return torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)

    def _current_depth_task_tokens(self, hidden_states, num_images=3):
        query_spans = prefix_query_token_spans(
            prefix_len=hidden_states.shape[1],
            num_task_tokens=self.num_task_tokens,
            use_depth_align=True,
            use_future_depth=getattr(self, "use_future_depth", False),
            use_future_video=getattr(self, "use_future_video", False),
            use_future_video_cls=getattr(self, "use_future_video_cls", False),
            use_future_video_patch=getattr(self, "use_future_video_patch", True),
            future_video_share_future_depth_query=getattr(
                self,
                "future_video_share_future_depth_query",
                False,
            ),
        )
        start, end = query_spans["current_depth"]
        return hidden_states[:, start:end, :]

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
        loss_type="fm",
        depth_targets=None,
        image_grid_thw=None,
        future_depth_targets=None,
        future_video_targets=None,
        future_video_cls_targets=None,
        future_video_current_patch=None,
    ) -> Tensor:
        dtype = state.dtype
        device = state.device
        if noise is None:
            noise = torch.randn(actions.shape, device=device, dtype=dtype)
        if time is None:
            time = self.sample_time(actions.size(0), device).to(dtype)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

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
        prefix_len = prefix_pad_masks.shape[1]
        if self.block_future_depth_to_action:
            att_2d_masks = block_suffix_to_fv_(
                att_2d_masks,
                suffix_row_start=prefix_len,
                prefix_len=prefix_len,
                num_task_tokens=self.num_task_tokens,
            )

        att_2d_masks = self._block_suffix_to_future_video_if_enabled_(
            att_2d_masks,
            suffix_row_start=prefix_len,
            prefix_len=prefix_len,
        )
        position_ids = self._build_full_position_ids(prefix_position_ids, prefix_pad_masks, suffix_pad_masks)

        (outputs_embeds, suffix_out), _, router_logits_list = self.qwenvl_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            vlm_position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        align_metrics = {}
        if self.config.align_params != {}:
            loss_depth, loss_future_depth, depth_preds, future_depth_preds = self.depth_emb_forward(
                outputs_embeds,
                depth_targets,
                img_masks,
                future_depth_targets,
            )
            loss_depth = loss_depth * self.config.align_params["depth_loss_weight"]
            loss_future_depth = loss_future_depth * self.config.align_params.get(
                "future_depth_loss_weight", 1.0
            )
            loss_future_video = 0
            future_video_preds = None
            current_video_preds = None
            if getattr(self, "use_future_video", False):
                loss_video, future_video_preds, video_metrics = self.video_emb_forward(
                    outputs_embeds,
                    future_video_targets,
                    future_video_cls_targets=future_video_cls_targets,
                    future_video_current_patch=future_video_current_patch,
                )
                video_total_loss = loss_video
                if getattr(self, "use_current_video_patch", False) and future_video_current_patch is not None:
                    current_video_loss, current_video_preds, current_video_metrics = (
                        self.current_video_emb_forward(
                            outputs_embeds,
                            future_video_current_patch,
                        )
                    )
                    video_total_loss = video_total_loss + current_video_loss
                    video_metrics.update(current_video_metrics)
                    video_metrics["align/current_video_loss"] = current_video_loss.detach()
                video_cfg = self.config.align_params.get("video", {})
                video_weight = video_cfg.get(
                    "future_video_loss_weight",
                    self.config.align_params.get(
                        "future_video_loss_weight",
                        self.config.align_params["depth_loss_weight"],
                    ),
                )
                loss_future_video = video_total_loss * video_weight
                align_metrics.update(video_metrics)
                if "align/current_video_loss" in align_metrics:
                    align_metrics["align/current_video_loss_weighted"] = (
                        align_metrics["align/current_video_loss"] * video_weight
                    )
                align_metrics["align/future_video_loss"] = loss_video.detach()
                align_metrics["align/future_video_loss_weighted"] = (loss_video * video_weight).detach()
                align_metrics["align/video_loss"] = video_total_loss.detach()
                align_metrics["align/video_loss_weighted"] = loss_future_video.detach()
            self.steps += 1
        else:
            loss_depth = 0
            loss_future_depth = 0
            loss_future_video = 0
            depth_preds = None
            future_depth_preds = None
            future_video_preds = None
            current_video_preds = None

        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if getattr(self.config, "action_fp32", False):
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)

        if loss_type == "fm":
            losses = F.mse_loss(u_t, v_t, reduction="none")
        elif loss_type == "L1_fm":
            losses = F.l1_loss(u_t, v_t, reduction="none")

        seq_wise_loss, router_z_loss, moe_metrics = self._moe_losses_and_metrics(router_logits_list, losses)
        if align_metrics:
            moe_metrics.update(align_metrics)
        return (
            losses,
            loss_depth,
            loss_future_depth,
            loss_future_video,
            depth_preds,
            seq_wise_loss,
            router_z_loss,
            moe_metrics,
            future_depth_preds,
            future_video_preds,
            current_video_preds,
        )

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        image_grid_thw=None,
    ) -> Tensor:
        """Do a full Qwen3-VL inference forward and compute the action."""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

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
            vlm_position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        count = 0
        predict_velocity_fn = self.predict_velocity
        if getattr(self, "_use_compile_predict_velocity", False):
            predict_velocity_fn = getattr(self, "_compiled_predict_velocity", None)
            if predict_velocity_fn is None:
                predict_velocity_fn = torch.compile(
                    self.predict_velocity,
                    fullgraph=False,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
                self._compiled_predict_velocity = predict_velocity_fn

        while time >= -dt / 2:
            count += 1
            expanded_time = time.expand(bsize)
            v_t = predict_velocity_fn(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                prefix_position_ids=prefix_position_ids,
            )

            x_t += dt * v_t
            time += dt
        print(f"Denoise {count} steps")
        return x_t

    def predict_velocity(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        prefix_position_ids=None,
    ):
        """Predict velocity at time t using cached Qwen3-VL prefix states."""
        if prefix_position_ids is None:
            raise ValueError("FlowMatchingV2.predict_velocity requires Qwen3-VL prefix_position_ids.")

        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state,
            x_t,
            timestep,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size,
            suffix_len,
            prefix_len,
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        if self.block_future_depth_to_action:
            # Query rows here are all suffix (state/action), so row start is 0.
            full_att_2d_masks = block_suffix_to_fv_(
                full_att_2d_masks,
                suffix_row_start=0,
                prefix_len=prefix_len,
                num_task_tokens=self.num_task_tokens,
            )
        full_att_2d_masks = self._block_suffix_to_future_video_if_enabled_(
            full_att_2d_masks,
            suffix_row_start=0,
            prefix_len=prefix_len,
        )

        full_position_ids = self._build_full_position_ids(
            prefix_position_ids,
            prefix_pad_masks,
            suffix_pad_masks,
        )
        position_ids = full_position_ids[:, :, -suffix_len:]

        outputs_embeds, _, _ = self.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if getattr(self.config, "action_fp32", False):
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)
        return v_t

    def _moe_losses_and_metrics(self, router_logits_list, losses):
        router_z_loss_coeff = getattr(self.config, "router_z_loss_coeff", 0)
        router_z_loss = losses.new_zeros(())
        router_z_layer_losses = None  # per-layer raw z-loss (pre-coeff), for monitoring
        if router_z_loss_coeff > 0 and router_logits_list:
            router_z_layer_losses = [
                torch.logsumexp(logits.float(), dim=-1).pow(2).mean() for logits in router_logits_list
            ]
            router_z_loss = router_z_loss_coeff * torch.stack(router_z_layer_losses).mean()

        seq_wise_loss_coeff = getattr(self.config, "sequence_wise_loss_coeff", 0)
        seq_wise_loss = 0
        seqwise_layer_losses = None  # per-layer raw seq-wise balance loss (pre-coeff), for monitoring
        if seq_wise_loss_coeff > 0 and router_logits_list:
            # router_logits are [B*T, E] (action-expert tokens, fixed length T per sample).
            # per_sequence -> balance experts within each sample's T tokens (DeepSeek-V3 intent);
            # global -> treat the whole B*T batch as one sequence.
            mode = getattr(self.config, "sequence_wise_mode", "per_sequence")
            score_func = getattr(self.config, "router_activation", "softmax")
            if mode == "global":
                seq_lengths = None
            else:
                B = losses.shape[0]
                N = router_logits_list[0].shape[0]
                seq_lengths = [N // B] * B
            seqwise_layer_losses = triton_sequence_wise_balance_loss(
                router_logits_list=tuple(router_logits_list),
                top_k=getattr(self.config, "token_top_k", 4),
                seq_lengths=seq_lengths,
                padding_len=0,
                score_func=score_func,
            )
            if seqwise_layer_losses:
                seq_wise_loss = seq_wise_loss_coeff * torch.stack(seqwise_layer_losses).mean()

        moe_metrics = {}
        if router_logits_list:
            token_moe_layers_list = sorted(getattr(self.config, "token_moe_layers", None) or [])
            all_moe_indices = token_moe_layers_list
            token_expert_counts = []
            # Per-layer token-MoE stats, collected for moe_summary/* cross-layer aggregates.
            tok_maxvio, tok_minvio, tok_minload, tok_entropy, tok_sigmoid = [], [], [], [], []
            tok_bias = []  # per-layer max(|e_score_correction_bias|) (loss-free); >1 -> bias dominates sigmoid score
            any_dead = None  # OR-accumulated bool: any token-MoE layer with a 0-count expert
            with torch.no_grad():
                for i, logits in enumerate(router_logits_list):
                    layer_id = all_moe_indices[i] if i < len(all_moe_indices) else i
                    num_experts = logits.shape[-1]
                    routing_probs = F.softmax(logits, dim=1, dtype=torch.float)
                    moe_block = self.qwenvl_with_expert.qwen_expert.model.layers[layer_id].mlp
                    if hasattr(moe_block, "last_tokens_per_expert"):
                        # Global (all-reduced), biased, true top-k load from the load-balance hook.
                        counts = moe_block.last_tokens_per_expert.clone()
                        if counts.sum() == 0:
                            # Buffer not yet populated by the load-balance hook (first step
                            # after run start / resume) -> skip this layer to avoid a spurious
                            # has_dead_expert / min_load_ratio spike on the very first viz.
                            continue
                    else:
                        _, selected = torch.topk(routing_probs, 1, dim=-1)
                        counts = F.one_hot(selected.squeeze(-1), num_classes=num_experts).float().sum(dim=0)
                    avg_load = counts.mean()
                    denom = avg_load.clamp(min=1e-9)
                    maxvio = (counts.max() - avg_load) / denom  # peak overload  (>=0, larger=worse)
                    minvio = (avg_load - counts.min()) / denom  # valley underload (=1 -> dead expert)
                    min_load_ratio = counts.min() / denom  # =0 -> dead expert
                    # entropy is rank-local (this rank's routing_probs, last micro-batch).
                    per_sample_entropy = -(routing_probs * routing_probs.clamp(min=1e-9).log()).sum(dim=-1)
                    entropy = per_sample_entropy.mean()
                    ll = f"{layer_id:02d}"
                    token_expert_counts.append((layer_id, counts))
                    moe_metrics[f"moe_maxvio/layer{ll}"] = maxvio
                    moe_metrics[f"moe_minvio/layer{ll}"] = minvio
                    moe_metrics[f"moe_minload/layer{ll}"] = min_load_ratio
                    moe_metrics[f"moe_entropy_rank0/layer{ll}"] = entropy
                    tok_maxvio.append(maxvio)
                    tok_minvio.append(minvio)
                    tok_minload.append(min_load_ratio)
                    tok_entropy.append(entropy)
                    dead = counts.min() == 0
                    any_dead = dead if any_dead is None else (any_dead | dead)
                    if hasattr(moe_block, "avg_topk_sigmoid_score"):
                        sig = moe_block.avg_topk_sigmoid_score.detach().reshape(()).to(denom)
                        moe_metrics[f"moe_topksigmoid_rank0/layer{ll}"] = sig
                        tok_sigmoid.append(sig)
                    if hasattr(moe_block, "e_score_correction_bias"):
                        bias_absmax = moe_block.e_score_correction_bias.detach().abs().max().to(denom)
                        moe_metrics[f"moe_bias/layer{ll}"] = bias_absmax
                        tok_bias.append(bias_absmax)
                # ---- moe_summary/* : cross-layer aggregates over token-MoE layers (written every step) ----
                if tok_maxvio:
                    moe_metrics["moe_summary/maxvio_avg"] = torch.stack(tok_maxvio).mean()
                    moe_metrics["moe_summary/maxvio_max"] = torch.stack(tok_maxvio).max()
                    moe_metrics["moe_summary/minvio_avg"] = torch.stack(tok_minvio).mean()
                    moe_metrics["moe_summary/minvio_max"] = torch.stack(tok_minvio).max()
                    moe_metrics["moe_summary/min_load_ratio"] = torch.stack(tok_minload).min()
                    moe_metrics["moe_summary/has_dead_expert"] = any_dead.float()
                    moe_metrics["moe_summary/entropy_avg_rank0"] = torch.stack(tok_entropy).mean()
                if tok_sigmoid:
                    moe_metrics["moe_summary/topk_sigmoid_avg_rank0"] = torch.stack(tok_sigmoid).mean()
                if tok_bias:
                    moe_metrics["moe_summary/bias_absmax"] = torch.stack(tok_bias).max()
                # ---- moe_seqwise/* : per-layer raw sequence-wise balance loss (pre-coeff) + average ----
                if seqwise_layer_losses and len(seqwise_layer_losses) == len(all_moe_indices):
                    sw_vals = []
                    for lid, sw in zip(all_moe_indices, seqwise_layer_losses):
                        v = sw.detach()
                        moe_metrics[f"moe_seqwise/layer{lid:02d}"] = v
                        sw_vals.append(v)
                    moe_metrics["moe_seqwise/avg"] = torch.stack(sw_vals).mean()
                # ---- moe_zloss/* : per-layer raw router z-loss (pre-coeff) + average/weighted loss ----
                if router_z_layer_losses and len(router_z_layer_losses) == len(all_moe_indices):
                    zl_vals = []
                    for lid, zl in zip(all_moe_indices, router_z_layer_losses):
                        v = zl.detach()
                        moe_metrics[f"moe_zloss/layer{lid:02d}"] = v
                        zl_vals.append(v)
                    moe_metrics["moe_zloss/avg_raw"] = torch.stack(zl_vals).mean()
                    moe_metrics["moe_zloss/weighted"] = router_z_loss.detach()
                if token_expert_counts:
                    moe_metrics["_token_moe_expert_counts"] = token_expert_counts
        return seq_wise_loss, router_z_loss, moe_metrics


class LingbotVlaV2Policy(PreTrainedModel):
    config_class = LingbotVLAV2Config
    name = "torch_lingbot_vla_v2"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer", "FixQwen2RMSNorm", "FixAdaRMSNorm"]

    def get_parallel_plan(self):
        from lingbotvla.distributed.parallel_plan import ParallelPlan
        from torch.distributed._tensor import Shard

        ep_plan = {
            "model.qwenvl_with_expert.qwen_expert.model.layers.*.mlp.experts.gate_proj": Shard(0),
            "model.qwenvl_with_expert.qwen_expert.model.layers.*.mlp.experts.up_proj": Shard(0),
            "model.qwenvl_with_expert.qwen_expert.model.layers.*.mlp.experts.down_proj": Shard(0),
        }
        return ParallelPlan(ep_plan=ep_plan)

    @classmethod
    def get_weight_loader(cls):
        return LingBotVLAWeightLoader()

    def __init__(self, config: LingbotVLAV2Config, eval: bool = False):
        super().__init__(config)
        self.config = config
        self.language_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.model = FlowMatchingV2(config, eval)
        if not getattr(self.config, "use_lm_head", False):
            del self.model.qwenvl_with_expert.qwenvl.lm_head
        del self.model.qwenvl_with_expert.qwen_expert.lm_head
        self.reset()
        torch.set_float32_matmul_precision("high")

    def reset(self):
        return None

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(
        self,
        images,
        img_masks,
        state,
        lang_tokens,
        lang_masks,
        actions,
        joint_mask=None,
        action_is_pad=None,
        noise=None,
        time=None,
        depth_targets=None,
        image_grid_thw=None,
        future_depth_targets=None,
        future_video_targets=None,
        future_video_cls_targets=None,
        future_video_current_patch=None,
        **kwargs,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss_dict = {}
        if getattr(self.config, "action_fp32", False):
            state = state.float()
            actions = actions.float()
        (
            losses,
            loss_depth,
            loss_future_depth,
            loss_future_video,
            depth_preds,
            seq_wise_loss,
            router_z_loss,
            moe_metrics,
            future_depth_preds,
            future_video_preds,
            current_video_preds,
        ) = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            loss_type=self.config.loss_type,
            depth_targets=depth_targets,
            image_grid_thw=image_grid_thw,
            future_depth_targets=future_depth_targets,
            future_video_targets=future_video_targets,
            future_video_cls_targets=future_video_cls_targets,
            future_video_current_patch=future_video_current_patch,
        )

        if joint_mask is not None:
            if "repeat" in self.config.loss_type:
                joint_mask = joint_mask.repeat(2, 1, 1)
            assert len(joint_mask.shape) == 3

            masked_losses = losses * joint_mask
            valid_counts = joint_mask.sum(dim=(1, 2)).clamp(min=1)
            batch_mean_losses = masked_losses.sum(dim=(1, 2)) / valid_counts
            loss_vla = masked_losses.sum() / joint_mask.sum().clamp(min=1)
        else:
            losses = losses[:, :, : self.config.action_dim]
            batch_mean_losses = losses.mean(dim=(1, 2))
            loss_vla = losses.mean()

        loss_dict["batch_mean_losses"] = batch_mean_losses.detach()
        total_loss = (
            loss_vla + loss_depth + loss_future_depth + loss_future_video + seq_wise_loss + router_z_loss
        )
        loss_dict["router_z_loss"] = (
            router_z_loss.detach() if torch.is_tensor(router_z_loss) else router_z_loss
        )
        if moe_metrics:
            loss_dict.update(moe_metrics)
        return (
            total_loss,
            loss_vla,
            loss_depth,
            loss_future_depth,
            loss_future_video,
            seq_wise_loss,
            loss_dict,
            depth_preds,
            future_depth_preds,
            future_video_preds,
            current_video_preds,
        )

    def sample_actions(self, *args, **kwargs) -> Tensor:
        return self.model.sample_actions(*args, **kwargs)


ModelClass = LingbotVlaV2Policy

__all__ = [
    "LingbotVlaV2Policy",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLTextModel",
    "Qwen3VLPreTrainedModel",
    "Qwen2ForCausalLM",
]


# ============================================================================
# LeRobot policy wrapper
# ============================================================================
# The classes above are vendored/adapted from the upstream LingBot-VLA 2.0 repo
# (Robbyant/lingbot-vla-v2). The wrapper below exposes them through LeRobot's
# ``PreTrainedPolicy`` interface (train ``forward`` + rolling ``select_action``),
# mirroring the v1 ``lingbot_vla`` policy. The LeRobot dataclass config carries
# every field ``FlowMatchingV2`` reads, so it is passed straight through.
from collections import deque  # noqa: E402

from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.policies.utils import populate_queues  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

from .configuration_lingbot_vla_v2 import LingbotVLAV2Config as LeRobotLingbotVLAV2Config  # noqa: E402


class LingbotVLAV2Policy(PreTrainedPolicy):
    """LingBot-VLA 2.0 policy for cross-embodiment robotic control.

    Couples a Qwen3-VL-4B vision-language backbone with a sparse-MoE action
    expert (pi0-style dual-stream) and predicts action chunks via flow matching.
    Native-resolution image tokens are described by ``image_grid_thw``.

    The model expects already model-ready tensors in the batch (produced by the
    lingbot_vla_v2 processor / feature transform):
        - ``images``: patchified pixels for Qwen3-VL
        - ``img_masks``: per-view validity mask
        - ``lang_tokens`` / ``lang_masks``: tokenized instruction + mask
        - ``image_grid_thw``: (num_images, 3) temporal/height/width patch grid
        - ``observation.state``: (B, max_state_dim) padded state
        - ``action``: (B, chunk_size, max_action_dim) padded action (training)
        - optional ``joint_mask``: (B, chunk_size, max_action_dim) valid-slot mask
    """

    config_class = LeRobotLingbotVLAV2Config
    name = "lingbot_vla_v2"
    _no_split_modules = ["Qwen2DecoderLayer", "FixQwen2RMSNorm", "FixAdaRMSNorm"]

    def __init__(self, config: LeRobotLingbotVLAV2Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.language_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.model = FlowMatchingV2(config, eval=False)

        if not getattr(self.config, "use_lm_head", False):
            del self.model.qwenvl_with_expert.qwenvl.lm_head
        del self.model.qwenvl_with_expert.qwen_expert.lm_head

        # The Qwen3-VL backbone builds in bfloat16 while our added projection/AdaRMSNorm
        # heads build in float32. Cast the whole model to one dtype so the dual streams
        # stay consistent (mixed dtypes raise "mat1 and mat2 must have the same dtype").
        model_dtype = getattr(torch, getattr(self.config, "dtype", "bfloat16"))
        if isinstance(model_dtype, torch.dtype) and model_dtype.is_floating_point:
            self.model.to(model_dtype)

        # Inference-time action de-normalizer: an unapply-only FeatureTransform (built
        # without the image processor / tokenizer) that inverts the per-slot normalization
        # and the canonical slot mapping on the model's actions. Built lazily-safe so a
        # missing robot_config just falls back to a plain truncation (see _postprocess_actions).
        self._action_unapply_ft = self._build_action_unapply_transform()

        self.reset()
        torch.set_float32_matmul_precision("high")

    def _build_action_unapply_transform(self):
        """Build a lightweight (processor-free) FeatureTransform for inference unapply."""
        cfg = self.config
        if not getattr(cfg, "robot_config_path", None):
            return None
        try:
            from types import SimpleNamespace

            from .feature_transform import FeatureTransform

            data_config = SimpleNamespace(
                joints=[f"{{'{k}': {v}}}" for k, v in cfg.canonical_joints.items()],
                norm_type=[f"{{'{k}': '{v}'}}" for k, v in cfg.canonical_norm_type.items()],
                cameras=list(cfg.canonical_cameras),
                img_size=cfg.resize_imgs_with_padding[0],
                chat_template="default",
                text_keys="task",
            )
            model_config = SimpleNamespace(
                max_state_dim=cfg.max_state_dim,
                max_action_dim=cfg.max_action_dim,
                chunk_size=cfg.chunk_size,
                tokenizer_max_length=cfg.tokenizer_max_length,
                use_qwen3_chat_template=True,
                return_image_grid_thw=True,
                qwen3vl_use_vision_boundaries=True,
                resize_imgs_with_padding=tuple(cfg.resize_imgs_with_padding),
            )
            return FeatureTransform(
                robot_config_path=cfg.robot_config_path,
                data_config=data_config,
                model_config=model_config,
                processor=None,
                chunk_size=cfg.chunk_size,
                norm_stats_path=cfg.norm_stats_path,
            )
        except Exception as exc:  # noqa: BLE001 - de-normalizer is best-effort at build time
            logging.get_logger(__name__).warning(
                "Could not build the inference action de-normalizer (%s); select_action will "
                "return actions in the normalized canonical space.",
                exc,
            )
            return None

    def _postprocess_actions(self, actions: Tensor, batch: dict) -> Tensor:
        """Invert normalization + the canonical slot mapping on a model action chunk.

        ``actions`` is ``(B, chunk, max_action_dim)`` in the normalized canonical space.
        Uses the per-joint masks and observation state carried in the (preprocessed) batch.
        Falls back to a plain truncation when the de-normalizer or masks are unavailable.
        """
        action_dim = self.config.output_features[ACTION].shape[0]
        ft = self._action_unapply_ft
        action_joint_mask = batch.get("action_joint_mask")
        state_joint_mask = batch.get("state_joint_mask")
        state = batch.get(OBS_STATE)
        if ft is None or action_joint_mask is None or state_joint_mask is None or state is None:
            return actions[:, :, :action_dim]

        recovered = []
        for i in range(actions.shape[0]):
            item = {
                "actions": actions[i].detach().to("cpu", torch.float32),
                "action_joint_mask": action_joint_mask[i].detach().to("cpu"),
                "state": state[i].detach().to("cpu", torch.float32),
                "state_joint_mask": state_joint_mask[i].detach().to("cpu"),
            }
            recovered.append(ft.unapply(item)[ACTION])
        return torch.stack(recovered, dim=0).to(actions.device)

    def reset(self):
        """Reset the rolling action queue used by select_action."""
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _extract_model_inputs(self, batch: dict):
        dtype = next(self.parameters()).dtype
        images = batch["images"].to(dtype=dtype)
        img_masks = batch["img_masks"]
        lang_tokens = batch["lang_tokens"]
        lang_masks = batch["lang_masks"]
        state = batch[OBS_STATE].to(dtype=dtype)
        state = F.pad(state, (0, self.config.max_state_dim - state.shape[-1]))
        image_grid_thw = batch.get("image_grid_thw")
        return images, img_masks, lang_tokens, lang_masks, state, image_grid_thw

    def forward(self, batch: dict) -> Tuple[Tensor, dict]:
        """Training forward pass returning the flow-matching loss (lerobot convention)."""
        images, img_masks, lang_tokens, lang_masks, state, image_grid_thw = self._extract_model_inputs(batch)
        actions = batch[ACTION].to(dtype=state.dtype)
        action_dim = actions.shape[-1]
        actions = F.pad(actions, (0, self.config.max_action_dim - action_dim))

        (
            losses,
            loss_depth,
            loss_future_depth,
            loss_future_video,
            _depth_preds,
            seq_wise_loss,
            router_z_loss,
            moe_metrics,
            _future_depth_preds,
            _future_video_preds,
            _current_video_preds,
        ) = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise=batch.get("noise"),
            time=batch.get("time"),
            loss_type=self.config.loss_type,
            depth_targets=batch.get("depth_targets"),
            image_grid_thw=image_grid_thw,
            future_depth_targets=batch.get("future_depth_targets"),
            future_video_targets=batch.get("future_video_targets"),
            future_video_cls_targets=batch.get("future_video_cls_targets"),
            future_video_current_patch=batch.get("future_video_current_patch"),
        )

        joint_mask = batch.get("joint_mask")
        if joint_mask is not None:
            masked_losses = losses * joint_mask
            loss_vla = masked_losses.sum() / joint_mask.sum().clamp(min=1)
        else:
            loss_vla = losses[:, :, :action_dim].mean()

        loss_dict: dict = {"l2_loss": loss_vla.item()}
        total_loss = loss_vla
        for loss_name, term in (
            ("depth_loss", loss_depth),
            ("future_depth_loss", loss_future_depth),
            ("future_video_loss", loss_future_video),
            ("seq_wise_loss", seq_wise_loss),
            ("router_z_loss", router_z_loss),
        ):
            if torch.is_tensor(term):
                loss_dict[loss_name] = term.item()
                total_loss = total_loss + term
        if moe_metrics:
            loss_dict.update({k: (v.item() if torch.is_tensor(v) else v) for k, v in moe_metrics.items()})
        loss_dict["loss"] = total_loss.item()
        return total_loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise: Tensor | None = None) -> Tensor:
        """Run flow-matching denoising and return a de-normalized action chunk (B, chunk, action_dim).

        ``sample_actions`` returns the normalized 55-D canonical action; this inverts the
        per-slot normalization and the canonical slot mapping back to the raw dataset action
        (see ``_postprocess_actions``).
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
        )
        return self._postprocess_actions(actions, batch)

    @torch.no_grad()
    def select_action(self, batch: dict, noise: Tensor | None = None) -> Tensor:
        """Select a single action for environment execution, buffering chunks in a queue."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()


__all__ = __all__ + ["LingbotVLAV2Policy"]
