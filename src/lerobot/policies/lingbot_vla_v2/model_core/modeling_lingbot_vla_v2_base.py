from typing import TypedDict

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers.utils import (
    logging,
)


class LossKwargs(TypedDict, total=False):
    labels: torch.LongTensor | None


# Qwen2.5-VL tower is unused on the Qwen3-VL v2 path.
Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLTextModel = Qwen2_5_VLPreTrainedModel = None

from transformers.models.qwen2.modeling_qwen2 import (  # noqa: E402
    Qwen2RMSNorm,
)

from .utils import (  # noqa: E402
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    sample_beta,
)

LingBotVLAWeightLoader = None  # noqa: N816  # lerobot PreTrainedPolicy handles weight loading
from .qwen2_action_expert import (  # noqa: E402
    FixQwen2RMSNorm,
    Qwen2ForCausalLM,
    Qwen2FusedExperts,
)

logger = logging.get_logger(__name__)


class AdaRMSNorm(nn.Module):
    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        """
        AdaRMSNorm: RMSNorm + FiLM
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)

        # DiT style init: gamma.weight=0, gamma.bias=1; beta.weight=0, beta.bias=0
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, hidden_states, cond):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        hidden_states = self.weight * hidden_states
        # cond = cond.to(torch.float32)
        gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        beta = self.beta(cond).unsqueeze(1)  # [B, 1, H]
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype)


class FixAdaRMSNorm(nn.Module):
    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        """
        AdaRMSNorm: RMSNorm + FiLM
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)

        # DiT style init: gamma.weight=0, gamma.bias=1; beta.weight=0, beta.bias=0
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, hidden_states, cond):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        hidden_states = self.weight * hidden_states
        cond = cond.to(torch.float32)
        gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        beta = self.beta(cond).unsqueeze(1)  # [B, 1, H]
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype)


def replace_lnorm_with_adanorm(module, hidden_size, cond_dim, final_norm_adanorm):
    for name, child in module.named_children():
        if final_norm_adanorm:
            if isinstance(child, Qwen2RMSNorm):
                if "q_layernorm" not in name and "k_layernorm" not in name:
                    setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
            elif isinstance(child, FixQwen2RMSNorm):
                if "q_layernorm" not in name and "k_layernorm" not in name:
                    setattr(module, name, FixAdaRMSNorm(hidden_size, cond_dim))
            else:
                replace_lnorm_with_adanorm(child, hidden_size, cond_dim, final_norm_adanorm)
        else:
            if isinstance(child, Qwen2RMSNorm):
                if "q_layernorm" not in name and "k_layernorm" not in name:
                    setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
            else:
                replace_lnorm_with_adanorm(child, hidden_size, cond_dim, final_norm_adanorm)


class FlowMatching(nn.Module):
    def __init__(self, config, eval):
        super().__init__()
        raise TypeError("FlowMatching is a helper base for FlowMatchingV2 and is not instantiated directly.")

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2FusedExperts):
            module.initializer_range = std
            module.reset_parameters()
        reset_post_init = getattr(module, "_reset_post_init_parameters", None)
        if reset_post_init is not None:
            reset_post_init()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    @staticmethod
    def _fp32_linear(module, x):
        """Compute linear layer in fp32 regardless of module's current parameter dtype."""
        return F.linear(
            x.float(), module.weight.float(), module.bias.float() if module.bias is not None else None
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, vlm_causal, precompute_grid_thw=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsize = images.shape[0]
        device = images.device

        # embed image
        if images.ndim == 5:
            images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        elif images.ndim == 4:
            images = einops.rearrange(images, "b n l d -> (b n) l d")
        elif images.ndim == 3:  # For inference bs=1
            bsize = 1
        img_emb = self.qwenvl_with_expert.embed_image(images, precompute_grid_thw=precompute_grid_thw)
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b (n l) d", b=bsize)  # bsize = 24
        num_img_embs = img_emb.shape[1]
        if img_masks.ndim == 1:  # For inference bs=1
            img_masks = img_masks.unsqueeze(0)
        img_masks = einops.repeat(img_masks, "b n -> b (n l)", l=num_patch)

        # embed language
        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens)
        num_lang_embs = lang_emb.shape[1]

        # assemble embeddings
        embs = torch.cat([img_emb, lang_emb], dim=1)
        pad_masks = torch.cat([img_masks, lang_masks], dim=1)

        # (see `make_att_2d_masks` to understand why zeros means bidirection)
        if not vlm_causal:
            att_masks = torch.zeros(
                (img_emb.size(0), num_img_embs + num_lang_embs), device=device, dtype=torch.bool
            )  # 1, bs_img*(768+48)
        else:
            att_masks = torch.ones(
                (img_emb.size(0), num_img_embs + num_lang_embs), device=device, dtype=torch.bool
            )  # 1, bs_img*(768+48)
        return embs, pad_masks, att_masks

    def embed_suffix(
        self, state, noisy_actions, timestep
    ):  # (torch.Size([state_bs, 32]), torch.Size([1, state_bs*50, 32]), torch.Size([1]))
        bsize = state.shape[0]  # state_bs = img_bs
        device = state.device
        dtype = state.dtype
        _fp32 = getattr(self.config, "action_fp32", False)
        # embed state
        state_emb = self._fp32_linear(self.state_proj, state) if _fp32 else self.state_proj(state)

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(  # 1, 1024
            timestep,  # torch.Size([1]))
            self.config.proj_width,  # 1024
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb_ori = time_emb

        # Fuse timestep + action information using an MLP
        action_emb = (
            self._fp32_linear(self.action_in_proj, noisy_actions)
            if _fp32
            else self.action_in_proj(noisy_actions)
        )  # torch.Size([1, state_bs*50, 1024])
        time_emb = einops.repeat(
            time_emb, "b d -> b n d", n=action_emb.shape[1]
        )  # [1, 1024] -> [1, state_bs*50, 1024]
        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)  # [1, state_bs*50, 2048]

        action_time_emb = (
            self._fp32_linear(self.action_time_mlp_in, action_time_emb)
            if _fp32
            else self.action_time_mlp_in(action_time_emb)
        )
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = (
            self._fp32_linear(self.action_time_mlp_out, action_time_emb)
            if _fp32
            else self.action_time_mlp_out(action_time_emb)
        )  # [1, state_bs*50, 1024]
        action_time_dim = action_time_emb.shape[1]

        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        pad_masks = torch.ones((bsize, action_time_dim + 1), device=device, dtype=torch.bool)

        # Set attention masks for suffix tokens so that prefix tokens cannot attend to suffix tokens.
        # And state token cannot attend action tokens.
        # Action tokens use a bidirectional attention.
        att_masks = torch.zeros((bsize, action_time_dim + 1), device=device, dtype=torch.bool)
        att_masks[:, :2] = True

        return time_emb_ori, embs, pad_masks, att_masks

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
        vlm_causal=False,
        loss_type="fm",
        precompute_grid_thw=False,
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

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vlm_causal, precompute_grid_thw=precompute_grid_thw
        )  # 1,bs_img*(768+48),2048  1,bs_img*(768+48)  1,bs_img*(768+48)
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time
        )  # [1, state_bs*(50+1), 1024], [1, state_bs*(50+1)], [1, state_bs*(50+1)]   state_bs=bs_img

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)  # 1,state_bs*(768+48+50+1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)  # 1,state_bs*(768+48+50+1)

        # pad_masks = pad_masks.reshape(state.size(0), -1)
        # att_masks = att_masks.reshape(state.size(0), -1)
        att_2d_masks = make_att_2d_masks(
            pad_masks, att_masks
        )  # torch.Size([state_bs, 768+48+50+1, 768+48+50+1])
        position_ids = torch.cumsum(pad_masks, dim=1) - 1  # torch.Size([state_bs, 768+48+50+1])
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # prefix_embs = prefix_embs.reshape(state.size(0), -1, prefix_embs.size(-1))
        # suffix_embs = suffix_embs.reshape(state.size(0), -1, suffix_embs.size(-1))
        (_, suffix_out), _, router_logits_list = self.qwenvl_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            vlm_position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],  # bs_img,(768+48),2048  [state_bs, (50+1), 1024]
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if getattr(self.config, "action_fp32", False):
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)
        # u_t = u_t.reshape(images.size(0), -1, u_t.size(-1))
        if loss_type == "fm":
            losses = F.mse_loss(u_t, v_t, reduction="none")
            # losses = torch.mean((v_t - u_t)**2, dim=-1)
        elif loss_type == "L1_fm":
            losses = F.l1_loss(u_t, v_t, reduction="none")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type!r} (expected 'fm' or 'L1_fm').")

        # Sequence-wise balance loss (DeepSeek-V3 style, for token-MoE only)
        seq_wise_loss_coeff = getattr(self.config, "sequence_wise_loss_coeff", 0)
        seq_wise_loss = 0

        if seq_wise_loss_coeff > 0 and router_logits_list:
            from .moe_loss import sequence_wise_balance_loss as triton_sequence_wise_balance_loss

            token_moe_layers_set = set(getattr(self.config, "token_moe_layers", None) or [])
            token_moe_layers_list = sorted(token_moe_layers_set)
            token_router_logits = tuple(
                logits
                for i, logits in enumerate(router_logits_list)
                if not token_moe_layers_list
                or (token_moe_layers_list[i] if i < len(token_moe_layers_list) else i) in token_moe_layers_set
            )
            token_router_biases = tuple(
                getattr(
                    self.qwenvl_with_expert.qwen_expert.model.layers[
                        token_moe_layers_list[i] if i < len(token_moe_layers_list) else i
                    ].mlp,
                    "e_score_correction_bias",
                    None,
                )
                for i, logits in enumerate(router_logits_list)
                if not token_moe_layers_list
                or (token_moe_layers_list[i] if i < len(token_moe_layers_list) else i) in token_moe_layers_set
            )

            if token_router_logits:
                token_top_k = getattr(self.config, "token_top_k", 4)

                # Batch-wise balance loss: treat all B×T tokens as one group.
                # seq_lengths=None makes the function use all tokens at once,
                # giving stable f_i statistics (B×T×K assignments / E experts).
                layer_losses = triton_sequence_wise_balance_loss(
                    router_logits_list=token_router_logits,
                    top_k=token_top_k,
                    seq_lengths=None,
                    padding_len=0,
                    e_score_correction_bias_list=token_router_biases,
                )
                if layer_losses:
                    seq_wise_loss = seq_wise_loss_coeff * torch.stack(layer_losses).mean()

        # MoE monitoring metrics for token-MoE.
        moe_metrics = {}
        if router_logits_list:
            all_moe_indices = sorted(getattr(self.config, "token_moe_layers", None) or [])
            token_expert_counts = []

            with torch.no_grad():
                for i, logits in enumerate(router_logits_list):
                    layer_id = all_moe_indices[i] if i < len(all_moe_indices) else i
                    num_experts = logits.shape[-1]
                    routing_probs = F.softmax(logits, dim=1, dtype=torch.float)

                    _, selected = torch.topk(routing_probs, 1, dim=-1)
                    expert_indices = selected.squeeze(-1)
                    counts = F.one_hot(expert_indices, num_classes=num_experts).float().sum(dim=0)

                    token_expert_counts.append((layer_id, counts))

                    # MaxVio: (max_load - avg_load) / avg_load (paper 2408.15664)
                    avg_load = counts.mean()
                    maxvio = (counts.max() - avg_load) / avg_load.clamp(min=1e-9)
                    moe_metrics[f"token_moe/layer{layer_id}_maxvio"] = maxvio

                    per_sample_entropy = -(routing_probs * routing_probs.clamp(min=1e-9).log()).sum(dim=-1)
                    moe_metrics[f"token_moe/layer{layer_id}_entropy"] = per_sample_entropy.mean()

                # Compute average MaxVio across token-MoE layers
                token_maxvio_values = [
                    moe_metrics[k]
                    for k in moe_metrics
                    if k.startswith("token_moe/") and k.endswith("_maxvio")
                ]
                if token_maxvio_values:
                    moe_metrics["token_moe/avg_maxvio"] = torch.stack(token_maxvio_values).mean()

                # Avg top-K sigmoid score (before norm) across token-MoE layers
                token_moe_layers_list = sorted(getattr(self.config, "token_moe_layers", None) or [])
                if token_moe_layers_list:
                    sigmoid_scores = []
                    for lid in token_moe_layers_list:
                        moe_block = self.qwenvl_with_expert.qwen_expert.model.layers[lid].mlp
                        if hasattr(moe_block, "avg_topk_sigmoid_score"):
                            sigmoid_scores.append(moe_block.avg_topk_sigmoid_score.detach().to(losses.device))
                    if sigmoid_scores:
                        moe_metrics["token_moe/avg_topk_sigmoid"] = torch.stack(sigmoid_scores).mean()

                if token_expert_counts:
                    moe_metrics["_token_moe_expert_counts"] = token_expert_counts

        return losses, seq_wise_loss, moe_metrics

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, vlm_causal=False, noise=None
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
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

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vlm_causal
        )
        prefix_att_2d_masks = make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks
        )  # bs, prefix_len, prefix_len
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values, _ = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        count = 0
        while time >= -dt / 2:
            count += 1
            expanded_time = time.expand(bsize)

            v_t = self.predict_velocity(state, prefix_pad_masks, past_key_values, x_t, expanded_time)

            # Euler step
            x_t += dt * v_t
            time += dt
        logger.debug("Denoised %s steps", count)
        return x_t

    def predict_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """predict velocity at time t using the suffix model."""
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
        )  # bs, suffix_len, prefix_len+suffix_len

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

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
            v_t = self.action_out_proj(suffix_out)
        return v_t


__all__ = [
    "Qwen2ForCausalLM",
]
