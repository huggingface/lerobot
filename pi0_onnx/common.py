"""Shared building blocks for the decomposed pi0 -> ONNX export.

The full pi0 `select_action` is not exportable as a single ONNX graph (10-step
Python flow-matching loop + transformers KV-cache + mixed precision). We instead
split inference into two graphs and drive the Euler loop on the host:

  * PrefixEncoder  : images + language -> per-layer K/V cache tensors (+ masks).
                     Runs the expensive SigLIP vision tower + gemma_2b prefix ONCE.
  * DenoiseStep    : (state, x_t, timestep, K/V, masks) -> velocity v_t.
                     The small gemma_300m action expert, run 10x in the host loop.

Everything runs in float32 on CPU so the ONNX graphs avoid bf16 (poorly supported
by onnxruntime, and this machine has no CUDA). The two wrappers call the *exact*
methods used by `PI0Pytorch.sample_actions` / `denoise_step`, so the decomposition
is numerically identical to the reference (verified in run_onnx.py).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from transformers import DynamicCache

from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

SEED = 1234


def patch_sincos_float32():
    """Force the flow-matching timestep embedding to float32.

    The library computes it in float64 (via get_safe_dtype on CPU), but onnxruntime's
    CPU Cos/Sin kernels only support float32. The result is cast to the timestep dtype
    (float32) immediately afterwards anyway, so this only changes internal precision by
    ~1e-7. Applied to BOTH the PyTorch reference and the ONNX path so they stay aligned.
    """
    import math

    import lerobot.policies.pi0.modeling_pi0 as m0

    def _sincos_f32(time, dimension, min_period, max_period, device="cpu"):
        if dimension % 2 != 0:
            raise ValueError(f"dimension ({dimension}) must be divisible by 2")
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
        period = min_period * (max_period / min_period) ** fraction
        scaling = 1.0 / period * 2 * math.pi
        sin_input = scaling[None, :] * time[:, None]
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

    m0.create_sinusoidal_pos_embedding = _sincos_f32


def load_policy_fp32(model_id: str) -> PI0Policy:
    """Load a pi0 checkpoint forced to float32 on CPU (for ONNX-friendly export)."""
    cfg = PI0Config.from_pretrained(model_id)
    cfg.dtype = "float32"
    cfg.device = "cpu"
    policy = PI0Policy.from_pretrained(model_id, config=cfg)
    policy.eval().to("cpu", dtype=torch.float32)
    return policy


def _remap_ckpt_key(k: str) -> str:
    """pi0 checkpoint key -> model parameter name (mirrors from_pretrained/_fix for pi0)."""
    if k.startswith("time_mlp_in."):
        k = "action_" + k
    elif k.startswith("time_mlp_out."):
        k = "action_" + k
    base = k[len("model.") :] if k.startswith("model.") else k
    return f"model.{base}"


def load_policy_fp32_lowmem(model_id: str) -> PI0Policy:
    """Low-memory float32/CPU load for RAM-constrained machines (e.g. 18GB Macs).

    The normal load transiently holds a freshly-initialized fp32 model (~13GB) AND the
    fully-loaded state dict (~9GB) at once (~22GB peak) -> OOM on 18GB. Here we build the
    model normally on CPU (buffers stay real) and copy each weight in from the mmap'd
    safetensors one tensor at a time, so peak stays near a single fp32 model (~13GB).
    """
    from safetensors import safe_open
    from transformers.utils import cached_file

    import sys

    cfg = PI0Config.from_pretrained(model_id)
    cfg.dtype = "float32"
    cfg.device = "cpu"
    print("  [lowmem] building fp32 model on cpu...", flush=True)
    policy = PI0Policy(cfg).eval()
    print("  [lowmem] model built; copying weights from safetensors...", flush=True)
    sys.stdout.flush()

    param_names = {n for n, _ in policy.named_parameters()}
    buffer_names = {n for n, _ in policy.named_buffers()}
    filled = set()

    def _copy(name: str, tensor: torch.Tensor):
        if name in param_names:
            policy.get_parameter(name).data.copy_(tensor)
        elif name in buffer_names:
            policy.get_buffer(name).data.copy_(tensor)
        else:
            return
        filled.add(name)

    st_path = cached_file(model_id, "model.safetensors")
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118
            t = f.get_tensor(k).to(torch.float32)
            # lm_head weight also initializes the tied language-model embedding.
            if k.endswith("paligemma_with_expert.paligemma.lm_head.weight"):
                _copy(
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
                    t,
                )
            _copy(_remap_ckpt_key(k), t)
            del t

    print("  [lowmem] weights copied.", flush=True)
    missing = (param_names | buffer_names) - filled
    # Non-persistent buffers (rotary inv_freq, siglip position_ids) are (re)computed in
    # __init__ and legitimately absent from the checkpoint; only real params must be filled.
    missing_params = [n for n in missing if n in param_names]
    if missing_params:
        raise RuntimeError(f"{len(missing_params)} params not filled from checkpoint: {missing_params[:8]}")
    return policy


def camera_keys(policy: PI0Policy) -> list[str]:
    return list(policy.config.image_features.keys())


def make_fixed_inputs(policy: PI0Policy, batch_size: int = 1):
    """Deterministic, self-contained network inputs (post-preprocessing values).

    We feed already-preprocessed images (float32 in [-1, 1], [B,3,224,224]) and a
    normalized state, plus a fixed language-token sequence. Normalization/tokenization
    are identical affine/lookup steps applied on the host, so skipping them here does
    not affect the ONNX-vs-PyTorch comparison of the network itself.
    """
    g = torch.Generator().manual_seed(SEED)
    cfg = policy.config
    H, W = cfg.image_resolution
    cams = camera_keys(policy)

    images = [torch.rand(batch_size, 3, H, W, generator=g) * 2.0 - 1.0 for _ in cams]
    img_masks = [torch.ones(batch_size, dtype=torch.bool) for _ in cams]
    lang_tokens = torch.arange(cfg.tokenizer_max_length).remainder(50).add(10)
    lang_tokens = lang_tokens[None].expand(batch_size, -1).contiguous().to(torch.long)
    lang_masks = torch.ones(batch_size, cfg.tokenizer_max_length, dtype=torch.bool)
    state = torch.randn(batch_size, cfg.max_state_dim, generator=g)
    noise = torch.randn(batch_size, cfg.chunk_size, cfg.max_action_dim, generator=g)

    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
        "noise": noise,
    }


class PrefixEncoder(nn.Module):
    """images (+ language) -> stacked K/V cache tensors + prefix pad mask.

    Output shapes (L = #prefix layers, P = prefix length):
      keys, values : [L, B, num_kv_heads, P, head_dim]
      prefix_pad_masks : [B, P] (bool)
    """

    def __init__(self, policy: PI0Policy):
        super().__init__()
        self.model = policy.model

    @torch.no_grad()
    def forward(self, *args: Tensor):
        n_cam = (len(args) - 2) // 2
        images = list(args[:n_cam])
        img_masks = list(args[n_cam : 2 * n_cam])
        lang_tokens, lang_masks = args[2 * n_cam], args[2 * n_cam + 1]

        m = self.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        # Cast bool -> int64: make_att_2d_masks runs cumsum on att_masks, and onnxruntime's
        # CumSum rejects bool inputs (same for the position-id cumsum below).
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks.to(torch.int64))
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(torch.int64), dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
        m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

        _, past_key_values = m.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        keys = torch.stack([item[0] for item in past_key_values], dim=0)
        values = torch.stack([item[1] for item in past_key_values], dim=0)
        return keys, values, prefix_pad_masks


class DenoiseStep(nn.Module):
    """(state, x_t, timestep, keys, values, prefix_pad_masks) -> velocity v_t."""

    def __init__(self, policy: PI0Policy):
        super().__init__()
        self.model = policy.model

    @torch.no_grad()
    def forward(
        self,
        state: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        keys: Tensor,
        values: Tensor,
        prefix_pad_masks: Tensor,
    ):
        m = self.model
        n_layers = keys.shape[0]
        past_key_values = DynamicCache(
            tuple((keys[i], values[i], None) for i in range(n_layers))
        )

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = m.embed_suffix(
            state, x_t, timestep
        )
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Cast bool -> int64 before reduce-sum / cumsum (onnxruntime rejects bool there).
        prefix_offsets = torch.sum(prefix_pad_masks.to(torch.int64), dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(torch.int64), dim=1) - 1
        full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
        m.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = m.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -m.config.chunk_size :].to(torch.float32)
        v_t = m.action_out_proj(suffix_out)
        return v_t


PREFIX_INPUT_NAMES_TMPL = "image_{i}"
PREFIX_MASK_NAMES_TMPL = "img_mask_{i}"


def prefix_arg_names(n_cam: int) -> list[str]:
    names = [PREFIX_INPUT_NAMES_TMPL.format(i=i) for i in range(n_cam)]
    names += [PREFIX_MASK_NAMES_TMPL.format(i=i) for i in range(n_cam)]
    names += ["lang_tokens", "lang_masks"]
    return names


def export_to_onnx(module, args, input_names, output_names, path, external_data=False):
    """Export a wrapper module to ONNX, preferring the classic tracer.

    The classic (TorchScript) tracer bakes the transformers KV-cache / mask control
    flow into constants for the fixed input shapes, which the dynamo/torch.export path
    tends to reject. We fall back to dynamo only if the classic tracer fails.
    """
    import os

    import torch as _torch

    force = os.environ.get("PI0_ONNX_EXPORTER", "").lower()  # "", "classic", or "dynamo"

    def _dynamo():
        kw = dict(input_names=input_names, output_names=output_names, dynamo=True)
        if external_data:
            kw["external_data"] = True
        try:
            _torch.onnx.export(module, args, path, **kw)
        except TypeError:
            kw.pop("external_data", None)
            _torch.onnx.export(module, args, path, **kw)
        return "dynamo"

    def _classic():
        _torch.onnx.export(
            module,
            args,
            path,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
        return "classic"

    if force == "dynamo":
        return _dynamo()
    if force == "classic":
        return _classic()

    # Default: classic tracer (bakes cache/mask control flow) first, dynamo as fallback.
    # The prefix graph (>2GB) requires dynamo + external_data, so callers set external_data
    # and/or PI0_ONNX_EXPORTER=dynamo for it.
    errors = {}
    try:
        return _classic()
    except Exception as e:  # noqa: BLE001
        errors["classic"] = repr(e)
    try:
        return _dynamo()
    except Exception as e:  # noqa: BLE001
        errors["dynamo"] = repr(e)
        raise RuntimeError(f"ONNX export failed. classic={errors.get('classic')} dynamo={errors.get('dynamo')}")


def make_ort_denoise_callable(session):
    """Wrap an onnxruntime InferenceSession for the DenoiseStep graph."""
    in_names = [i.name for i in session.get_inputs()]

    def _call(state, x_t, timestep, keys, values, prefix_pad_masks):
        feeds = {
            in_names[0]: state.astype(np.float32),
            in_names[1]: x_t.astype(np.float32),
            in_names[2]: timestep.astype(np.float32),
            in_names[3]: keys.astype(np.float32),
            in_names[4]: values.astype(np.float32),
            in_names[5]: prefix_pad_masks,
        }
        return session.run(None, feeds)[0]

    return _call


def euler_loop_numpy(prefix_out, denoise_callable, noise: np.ndarray, num_steps: int) -> np.ndarray:
    """Host-side Euler integration mirroring flow_matching.euler_integrate.

    ``denoise_callable(state, x_t, timestep, keys, values, prefix_pad_masks) -> v_t``
    is either the PyTorch DenoiseStep or an onnxruntime session wrapper.
    """
    keys, values, prefix_pad_masks, state = prefix_out
    bsize = noise.shape[0]
    dt = -1.0 / num_steps
    x_t = noise.astype(np.float32)
    for step in range(num_steps):
        t = np.float32(1.0 + step * dt)
        timestep = np.full((bsize,), t, dtype=np.float32)
        v_t = denoise_callable(state, x_t, timestep, keys, values, prefix_pad_masks)
        x_t = x_t + dt * v_t
    return x_t


__all__ = [
    "SEED",
    "patch_sincos_float32",
    "load_policy_fp32",
    "load_policy_fp32_lowmem",
    "camera_keys",
    "make_fixed_inputs",
    "PrefixEncoder",
    "DenoiseStep",
    "prefix_arg_names",
    "export_to_onnx",
    "make_ort_denoise_callable",
    "euler_loop_numpy",
    "ACTION",
    "OBS_STATE",
    "OBS_LANGUAGE_TOKENS",
    "OBS_LANGUAGE_ATTENTION_MASK",
]
