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

"""Shared NVIDIA Transformer Engine (TE) FP8 machinery for the pi0 / pi05 VLM MLP.

This module is imported by both ``policies/pi0/modeling_pi0.py`` and
``policies/pi05/modeling_pi05.py``. It implements a single FP8 backend,
``te_layernorm_mlp``, that fuses each VLM language-model layer's
``post_attention_layernorm`` (a plain RMSNorm, ``cond_dim=None`` for both pi0
and pi05 VLMs) together with the gate/up/down projections into one
``te.LayerNormMLP`` kernel. Fusing the norm into the MLP removes the HBM
round-trip between the RMSNorm and FC1 and lets TE run the whole
RMSNorm + FC1(geglu) + FC2 block in FP8.

The action expert (which uses adaRMS on pi05) is never touched and stays bf16.

Transformer Engine is imported lazily (only when FP8 is actually enabled) so
that importing pi0 / pi05 on a machine without transformer_engine keeps working.
"""

import contextlib
import logging

import torch
from torch import nn

# Transformer Engine is imported lazily (only when FP8 is actually enabled) to avoid
# CUDA-level side effects (cuBLAS algorithm changes, custom extension registration)
# that can alter numerical behavior of standard nn.Linear ops during bf16 training,
# and to keep pi0 / pi05 importable on machines without transformer_engine installed.
te = None
DelayedScaling = None
Float8BlockScaling = None
Format = None
te_checkpoint = None
_transformer_engine_available = False


def _ensure_transformer_engine() -> bool:
    """Lazily import Transformer Engine. Call only when FP8 is actually needed."""
    global te, DelayedScaling, Float8BlockScaling, Format, te_checkpoint
    global _transformer_engine_available
    if _transformer_engine_available:
        return True
    try:
        import transformer_engine.pytorch as _te
        from transformer_engine.common.recipe import (
            DelayedScaling as _DS,
            Float8BlockScaling as _F8B,
            Format as _Fmt,
        )
        from transformer_engine.pytorch.distributed import checkpoint as _te_ckpt

        te = _te
        DelayedScaling = _DS
        Float8BlockScaling = _F8B
        Format = _Fmt
        te_checkpoint = _te_ckpt
        _transformer_engine_available = True
        return True
    except ImportError:
        return False


def build_vlm_mlp_fp8_recipe(config):
    """Build a Transformer Engine FP8 recipe for the VLM MLP layers.

    Dispatch on ``config.vlm_mlp_fp8_recipe_kind``:
      - "delayed_scaling"      → per-tensor DelayedScaling (default, 16-step amax history).
      - "float8_block_scaling" → block-wise Float8BlockScaling (1D activations/grads, 2D weights).
    HYBRID format (E4M3 fwd, E5M2 bwd) is used for both unless ``vlm_mlp_fp8_format='e4m3'``.

    Returns ``None`` when FP8 is disabled (so the bf16 path is untouched).
    """
    if config is None or not getattr(config, "vlm_mlp_fp8_enable", False):
        return None

    if not _ensure_transformer_engine():
        raise RuntimeError(
            "vlm_mlp_fp8_enable=True but transformer_engine is not available in this environment."
        )

    fp8_format = Format.HYBRID if config.vlm_mlp_fp8_format == "hybrid" else Format.E4M3

    kind = config.vlm_mlp_fp8_recipe_kind
    if kind == "delayed_scaling":
        return DelayedScaling(
            margin=config.vlm_mlp_fp8_margin,
            fp8_format=fp8_format,
            amax_history_len=config.vlm_mlp_fp8_amax_history_len,
            amax_compute_algo=config.vlm_mlp_fp8_amax_compute_algo,
        )
    if kind == "float8_block_scaling":
        # TE defaults: x_block_scaling_dim=1, w_block_scaling_dim=2, grad_block_scaling_dim=1,
        # use_f32_scales=False (power-of-2). All four are exposed through the policy config so
        # the script can flip them with --policy.vlm_mlp_fp8_blockscale_*.
        return Float8BlockScaling(
            fp8_format=fp8_format,
            use_f32_scales=config.vlm_mlp_fp8_blockscale_use_f32_scales,
            x_block_scaling_dim=config.vlm_mlp_fp8_blockscale_x_dim,
            w_block_scaling_dim=config.vlm_mlp_fp8_blockscale_w_dim,
            grad_block_scaling_dim=config.vlm_mlp_fp8_blockscale_grad_dim,
        )
    raise ValueError(f"Unknown vlm_mlp_fp8_recipe_kind: {kind}")


def validate_fp8_linear_features(in_features: int, out_features: int, where: str, strict: bool = True) -> bool:
    """Validate TE FP8 linear shape constraints (both dims divisible by 16)."""
    is_valid = (in_features % 16 == 0) and (out_features % 16 == 0)
    if not is_valid and strict:
        raise ValueError(
            f"FP8 requires linear dims divisible by 16 at {where}, got in={in_features}, out={out_features}."
        )
    return is_valid


class TELayerNormGemmaMLP(nn.Module):
    """Gemma MLP replacement using te.LayerNormMLP for deeper FP8 kernel fusion.

    Absorbs the external ``post_attention_layernorm`` INTO this module, so the
    fused ``te.LayerNormMLP`` kernel handles RMSNorm + FC1 (gate+up) + activation
    + FC2 in one shot. This eliminates the HBM round-trip between norm and FC1.

    This is valid ONLY for the VLM branch (i=0) where:
    - adarms_cond is always None (standard RMSNorm, no adaptive conditioning)
    - gate returned by the norm is always None (_gated_residual becomes a simple add)

    The external ``post_attention_layernorm`` call must be SKIPPED for VLM layers
    using this module — see ``compute_layer_complete`` for the branching logic.

    Activation: TE's 'geglu' = gated GELU with tanh approximation, matching Gemma's
    gelu_pytorch_tanh (NOT 'qgeglu' which is the "quick GELU" x * sigmoid(1.702*x)).
    """

    # Marker so compute_layer_complete / the decoder layer can detect this module type.
    absorbs_post_attention_layernorm = True

    def __init__(self, hidden_size, intermediate_size, eps, device, dtype):
        super().__init__()
        self.layernorm_mlp = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=intermediate_size,
            eps=eps,
            bias=False,
            normalization="RMSNorm",
            activation="geglu",
            zero_centered_gamma=True,
            params_dtype=dtype,
            device=device,
        )

    def forward(self, x):
        return self.layernorm_mlp(x)


def _load_weights_into_te_layernorm_mlp(te_module, gate_proj, up_proj, down_proj, layernorm_weight):
    """Copy weights from the original GemmaMLP projections and layernorm into te.LayerNormMLP.

    fc1_weight shape: [2*intermediate_size, hidden_size] (gate + up concatenated)
    fc2_weight shape: [hidden_size, intermediate_size]   (down_proj)
    layer_norm_weight: copied from post_attention_layernorm.weight (absorbed into fused module)
    """
    with torch.no_grad():
        te_module.layernorm_mlp.fc1_weight.copy_(torch.cat([gate_proj.weight, up_proj.weight], dim=0))
        te_module.layernorm_mlp.fc2_weight.copy_(down_proj.weight)
        # Absorb the external norm weight. TE uses zero_centered_gamma=True convention:
        # output = (x / RMS(x)) * (1 + gamma), matching Gemma's (1 + weight) convention.
        te_module.layernorm_mlp.layer_norm_weight.copy_(layernorm_weight)


def get_mlp_weight_dtype(mlp):
    """Get the weight dtype of an MLP module, supporting stock GemmaMLP and TELayerNormGemmaMLP."""
    if hasattr(mlp, "up_proj") and hasattr(mlp.up_proj, "weight"):
        return mlp.up_proj.weight.dtype
    if hasattr(mlp, "layernorm_mlp") and hasattr(mlp.layernorm_mlp, "fc1_weight"):
        return mlp.layernorm_mlp.fc1_weight.dtype
    return torch.bfloat16  # safe default


def configure_vlm_mlp_fp8(vlm_layers, config):
    """Swap each VLM language-model layer's ``mlp`` for a fused ``TELayerNormGemmaMLP``.

    Iterates the VLM decoder layers and, for each, reads hidden/intermediate/device/dtype
    from the existing ``layer.mlp.gate_proj`` and eps from ``layer.post_attention_layernorm``,
    builds a fused te.LayerNormMLP, loads the gate/up/down + norm weights into it, and
    replaces ``layer.mlp`` in place. No-op when ``config.vlm_mlp_fp8_enable`` is False, so
    the bf16 default path is byte-for-byte unchanged.
    """
    if config is None or not getattr(config, "vlm_mlp_fp8_enable", False):
        return

    if not _ensure_transformer_engine():
        raise RuntimeError("vlm_mlp_fp8_enable=True but transformer_engine is unavailable")

    strict = getattr(config, "vlm_mlp_fp8_strict_shape_check", True)
    converted = 0
    for layer_idx, layer in enumerate(vlm_layers):
        mlp = layer.mlp
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj
        down_proj = mlp.down_proj

        # Validate TE FP8 shape constraints at swap time (once per layer).
        if strict:
            for proj_name, proj in (
                ("gate_proj", gate_proj),
                ("up_proj", up_proj),
                ("down_proj", down_proj),
            ):
                validate_fp8_linear_features(
                    proj.in_features,
                    proj.out_features,
                    f"vlm.layer[{layer_idx}].mlp.{proj_name}",
                    strict=True,
                )

        hidden_size = gate_proj.in_features
        intermediate_size = gate_proj.out_features
        device = gate_proj.weight.device
        dtype = gate_proj.weight.dtype

        # eps from the (plain RMSNorm) post_attention_layernorm being folded into the MLP.
        eps = layer.post_attention_layernorm.eps
        te_mlp = TELayerNormGemmaMLP(
            hidden_size,
            intermediate_size,
            eps,
            device=device,
            dtype=dtype,
        )
        _load_weights_into_te_layernorm_mlp(
            te_mlp,
            gate_proj,
            up_proj,
            down_proj,
            layer.post_attention_layernorm.weight,
        )
        layer.mlp = te_mlp
        converted += 1

    if getattr(config, "vlm_mlp_fp8_log_once", True):
        logging.info(
            "Configured VLM MLP FP8 backend=te_layernorm_mlp (converted=%d layers)",
            converted,
        )


def vlm_mlp_fp8_autocast(config, recipe, training):
    """Return the TE FP8 autocast context for the VLM layer loop, else nullcontext.

    FP8 autocast is applied when FP8 is enabled AND (we are training OR
    ``config.vlm_mlp_quant_inference`` is True). During eval with
    ``vlm_mlp_quant_inference=False`` the forward runs in bf16 from the stored
    master weights. Returns ``nullcontext()`` whenever FP8 is disabled.
    """
    if (
        config is not None
        and getattr(config, "vlm_mlp_fp8_enable", False)
        and (training or getattr(config, "vlm_mlp_quant_inference", True))
    ):
        # TE must already be importable here (recipe was built at model construction).
        _ensure_transformer_engine()
        return te.autocast(enabled=True, recipe=recipe)
    return contextlib.nullcontext()


def remap_fp8_state_dict_keys(state_dict, num_vlm_layers, vlm_layer_prefix):
    """Remap a stock bf16 VLM MLP checkpoint into the fused te.LayerNormMLP layout.

    For each VLM language-model layer:
      mlp.gate_proj.weight + mlp.up_proj.weight → mlp.layernorm_mlp.fc1_weight (gate ++ up)
      mlp.down_proj.weight                      → mlp.layernorm_mlp.fc2_weight
      post_attention_layernorm.weight           → mlp.layernorm_mlp.layer_norm_weight

    ``vlm_layer_prefix`` is the key prefix up to (but excluding) the layer index, e.g.
    ``"paligemma_with_expert.paligemma.model.language_model.layers"``. An optional leading
    ``"model."`` prefix (added by lerobot's PIxPolicy wrapper) is tolerated. Both backends
    concatenate gate then up, matching TE's geglu convention (first half = gate).

    If the checkpoint is already in TE format (fc1_weight present) the remap is a no-op:
    re-applying it would rename the dead post_attention_layernorm.weight onto the trained
    layer_norm_weight (collision) and discard the warm start. Returns a new dict.
    """
    import re

    # Already-TE checkpoint (saved by a model that already used te.LayerNormMLP): skip remap.
    already_te_format = any("mlp.layernorm_mlp.fc1_weight" in k for k in state_dict)
    if already_te_format:
        return dict(state_dict)

    pat = re.compile(r"^(model\.)?" + re.escape(vlm_layer_prefix) + r"\.(\d+)\.")

    # First pass: collect gate/up weights per VLM layer so they can be concatenated
    # into the fused fc1_weight in the second pass.
    vlm_gate: dict[int, torch.Tensor] = {}
    vlm_up: dict[int, torch.Tensor] = {}
    for key, value in state_dict.items():
        m = pat.match(key)
        if m:
            rest = key[m.end():]
            layer_idx = int(m.group(2))
            if rest == "mlp.gate_proj.weight":
                vlm_gate[layer_idx] = value
            elif rest == "mlp.up_proj.weight":
                vlm_up[layer_idx] = value

    if num_vlm_layers is not None and len(vlm_gate) not in (0, num_vlm_layers):
        logging.warning(
            "FP8 state_dict remap: found %d VLM gate_proj weights, expected %d layers.",
            len(vlm_gate),
            num_vlm_layers,
        )

    new_state_dict = {}
    for key, value in state_dict.items():
        m = pat.match(key)
        if m:
            model_prefix = m.group(1) or ""
            layer_idx = int(m.group(2))
            rest = key[m.end():]
            layer_base = f"{model_prefix}{vlm_layer_prefix}.{layer_idx}."

            if rest == "mlp.gate_proj.weight":
                gate_w = vlm_gate.get(layer_idx)
                up_w = vlm_up.get(layer_idx)
                if gate_w is not None and up_w is not None:
                    new_state_dict[f"{layer_base}mlp.layernorm_mlp.fc1_weight"] = torch.cat(
                        [gate_w, up_w], dim=0
                    )
                continue  # skip the individual gate_proj key
            elif rest == "mlp.up_proj.weight":
                continue  # already handled above with gate_proj
            elif rest == "mlp.down_proj.weight":
                new_state_dict[f"{layer_base}mlp.layernorm_mlp.fc2_weight"] = value
                continue
            elif rest == "post_attention_layernorm.weight":
                # Gemma uses (1+weight)*normalized_x with weight~0; TE zero_centered_gamma
                # uses (1+gamma)*normalized_x. Copy weight directly (same effective scale).
                new_state_dict[f"{layer_base}mlp.layernorm_mlp.layer_norm_weight"] = value
                # KEEP the original key too: configure_vlm_mlp_fp8 only swaps layer.mlp, so the
                # external post_attention_layernorm module still exists on the VLM layer (now dead —
                # compute_layer_complete skips it via the absorbs_post_attention_layernorm marker).
                # Its param must still load or load_state_dict(strict=True) raises a missing-key error.
                new_state_dict[key] = value
                continue

        new_state_dict[key] = value

    return new_state_dict
