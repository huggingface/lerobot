#!/usr/bin/env python

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

"""Benchmark ``PI052Policy.forward + backward`` on a single GPU.

Compares the new SDPA attention path against the eager baseline by
monkeypatching ``sdpa_attention_forward`` before the first model
forward — so both runs share identical Q/K/V plumbing and only the
attention kernel differs. Reports steps/sec and peak GPU memory.

SLURM-only:

    sbatch examples/benchmark/bench_pi052_step.slurm

Or one-off:

    srun --partition=hopper-prod --qos=high --gpus=1 --time=15 \\
        python examples/benchmark/bench_pi052_step.py --attn sdpa --batch-size 8
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time

import torch


def _maybe_patch_eager() -> None:
    """Swap ``sdpa_attention_forward`` for the original eager forward.

    Must be called BEFORE PI052Policy is instantiated — the layer
    compute functions resolve the symbol at call time (module-level
    lookup), so this patch covers both pi05 and pi052 KI paths."""
    from transformers.models.gemma import modeling_gemma

    from lerobot.policies.pi05 import modeling_pi05

    modeling_pi05.sdpa_attention_forward = modeling_gemma.eager_attention_forward


_LIGER_SUBKERNELS = ("rope", "rms_norm", "geglu", "layer_norm")


def _maybe_patch_liger(spec: str) -> dict:
    """Globally patch PaliGemma/Gemma/Siglip modules with Liger Triton kernels.

    Must be called BEFORE PI052Policy is instantiated — Liger replaces
    classes inside ``transformers.models.{gemma,gemma2,siglip,paligemma}``,
    so any model built after the call picks up the fused forwards.

    ``spec`` is a comma-separated subset of {rope, rms_norm, geglu,
    layer_norm} (also ``all`` and ``none``). ``cross_entropy`` and
    ``fused_linear_cross_entropy`` are intentionally skipped — pi052's
    losses use ``F.cross_entropy`` directly (not ``nn.CrossEntropyLoss``)
    and never traverse ``PaliGemmaForConditionalGeneration.forward``,
    so neither patch would fire without invasive model-code changes.
    """
    enabled = dict.fromkeys(_LIGER_SUBKERNELS, False)
    if spec in ("", "none"):
        return enabled
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    if tokens == ["all"]:
        enabled = dict.fromkeys(_LIGER_SUBKERNELS, True)
    else:
        for t in tokens:
            if t not in enabled:
                raise SystemExit(f"Unknown liger subkernel: {t!r}. Choose from {_LIGER_SUBKERNELS} or 'all'.")
            enabled[t] = True

    from liger_kernel.transformers import apply_liger_kernel_to_paligemma

    apply_liger_kernel_to_paligemma(
        rope=enabled["rope"],
        rms_norm=enabled["rms_norm"],
        geglu=enabled["geglu"],
        layer_norm=enabled["layer_norm"],
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )
    return enabled


def _maybe_patch_flex() -> None:
    """Swap ``sdpa_attention_forward`` for a FlexAttention-backed forward.

    Experimental: builds a per-call ``score_mod`` from the additive
    mask and dispatches to a compiled ``flex_attention`` kernel.

    Known issue on torch 2.7.1: dynamo errors out with
    ``FlexAttentionHigherOrderVariable() has no type`` when the
    ``score_mod`` closure captures a per-call bias tensor. A proper
    port needs ``create_block_mask(mask_mod, ...)`` plumbed at the
    PI05Pytorch.forward level so a BlockMask object can be passed
    down to the layer compute, not a per-call closure. Left as
    future work; keep this stub for benchmark experimentation."""
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    from lerobot.policies.pi05 import modeling_pi05

    compiled_flex = torch.compile(flex_attention, dynamic=True)

    def flex_forward(module, query, key, value, attention_mask, scaling, dropout=0.0):
        n_rep = module.num_key_value_groups
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)

        bias = attention_mask  # (B, 1, Lq, Lk) additive

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[b, 0, q_idx, kv_idx]

        attn_output = compiled_flex(query, key, value, score_mod=score_mod, scale=scaling)
        return attn_output.transpose(1, 2).contiguous(), None

    modeling_pi05.sdpa_attention_forward = flex_forward


def _build_policy(args, device: torch.device):
    """Random-init PI052Policy at production-relevant shapes."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi052.configuration_pi052 import PI052Config
    from lerobot.policies.pi052.modeling_pi052 import PI052Policy

    # Production has ``unfreeze_lm_head=True`` + ``text_loss_weight>0``,
    # which flips ``train_expert_only=False`` in __post_init__ and
    # makes the whole PaliGemma + Gemma-expert stack trainable. We
    # mirror that here so the optimizer-state count reflects reality;
    # the loss path still goes through ``PI05Policy.forward`` because
    # ``text_labels`` / FAST tokens are absent from the synthetic batch
    # (see ``PI052Policy.forward`` early-return).
    config = PI052Config(
        max_action_dim=args.action_dim,
        max_state_dim=args.state_dim,
        dtype=args.dtype,
        knowledge_insulation=args.knowledge_insulation,
        text_loss_weight=1e-3 if args.train_full else 0.0,
        flow_loss_weight=1.0,
        enable_fast_action_loss=False,
        unfreeze_lm_head=args.train_full,
        tokenizer_max_length=args.lang_tokens,
        device="cuda",
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,)),
    }
    policy = PI052Policy(config)
    policy.to(device)
    if args.gradient_checkpointing:
        policy.model.gradient_checkpointing_enable()
    policy.train()
    return policy, config


def _build_batch(args, config, device: torch.device) -> dict:
    """Synthetic batch matching the training-loop input contract."""
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    B = args.batch_size
    L = args.lang_tokens
    return {
        OBS_LANGUAGE_TOKENS: torch.randint(0, 250000, (B, L), device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(B, L, dtype=torch.bool, device=device),
        "observation.images.base_0_rgb": torch.rand(B, 3, 224, 224, device=device),
        "observation.images.base_0_rgb_padding_mask": torch.ones(B, dtype=torch.bool, device=device),
        "observation.state": torch.randn(B, args.state_dim, device=device),
        ACTION: torch.randn(B, config.chunk_size, args.action_dim, device=device),
        "action_is_pad": torch.zeros(B, config.chunk_size, dtype=torch.bool, device=device),
        "task": ["bench task"] * B,
    }


def _step(policy, batch, optimizer=None) -> torch.Tensor:
    loss, _ = policy.forward(batch)
    loss.backward()
    if optimizer is not None:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    else:
        for p in policy.parameters():
            if p.grad is not None:
                p.grad = None
    return loss.detach()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn", choices=["sdpa", "eager", "flex"], default="sdpa")
    parser.add_argument(
        "--kernels",
        default="none",
        help=(
            "Liger sub-kernels to enable, comma-separated. Choose from "
            f"{_LIGER_SUBKERNELS} or use 'all' / 'none' (default). Applied "
            "via apply_liger_kernel_to_paligemma() BEFORE model build."
        ),
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="Set policy.config.compile_model=True (torch.compile the forward).",
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        help="torch.compile mode (default | reduce-overhead | max-autotune).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lang-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--knowledge-insulation", action="store_true", default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--optimizer",
        choices=["none", "adamw", "adamw_fused"],
        default="adamw_fused",
        help=(
            "Whether to include an AdamW step in the timed iteration. "
            "'none' mirrors the fwd+bwd-only original bench; 'adamw' / "
            "'adamw_fused' add the realistic ~2x param-bytes optimizer "
            "state and ``optimizer.step()`` cost."
        ),
    )
    parser.add_argument(
        "--train-full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mirror production: unfreeze the PaliGemma backbone (full "
            "~3B trainable params) instead of training only the 300M "
            "action expert."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Benchmark requires CUDA; submit via slurm (srun/sbatch).")

    if args.attn == "eager":
        _maybe_patch_eager()
    elif args.attn == "flex":
        _maybe_patch_flex()

    liger_flags = _maybe_patch_liger(args.kernels)

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    policy, config = _build_policy(args, device)
    batch = _build_batch(args, config, device)

    optimizer = None
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if args.optimizer != "none":
        trainable = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=5e-5, fused=(args.optimizer == "adamw_fused")
        )

    for _ in range(args.warmup):
        _step(policy, batch, optimizer)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(args.steps):
        _step(policy, batch, optimizer)
    ender.record()
    torch.cuda.synchronize()
    total_ms = starter.elapsed_time(ender)
    step_ms = total_ms / args.steps
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    optim_gb = 0.0
    if optimizer is not None:
        for st in optimizer.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    optim_gb += v.numel() * v.element_size() / (1024**3)

    liger_on = ",".join(k for k, v in liger_flags.items() if v) or "none"
    name = (
        f"{args.attn:>5} | BS={args.batch_size} | L={args.lang_tokens} | "
        f"KI={args.knowledge_insulation} | GC={args.gradient_checkpointing} | "
        f"compile={args.compile_model} | liger={liger_on} | opt={args.optimizer} | dtype={args.dtype}"
    )
    print(
        f"{name}\n  step_ms={step_ms:.1f}  steps/sec={1000.0 / step_ms:.3f}  "
        f"peak_mem={peak_gb:.2f} GiB  optim_state={optim_gb:.2f} GiB  "
        f"trainable_params={trainable_params / 1e9:.2f}B"
    )

    del policy, batch
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
