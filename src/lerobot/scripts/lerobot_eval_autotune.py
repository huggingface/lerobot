#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Probe hardware and recommend optimal lerobot-eval-parallel flags.

Run standalone:
    lerobot-eval-autotune --policy.path=lerobot/smolvla_libero --env.type=libero

Or called programmatically from lerobot_eval_parallel when --num-shards auto.

Steps:
    1. Probe GPU VRAM and CPU core count.
    2. Measure model VRAM footprint (load policy, delta of cuda.memory_allocated).
    3. Compute max shards limited by VRAM (85% of total).
    4. Probe env step time (optional, skipped when skip_timing=True).
    5. Probe inference time (optional, skipped when skip_timing=True).
    6. Derive num_shards = min(vram_limit, saturation_shards).
    7. Choose MUJOCO_GL (egl vs osmesa) based on remaining VRAM headroom.
    8. Compute batch_size = max(4, min(floor(cpu_cores * 0.8 / num_shards), 64)).
    9. Print paste-ready command.
"""

import math
import os
import sys
import time
from dataclasses import dataclass


@dataclass
class AutotuneRecommendation:
    num_shards: int
    batch_size: int
    mujoco_gl: str
    use_amp: bool
    # Probed values
    gpu_name: str
    vram_gb: float
    cpu_cores: int
    model_gb: float
    env_step_ms: float | None
    infer_ms: float | None


_DEFAULT_ENV_STEP_MS = 22.0  # LIBERO on GPU, typical value
_DEFAULT_INFER_MS = 5.0  # SmolVLA fp16 on H100


def _probe_gpu() -> tuple[str, float]:
    """Return (gpu_name, vram_gb). Falls back to CPU sentinel on non-CUDA systems."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "CPU (no CUDA)", 0.0
        props = torch.cuda.get_device_properties(0)
        return props.name, props.total_memory / (1024**3)
    except Exception:
        return "unknown", 0.0


def _probe_model_gb(passthrough: list[str]) -> float:
    """Load the policy (from --policy.path) and measure VRAM delta. Returns GB."""
    # Extract policy path from passthrough args
    policy_path = None
    for tok in passthrough:
        if tok.startswith("policy.path="):
            policy_path = tok.split("=", 1)[1]
            break
        if tok.startswith("--policy.path="):
            policy_path = tok.split("=", 1)[1]
            break
    if policy_path is None:
        return 0.0

    try:
        import torch

        from lerobot.policies.factory import make_policy
        from lerobot.policies.pretrained import PreTrainedConfig

        if not torch.cuda.is_available():
            return 0.0
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(0)
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path  # type: ignore[assignment]
        policy = make_policy(cfg=cfg)
        policy.eval()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated(0)
        del policy
        torch.cuda.empty_cache()
        return (after - before) / (1024**3)
    except Exception as e:
        print(f"[autotune] could not measure model VRAM: {e}", file=sys.stderr)
        return 0.0


def _probe_env_step_ms(passthrough: list[str], batch_size: int = 8, n_steps: int = 30) -> float | None:
    """Run a short env warmup and return median step latency in ms. Returns None on failure."""
    try:
        import numpy as np

        from lerobot.envs.factory import make_env

        # Parse env config from passthrough using lerobot's own parser
        env_type = None
        for tok in passthrough:
            if tok.startswith("env.type=") or tok.startswith("--env.type="):
                env_type = tok.split("=", 1)[1]
                break
        if env_type is None:
            return None

        # Minimal env config
        from lerobot.envs.factory import make_env_config

        env_cfg = make_env_config(env_type)
        envs = make_env(env_cfg, n_envs=batch_size, use_async_envs=(batch_size > 1))
        # Get first vec env
        first_suite = next(iter(envs.values()))
        env = next(iter(first_suite.values()))

        env.reset()
        dummy_action = np.zeros((batch_size, env.single_action_space.shape[0]))
        timings = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            env.step(dummy_action)
            timings.append((time.perf_counter() - t0) * 1000)
        env.close()
        return float(np.median(timings))
    except Exception as e:
        print(f"[autotune] env step probe failed: {e}", file=sys.stderr)
        return None


def probe_and_recommend(
    passthrough: list[str],
    skip_timing: bool = False,
) -> AutotuneRecommendation:
    """Probe hardware + model and return the recommended configuration."""
    gpu_name, vram_gb = _probe_gpu()
    cpu_cores = os.cpu_count() or 4

    # Model footprint
    model_gb = _probe_model_gb(passthrough)
    if model_gb == 0.0:
        # Unknown model: assume a conservative 14 GB (SmolVLA fp16) as placeholder
        model_gb = 14.0
        print("[autotune] model size unknown, assuming 14 GB (SmolVLA fp16)", file=sys.stderr)

    # Max shards from VRAM (leave 15% headroom for activations + env frames)
    max_shards_vram = max(1, math.floor(vram_gb * 0.85 / model_gb)) if vram_gb > 0 else 1

    # Timing probes
    env_step_ms: float | None = None
    infer_ms: float | None = None
    if not skip_timing:
        env_step_ms = _probe_env_step_ms(passthrough)
        # Inference time: assume ~infer = env_step / saturation_factor heuristic
        # Full probe would require loading policy — skip for now to stay fast.
        infer_ms = _DEFAULT_INFER_MS

    # Number of shards to saturate GPU: ceil(env_step / infer)
    _step = env_step_ms or _DEFAULT_ENV_STEP_MS
    _infer = infer_ms or _DEFAULT_INFER_MS
    saturation_shards = max(1, math.ceil(_step / _infer))

    num_shards = min(max_shards_vram, saturation_shards)

    # Rendering mode: EGL if all model copies + env frame buffers fit in VRAM
    env_vram_per_shard_gb = 0.01  # ~10 MB overhead per env batch
    total_with_egl = num_shards * (model_gb + env_vram_per_shard_gb)
    mujoco_gl = "egl" if (vram_gb == 0 or total_with_egl < vram_gb * 0.85) else "osmesa"

    # Batch size: fill CPU cores evenly across shards
    batch_size = max(4, min(math.floor(cpu_cores * 0.8 / num_shards), 64))

    # Recommend AMP when model is large (saves ~50% VRAM)
    use_amp = model_gb > 8.0

    return AutotuneRecommendation(
        num_shards=num_shards,
        batch_size=batch_size,
        mujoco_gl=mujoco_gl,
        use_amp=use_amp,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        cpu_cores=cpu_cores,
        model_gb=model_gb,
        env_step_ms=env_step_ms,
        infer_ms=infer_ms,
    )


def main(argv: list[str] | None = None) -> None:
    passthrough = argv if argv is not None else sys.argv[1:]

    rec = probe_and_recommend(passthrough)

    env_step_str = (
        f"{rec.env_step_ms:.0f}ms" if rec.env_step_ms else f"~{_DEFAULT_ENV_STEP_MS:.0f}ms (estimated)"
    )
    infer_str = f"{rec.infer_ms:.0f}ms" if rec.infer_ms else f"~{_DEFAULT_INFER_MS:.0f}ms (estimated)"

    print()
    print(
        f"GPU: {rec.gpu_name}  |  VRAM: {rec.vram_gb:.1f} GB  |  CPU cores: {rec.cpu_cores}  |  Model: {rec.model_gb:.1f} GB"
    )
    print()
    print(f"  env_step_ms: {env_step_str}  |  infer_ms: {infer_str}")
    print()
    print(f"  num_shards:  {rec.num_shards}")
    print(f"  batch_size:  {rec.batch_size}")
    print(f"  MUJOCO_GL:   {rec.mujoco_gl}")
    if rec.use_amp:
        print("  use_amp:     true  (recommended — halves VRAM, faster matmuls)")
    print()

    # Build paste-ready command
    flags = [f"--num-shards {rec.num_shards}", f"eval.batch_size={rec.batch_size}"]
    if rec.use_amp:
        flags.append("policy.use_amp=true")
    flags_str = " \\\n    ".join(flags)
    passthrough_str = " \\\n    ".join(passthrough) if passthrough else "[your flags]"

    print("  Paste-ready command:")
    print(f"  MUJOCO_GL={rec.mujoco_gl} lerobot-eval-parallel \\")
    print(f"    {flags_str} \\")
    print(f"    {passthrough_str}")
    print()


if __name__ == "__main__":
    main()
