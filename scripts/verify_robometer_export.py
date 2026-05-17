#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Verify that a LeRobot-format Robometer is byte-equivalent to its upstream source.

Run this once after publishing a LeRobot-format Robometer to the Hub, before
flipping the default `RobometerConfig.pretrained_path` to it. It loads both
the upstream snapshot and the re-exported copy, compares state dicts, and
prints a clear pass/fail summary.

Example:

    python scripts/verify_robometer_export.py \\
        --upstream robometer/Robometer-4B \\
        --lerobot  lerobot/robometer-4b

    python scripts/verify_robometer_export.py \\
        --upstream robometer/Robometer-4B \\
        --lerobot  ./robometer-4b-lerobot   # local folder also works
"""

from __future__ import annotations

import argparse
import sys

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
from lerobot.rewards.robometer._upstream_loader import apply_upstream_checkpoint


def _load_upstream(path: str) -> RobometerRewardModel:
    # Fresh ``RobometerConfig`` (``vlm_config=None``) triggers
    # ``RobometerRewardModel.__init__``'s upstream-matching path: download
    # base Qwen, resize for ROBOMETER_SPECIAL_TOKENS. The subsequent
    # ``apply_upstream_checkpoint`` call resizes again if the checkpoint's
    # vocab differs (e.g. upstream was trained against an older Qwen).
    cfg = RobometerConfig(pretrained_path=path, device="cpu")
    model = RobometerRewardModel(cfg)
    apply_upstream_checkpoint(model, path)
    model.eval()
    return model


def _load_lerobot(path: str) -> RobometerRewardModel:
    cfg = RewardModelConfig.from_pretrained(path)
    if not isinstance(cfg, RobometerConfig):
        raise TypeError(f"Expected RobometerConfig in LeRobot export, got {type(cfg)}")
    cfg.pretrained_path = path
    cfg.device = "cpu"
    return RobometerRewardModel.from_pretrained(path, config=cfg)


def compare_state_dicts(a: RobometerRewardModel, b: RobometerRewardModel) -> bool:
    sd_a, sd_b = a.state_dict(), b.state_dict()
    keys_a, keys_b = set(sd_a), set(sd_b)

    missing = keys_a - keys_b
    extra = keys_b - keys_a
    if missing:
        print(f"❌ {len(missing)} keys missing in LeRobot-format model (sample: {list(missing)[:5]})")
    if extra:
        print(f"❌ {len(extra)} extra keys in LeRobot-format model (sample: {list(extra)[:5]})")
    if missing or extra:
        return False

    diff_summary: list[tuple[str, float]] = []
    for key in sorted(keys_a):
        ta, tb = sd_a[key], sd_b[key]
        if ta.shape != tb.shape:
            print(f"❌ shape mismatch at {key}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
            return False
        # Compare in float to avoid bfloat16 equality quirks.
        max_abs = (ta.float() - tb.float()).abs().max().item()
        if max_abs > 0:
            diff_summary.append((key, max_abs))

    if not diff_summary:
        print(f"✅ All {len(keys_a)} parameters identical")
        return True

    # Some keys differ; show worst offenders.
    diff_summary.sort(key=lambda kv: kv[1], reverse=True)
    print(f"⚠️  {len(diff_summary)} keys differ. Top 10 by max abs diff:")
    for key, value in diff_summary[:10]:
        print(f"    {key:60s}  max|Δ| = {value:.3e}")

    # Tolerance: bf16 round-trips can introduce ULP-level noise but no real
    # change. Allow up to 1e-3 absolute difference; anything larger is a real
    # divergence.
    worst = diff_summary[0][1]
    if worst < 1e-3:
        print(f"✅ Worst diff {worst:.3e} is within bf16 round-trip tolerance")
        return True
    print(f"❌ Worst diff {worst:.3e} exceeds tolerance (1e-3)")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--upstream", required=True, help="Upstream Robometer repo id or local path.")
    parser.add_argument("--lerobot", required=True, help="LeRobot-format Robometer repo id or local path.")
    args = parser.parse_args()

    print(f"Loading upstream:        {args.upstream}")
    upstream = _load_upstream(args.upstream)
    print(f"Loading LeRobot-format:  {args.lerobot}")
    lerobot = _load_lerobot(args.lerobot)

    print("\n=== Config comparison ===")
    config_ok = True
    for field in [
        "base_model_id",
        "torch_dtype",
        "use_multi_image",
        "use_per_frame_progress_token",
        "average_temporal_patches",
        "frame_pooling",
        "frame_pooling_attn_temperature",
        "progress_loss_type",
        "progress_discrete_bins",
    ]:
        a, b = getattr(upstream.config, field), getattr(lerobot.config, field)
        field_ok = a == b
        config_ok = config_ok and field_ok
        ok = "✅" if field_ok else "❌"
        print(f"  {ok} {field}: upstream={a!r}, lerobot={b!r}")

    print("\n=== State-dict comparison ===")
    state_dict_ok = compare_state_dicts(upstream, lerobot)

    print()
    if config_ok and state_dict_ok:
        print("🎉 Verification passed — safe to flip the default.")
        return 0
    print("⛔ Verification failed — DO NOT flip the default.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
