#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Pinpoint exactly which rows of ``embed_tokens`` / ``lm_head`` differ.

Useful follow-up to ``scripts/verify_robometer_export.py`` when the verifier
reports a small tail of differing keys but you want to know whether the
diff is:

1. Concentrated in the 5 special-token rows added by ``resize_token_embeddings``
   (expected non-determinism: mean-resize sampling differs between runs).
2. Spread across the full vocabulary (would point to a real loading bug).

Also confirms whether ``apply_upstream_checkpoint`` actually overwrites the
embed/lm-head tensors when loading the upstream state dict (vs. silently
skipping them due to a key mismatch).
"""

from __future__ import annotations

import argparse
import sys

import torch
from safetensors.torch import load_file

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
from lerobot.rewards.robometer._upstream_loader import (
    _download_robometer_snapshot,
    _remap_state_dict_keys,
    _resolve_checkpoint_safetensors_files,
    apply_upstream_checkpoint,
)

EMBED_KEY = "model.model.language_model.embed_tokens.weight"
LMHEAD_KEY = "model.lm_head.weight"


def _load_upstream(path: str) -> RobometerRewardModel:
    cfg = RobometerConfig(pretrained_path=path, device="cpu")
    model = RobometerRewardModel(cfg)
    apply_upstream_checkpoint(model, path)
    model.eval()
    return model


def _load_lerobot(path: str) -> RobometerRewardModel:
    cfg = RewardModelConfig.from_pretrained(path)
    if not isinstance(cfg, RobometerConfig):
        raise TypeError(f"Expected RobometerConfig, got {type(cfg)}")
    cfg.pretrained_path = path
    cfg.device = "cpu"
    return RobometerRewardModel.from_pretrained(path, config=cfg)


def _inspect_upstream_state_dict(upstream_path: str, model: RobometerRewardModel) -> None:
    """Dump the upstream state-dict view of the embed/lm-head tensors.

    Loads the raw upstream safetensors (pre-remap), runs the remapper, and
    reports whether the embed/lm-head keys survive into the merged dict that
    eventually hits ``model.load_state_dict``.
    """
    snapshot_dir = _download_robometer_snapshot(upstream_path)
    files = _resolve_checkpoint_safetensors_files(snapshot_dir)
    merged: dict[str, torch.Tensor] = {}
    for path in files:
        merged.update(load_file(str(path)))
    remapped = _remap_state_dict_keys(merged, model)

    print(f"\n=== Upstream state-dict inspection (snapshot at {snapshot_dir}) ===")
    print(f"raw keys (before remap)  : {len(merged)}")
    print(f"keys after remap         : {len(remapped)}")
    print(f"model expects (state_dict): {len(model.state_dict())}")

    expected = set(model.state_dict())
    present_after_remap = set(remapped) & expected
    print(f"keys present after remap : {len(present_after_remap)}")

    missing_keys = expected - set(remapped)
    print(f"keys missing from remap  : {len(missing_keys)}")
    if missing_keys:
        sample = list(missing_keys)[:10]
        print(f"  sample missing keys    : {sample}")

    unexpected_keys = set(remapped) - expected
    print(f"keys unexpected by model : {len(unexpected_keys)}")
    if unexpected_keys:
        sample = list(unexpected_keys)[:10]
        print(f"  sample unexpected keys : {sample}")

    for key in (EMBED_KEY, LMHEAD_KEY):
        present = key in remapped
        shape = tuple(remapped[key].shape) if present else None
        print(f"  {key:60s}  present={present}, shape={shape}")


def _diff_embed(name: str, a: torch.Tensor, b: torch.Tensor, special_token_count: int) -> None:
    a = a.float()
    b = b.float()
    if a.shape != b.shape:
        print(f"❌ {name} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
        return

    abs_diff = (a - b).abs()
    per_row_max = abs_diff.max(dim=1).values
    nz_rows = (per_row_max > 0).nonzero(as_tuple=True)[0].tolist()
    print(f"\n=== {name} (shape {tuple(a.shape)}) ===")
    print(f"global max|Δ|         = {abs_diff.max().item():.3e}")
    print(f"rows with any diff    = {len(nz_rows)}")
    if nz_rows:
        first = nz_rows[:10]
        last = nz_rows[-10:]
        print(f"  first nonzero rows  = {first}")
        print(f"  last nonzero rows   = {last}")
        vocab_size = a.shape[0]
        base_vocab = vocab_size - special_token_count
        special_rows = list(range(base_vocab, vocab_size))
        in_special = [r for r in nz_rows if r in special_rows]
        out_special = [r for r in nz_rows if r not in special_rows]
        print(
            f"  diffs in special-token rows ({base_vocab}..{vocab_size - 1}): {len(in_special)}/{special_token_count}"
        )
        print(f"  diffs in base-vocab rows  (0..{base_vocab - 1})           : {len(out_special)}")
        for r in special_rows:
            print(
                f"    row {r}: max|Δ|={per_row_max[r].item():.3e}, "
                f"upstream_norm={a[r].norm().item():.3e}, lerobot_norm={b[r].norm().item():.3e}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--lerobot", required=True)
    parser.add_argument(
        "--special-token-count",
        type=int,
        default=5,
        help="Number of special tokens Robometer adds. Defaults to len(ROBOMETER_SPECIAL_TOKENS)=5.",
    )
    args = parser.parse_args()

    print(f"Loading upstream:        {args.upstream}")
    upstream = _load_upstream(args.upstream)
    print(f"Loading LeRobot-format:  {args.lerobot}")
    lerobot = _load_lerobot(args.lerobot)

    _inspect_upstream_state_dict(args.upstream, upstream)

    sd_u, sd_l = upstream.state_dict(), lerobot.state_dict()

    for key in (EMBED_KEY, LMHEAD_KEY):
        if key not in sd_u or key not in sd_l:
            print(f"❌ key missing: {key} (upstream={key in sd_u}, lerobot={key in sd_l})")
            continue
        _diff_embed(key, sd_u[key], sd_l[key], args.special_token_count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
