#!/usr/bin/env python
"""Generate the ``examples/vllm_policy/`` directory consumed by ``--policy.path``.

Writes ``config.json`` plus the (identity) ``policy_preprocessor.json`` /
``policy_postprocessor.json`` pipelines, using lerobot's own save machinery so the
schema always matches what ``lerobot-eval`` loads.

Usage:
    python examples/vllm_policy/scripts/generate_policy_dir.py [--out examples/vllm_policy]
"""

from __future__ import annotations

import argparse
import os

import lerobot.policies  # noqa: F401  (register built-in policies)
from lerobot.policies.vllm.configuration_vllm import VllmConfig  # noqa: E402
from lerobot.policies.vllm.processor_vllm import make_vllm_pre_post_processors  # noqa: E402
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="examples/vllm_policy", help="output directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--n-action-steps", type=int, default=16)
    args = p.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    cfg = VllmConfig(
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        image_height=256,
        image_width=256,
    )
    cfg.save_pretrained(out_dir)

    pre, post = make_vllm_pre_post_processors(cfg)
    pre.save_pretrained(out_dir, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    post.save_pretrained(out_dir, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")

    print(f"Wrote {sorted(os.listdir(out_dir))} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
