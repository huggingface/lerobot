#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the complete RL-token software path without PI0 weights or robot hardware."""

import argparse
import logging
from pathlib import Path

from lerobot.rl.algorithms.rlt.toy import run_toy_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rl_token_toy"))
    parser.add_argument("--env-steps", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_toy_workflow(args.output_dir, total_env_steps=args.env_steps)
    logging.info(
        "complete: stage1=%d env_steps=%d replay=%d updates=%d actor_updates=%d",
        result.stage1_steps,
        result.env_steps,
        result.replay_size,
        result.gradient_updates,
        result.actor_updates,
    )


if __name__ == "__main__":
    main()
