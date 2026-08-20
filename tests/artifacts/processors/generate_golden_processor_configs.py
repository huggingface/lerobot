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

"""Pin the serialized shape of every policy's pre/post-processor pipelines.

These fixtures are the reference for the processor refactor that makes the policy config
authoritative over a checkpoint's saved pipeline (see `tests/policies/test_processor_authority.py`).
Rebuilding a pipeline from a config must produce the same structure the old deserializing loader
produced, and that can only be verified against output captured *before* the refactor landed.

Regenerate deliberately, never to make a failing test pass:

    uv run python tests/artifacts/processors/generate_golden_processor_configs.py

Check the working tree against the committed fixtures (what CI does):

    uv run python tests/artifacts/processors/generate_golden_processor_configs.py --check

A policy that cannot be built offline is recorded in `manifest.json` with its reason instead of
being silently dropped, so the coverage gap stays visible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_IMAGES, OBS_STATE

GOLDEN_DIR = Path(__file__).parent / "golden"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"

# Deliberately small and fixed. The fixtures pin pipeline *structure*, so the only thing these
# numbers have to do is be shaped plausibly and never change.
STATE_DIM = 6
ACTION_DIM = 6
IMAGE_SHAPE = (3, 96, 96)
CAMERA_NAME = "cam"


def _synthetic_features() -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    """Feature dicts covering both image key conventions, since policies disagree on which they use."""
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        f"{OBS_IMAGE}.{CAMERA_NAME}": PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE),
        f"{OBS_IMAGES}.{CAMERA_NAME}": PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return input_features, output_features


def _synthetic_stats(features: dict[str, PolicyFeature]) -> dict[str, dict[str, torch.Tensor]]:
    """Stats for every feature, covering all stat names any `NormalizationMode` may ask for."""
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, feature in features.items():
        if feature.type is FeatureType.VISUAL:
            # Per-channel, in the (C, 1, 1) layout the normalizer reshapes visual stats into.
            shape: tuple[int, ...] = (feature.shape[0], 1, 1)
        else:
            shape = feature.shape
        stats[key] = {
            "mean": torch.zeros(shape),
            "std": torch.ones(shape),
            "min": -torch.ones(shape),
            "max": torch.ones(shape),
            "q01": -torch.ones(shape),
            "q99": torch.ones(shape),
            "q10": -torch.ones(shape),
            "q90": torch.ones(shape),
        }
    return stats


def _build_pipelines(policy_type: str):
    """Build one policy's pre/post-processor pipelines from its config alone.

    Raises whatever the policy's factory raises; the caller records it as a skip.
    """
    input_features, output_features = _synthetic_features()
    config = make_policy_config(policy_type, push_to_hub=False)
    config.input_features = dict(input_features)
    config.output_features = dict(output_features)
    # cpu keeps `device_processor` output identical regardless of the generating machine.
    config.device = "cpu"

    stats = _synthetic_stats({**input_features, **output_features})
    return make_pre_post_processors(config, dataset_stats=stats)


def _build_configs(policy_type: str) -> dict[str, Any]:
    """Build one policy's pipelines and return both serialized configs."""
    preprocessor, postprocessor = _build_pipelines(policy_type)
    # Canonicalize through JSON: what these fixtures pin is the on-disk contract, and `get_config()`
    # returns Python objects that JSON flattens (feature shapes are tuples in memory, lists on disk).
    # Comparing the in-memory form against a parsed fixture would report a difference on every shape.
    return {
        "pre": _as_serialized(preprocessor.get_config()),
        "post": _as_serialized(postprocessor.get_config()),
    }


def _as_serialized(config: dict[str, Any]) -> dict[str, Any]:
    """Return `config` as it would come back after `save_pretrained` and a reload."""
    return json.loads(json.dumps(config))


def _generate() -> dict[str, dict[str, Any]]:
    """Build every known policy. A failure is fatal, including a missing optional dependency.

    Skipping was tempting and wrong: `_write` records only what it built, so regenerating in an
    environment missing an extra silently drops those policies from `covered` and the fixture set
    shrinks without anyone noticing. Run this with `--extra all`.
    """
    return {
        policy_type: _build_configs(policy_type)
        for policy_type in sorted(PreTrainedConfig.get_known_choices())
    }


def _write(built: dict[str, dict[str, Any]]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for policy_type, configs in built.items():
        for role, config in configs.items():
            path = GOLDEN_DIR / f"{policy_type}_{role}.json"
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    MANIFEST_PATH.write_text(json.dumps({"covered": sorted(built)}, indent=2, sort_keys=True) + "\n")


def _check(built: dict[str, dict[str, Any]]) -> int:
    """Compare freshly built configs against the committed fixtures. Returns an exit code."""
    if not MANIFEST_PATH.exists():
        logging.error("no fixtures committed yet; run without --check first")
        return 1
    covered = json.loads(MANIFEST_PATH.read_text())["covered"]
    mismatched: list[str] = []
    for policy_type in covered:
        if policy_type not in built:
            mismatched.append(f"{policy_type}: covered by the fixtures but no longer builds")
            continue
        for role, config in built[policy_type].items():
            path = GOLDEN_DIR / f"{policy_type}_{role}.json"
            expected = json.loads(path.read_text())
            if config != expected:
                mismatched.append(f"{policy_type} ({role}): differs from {path.name}")
    for line in mismatched:
        logging.error("%s", line)
    return 1 if mismatched else 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed fixtures instead of rewriting them",
    )
    args = parser.parse_args()

    built = _generate()
    logging.info("built %d policies", len(built))
    if args.check:
        return _check(built)
    _write(built)
    logging.info("wrote fixtures for %s", ", ".join(sorted(built)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
