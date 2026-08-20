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

"""Pin the serialized shape of every policy's processor pipelines, and its save/load contract.

Touching a `ProcessorStep` — converting it to a specialized base, adding a field, changing what
`get_config()` emits — changes what a checkpoint's pipeline looks like on disk. Those changes are
mostly invisible: a step silently added to a pipeline is train/inference skew with no warning, and a
`get_config()` key the constructor cannot accept only fails the next time somebody loads a
checkpoint. These tests make both loud.

Three guarantees, cheapest first:

1. `test_every_policy_builds_from_config_alone` — the pipeline builds at all.
2. `test_serialized_pipeline_matches_the_golden_fixture` — its serialized structure has not drifted
   from the committed fixture (step list, per-step config, state files).
3. `test_pipelines_survive_a_save_load_roundtrip` — `save_pretrained` output can actually be read
   back by `from_pretrained`, and the reloaded pipeline serializes identically. This is the one that
   catches a step which is registered but cannot be reconstructed from its own `get_config()`.

**Marked `all_extras`, and deliberately without a skip path.** A missing dependency is a failure
here, not a skip: the fixture set is only meaningful if it covers every policy, and a per-policy skip
silently narrows coverage to whatever the runner happened to have installed. Building a pipeline is
also not hermetic — `MolmoAct2PackInputsProcessorStep.__post_init__` calls `require_package("scipy")`,
and `ActionTokenizerProcessorStep.__post_init__` reaches the Hub for a `trust_remote_code` processor —
so these need `--extra all` plus network, which is `full_tests.yml`. `fast_tests.yml` deselects the
marker.

Fixtures live in `tests/artifacts/processors/golden/` and are regenerated deliberately:

    uv run python tests/artifacts/processors/generate_golden_processor_configs.py

Never regenerate to make a failing test pass. A diff here is a real change to the on-disk contract;
the fixture update belongs in the same commit as the change that caused it, so review sees both.
"""

import json
import sys
from pathlib import Path

import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

ARTIFACTS_DIR = Path(__file__).parents[1] / "artifacts" / "processors"
sys.path.insert(0, str(ARTIFACTS_DIR))
from generate_golden_processor_configs import (  # noqa: E402
    _as_serialized,
    _build_configs,
    _build_pipelines,
)

pytestmark = pytest.mark.all_extras

GOLDEN_DIR = ARTIFACTS_DIR / "golden"
GENERATOR_PATH = "tests/artifacts/processors/generate_golden_processor_configs.py"
COVERED = sorted(json.loads((GOLDEN_DIR / "manifest.json").read_text())["covered"])

# `save_pretrained` names each config file after the pipeline, so these are the filenames
# `from_pretrained` has to be pointed at.
ROLE_FILENAMES = {
    "pre": f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    "post": f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
}


def test_the_fixtures_cover_every_known_policy():
    """`covered` must list every policy, so no policy can quietly fall out of the fixture set."""
    known = set(PreTrainedConfig.get_known_choices())
    missing = sorted(known - set(COVERED))
    stale = sorted(set(COVERED) - known)
    assert not (missing or stale), (
        f"golden fixtures are out of sync with the policy registry: "
        f"no fixture for {missing or 'nothing'}, fixture for unknown {stale or 'nothing'}. "
        f"Regenerate with `uv run python {GENERATOR_PATH}` using --extra all."
    )


@pytest.mark.parametrize("policy_type", sorted(PreTrainedConfig.get_known_choices()))
def test_every_policy_builds_from_config_alone(policy_type):
    """Every policy must build its pipelines from its config, with no checkpoint involved."""
    _build_configs(policy_type)


@pytest.mark.parametrize("policy_type", COVERED)
def test_serialized_pipeline_matches_the_golden_fixture(policy_type):
    """The serialized pipeline structure must match the committed fixture, step for step."""
    for role, config in _build_configs(policy_type).items():
        expected = json.loads((GOLDEN_DIR / f"{policy_type}_{role}.json").read_text())
        assert config == expected, (
            f"{policy_type} {role} pipeline drifted from {policy_type}_{role}.json. "
            "If the change is intended, regenerate the fixtures in this same commit."
        )


def _shape(config: dict) -> list[tuple[str | None, tuple[str, ...]]]:
    """The part of a serialized pipeline a reload must reproduce: each step, and its config keys.

    Config *values* are deliberately excluded, because a step is allowed to rewrite them when it
    vendors an asset into the checkpoint. `TokenizerProcessorStep` does exactly that: it writes the
    tokenizer files next to the config so the checkpoint loads offline, so on reload `tokenizer_name`
    points at the vendored copy rather than the original Hub id. That rewrite is the feature. Exact
    values are already pinned by `test_serialized_pipeline_matches_the_golden_fixture`.
    """
    return [
        (step.get("registry_name") or step.get("class_name"), tuple(sorted(step.get("config", {}))))
        for step in config["steps"]
    ]


@pytest.mark.parametrize("policy_type", COVERED)
def test_pipelines_survive_a_save_load_roundtrip(policy_type, tmp_path):
    """What `save_pretrained` writes must be readable back by `from_pretrained`.

    A step can serialize fine and still be unloadable — if `get_config()` omits a required
    constructor argument, or emits a key the constructor does not accept, the failure only surfaces
    the next time somebody loads a checkpoint. `from_pretrained` raising is the primary assertion
    here; comparing shapes then catches a step silently dropped or reconstructed differently.
    """
    preprocessor, postprocessor = _build_pipelines(policy_type)

    for role, pipeline in (("pre", preprocessor), ("post", postprocessor)):
        save_dir = tmp_path / f"{policy_type}_{role}"
        pipeline.save_pretrained(save_dir)

        reloaded = PolicyProcessorPipeline.from_pretrained(save_dir, config_filename=ROLE_FILENAMES[role])

        expected = json.loads((GOLDEN_DIR / f"{policy_type}_{role}.json").read_text())
        assert _shape(_as_serialized(reloaded.get_config())) == _shape(expected), (
            f"{policy_type} {role} pipeline did not survive a save_pretrained/from_pretrained "
            "roundtrip: the reloaded pipeline has a different step or config shape."
        )
