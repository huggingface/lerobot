#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for legacy in-policy normalization detection during checkpoint load."""

import logging
import shlex

from lerobot.policies.utils import (
    has_legacy_normalization_keys,
    log_model_loading_keys,
    warn_legacy_normalization_keys,
)
from lerobot.processor.pipeline import build_processor_migration_command


def test_has_legacy_normalization_keys_detects_buffer_prefixes():
    assert has_legacy_normalization_keys(["normalize_inputs.buffer_observation_image.mean", "model.weight"])
    assert has_legacy_normalization_keys(["unnormalize_outputs.buffer_action.std"])
    assert not has_legacy_normalization_keys(["model.weight", "model.bias"])


def test_build_processor_migration_command_uses_module_execution():
    command = build_processor_migration_command("lerobot/diffusion_pusht")
    assert command.startswith("python -m lerobot.processor.migrate_policy_normalization ")
    assert "--pretrained-path lerobot/diffusion_pusht" in command
    assert "--revision" not in command


def test_build_processor_migration_command_quotes_paths_with_spaces():
    path_with_spaces = "/models/old policy"
    command = build_processor_migration_command(path_with_spaces, revision="legacy-branch")
    assert "python -m lerobot.processor.migrate_policy_normalization" in command
    assert f"--pretrained-path {shlex.quote(path_with_spaces)}" in command
    assert "--revision legacy-branch" in command


def test_warn_legacy_normalization_keys_emits_migration_command(caplog):
    unexpected = [
        "normalize_inputs.buffer_observation_state.mean",
        "unnormalize_outputs.buffer_action.std",
    ]

    with caplog.at_level(logging.WARNING):
        warn_legacy_normalization_keys(
            unexpected,
            pretrained_name_or_path="lerobot/diffusion_pusht",
        )

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "legacy in-policy normalization" in joined
    assert "python -m lerobot.processor.migrate_policy_normalization" in joined
    assert "--pretrained-path lerobot/diffusion_pusht" in joined
    assert "--revision" not in joined


def test_warn_legacy_normalization_keys_includes_revision_in_migration_command(caplog):
    unexpected = ["normalize_inputs.buffer_observation_state.mean"]

    with caplog.at_level(logging.WARNING):
        warn_legacy_normalization_keys(
            unexpected,
            pretrained_name_or_path="lerobot/diffusion_pusht",
            revision="legacy-branch",
        )

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "--pretrained-path lerobot/diffusion_pusht" in joined
    assert "--revision legacy-branch" in joined


def test_warn_legacy_normalization_keys_quotes_local_paths_with_spaces(caplog):
    path_with_spaces = "/models/old policy"
    unexpected = ["normalize_inputs.buffer_observation_state.mean"]

    with caplog.at_level(logging.WARNING):
        warn_legacy_normalization_keys(
            unexpected,
            pretrained_name_or_path=path_with_spaces,
        )

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert f"--pretrained-path {shlex.quote(path_with_spaces)}" in joined


def test_warn_legacy_normalization_keys_skips_unrelated_keys(caplog):
    with caplog.at_level(logging.WARNING):
        warn_legacy_normalization_keys(["some_other.unexpected.key"])

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "legacy in-policy normalization" not in joined


def test_log_model_loading_keys_does_not_emit_migration_warning(caplog):
    """Legacy migration is handled via pre-dispatch inspection, not key logging."""
    unexpected = ["normalize_inputs.buffer_observation_state.mean"]

    with caplog.at_level(logging.WARNING):
        log_model_loading_keys([], unexpected)

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "Unexpected key(s) when loading model" in joined
    assert "legacy in-policy normalization" not in joined
