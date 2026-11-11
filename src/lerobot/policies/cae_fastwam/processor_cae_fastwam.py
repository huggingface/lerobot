# Copyright 2026 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
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

"""Processors of CAE-FastWAM: FastWAM's pipelines, normalized with the selected task suite's statistics.

The steps are FastWAM's (standard min/max normalizer and unnormalizer, gripper toggle; images are resized
inside the model). The only addition: a checkpoint trained on several task suites keeps one set of
statistics per suite in `CAEFastWAMConfig.task_suite_stats`, and the (un)normalizer uses the set of
`config.task_suite` (one suite per run).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lerobot.configs import PreTrainedConfig
from lerobot.policies.fastwam.processor_fastwam import (
    make_fastwam_pre_post_processors,
    make_fastwam_pre_post_processors_from_pretrained,
)
from lerobot.processor import PolicyProcessorPipeline, hotswap_stats
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_cae_fastwam import CAEFastWAMConfig


def _selected_suite_stats(config: CAEFastWAMConfig) -> dict[str, dict[str, Any]] | None:
    """Statistics of the suite to run, or None when the config has no per-suite statistics or no suite is
    selected (several suites and no `task_suite`)."""
    if not config.task_suite_stats or (config.task_suite is None and len(config.task_suites) > 1):
        return None
    return config.task_suite_stats[config.resolve_task_suite()]


def make_cae_fastwam_pre_post_processors(
    config: CAEFastWAMConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Create the CAE-FastWAM pre- and post-processing pipelines (FastWAM's).

    Args:
        config (CAEFastWAMConfig): Policy configuration.
        dataset_stats (dict | None): LeRobot statistics, used when the config has no statistics for the
            selected suite.

    Returns:
        tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]: Input and output pipelines.
    """
    return make_fastwam_pre_post_processors(config, _selected_suite_stats(config) or dataset_stats)


def make_cae_fastwam_pre_post_processors_from_pretrained(
    config: CAEFastWAMConfig,
    pretrained_path: str | Path,
    *,
    revision: str | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    **_: Any,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Load FastWAM's saved processors and give them the statistics of `config.task_suite`.

    The action VAE was trained on actions normalized with these statistics, so they also win over the
    dataset statistics that `lerobot-train` sends when fine-tuning (as in the original recipe). A config
    without statistics (`--policy.type=cae_fastwam`, which initializes from `base_model_id`) takes the
    pretrained checkpoint's.
    """
    if config.task_suite_stats is None:
        pretrained_config = PreTrainedConfig.from_pretrained(pretrained_path, revision=revision)
        config.task_suite_stats = getattr(pretrained_config, "task_suite_stats", None)
    preprocessor, postprocessor = make_fastwam_pre_post_processors_from_pretrained(
        config,
        pretrained_path,
        revision=revision,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
        preprocessor_config_filename=preprocessor_config_filename,
        postprocessor_config_filename=postprocessor_config_filename,
    )
    stats = _selected_suite_stats(config)
    if stats is None:
        return preprocessor, postprocessor
    if "stats" in (preprocessor_overrides or {}).get("normalizer_processor", {}):
        logging.warning(
            "CAE-FastWAM normalizes with the checkpoint's %r statistics, not the dataset statistics.",
            config.resolve_task_suite(),
        )
    return hotswap_stats(preprocessor, stats), hotswap_stats(postprocessor, stats)
