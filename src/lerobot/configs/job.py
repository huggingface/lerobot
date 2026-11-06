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

from dataclasses import dataclass, field


@dataclass
class JobConfig:
    # Where training runs. None (omitted) or "local" runs on this machine.
    # Any other value is an HF Jobs flavor and submits the run to HF Jobs.
    # List available flavors + pricing with `hf jobs hardware` command.
    target: str | None = None
    # Runtime image for the remote job (ignored for local runs).
    image: str = "huggingface/lerobot-gpu:latest"
    # Max wall-clock for the remote job as an HF Jobs duration string (e.g. "2h").
    # Defaults to "2d": We pass an explicit, generous cap instead. Set a smaller
    # value to fail fast, or a larger one for long runs.
    timeout: str | None = "2d"
    # Submit and exit instead of streaming the job logs in the foreground.
    detach: bool = False
    # Extra tags attached to the HF job and to any dataset this run pushes to the
    # Hub. A "lerobot" tag is always added; e.g. --job.tags '["lelab"]' adds more.
    tags: list[str] = field(default_factory=list)

    # Two entry points to the same predicate: the staticmethod tests a raw target string
    # straight from argv (before any JobConfig exists, to decide dispatch early), while the
    # property is the ergonomic accessor for code that already holds a config instance.
    @staticmethod
    def is_remote_target(target: str | None) -> bool:
        """True when `target` names an HF Jobs flavor rather than a local run."""
        return target not in (None, "local")

    @property
    def is_remote(self) -> bool:
        """True when training should run on HF Jobs rather than this machine."""
        return self.is_remote_target(self.target)
