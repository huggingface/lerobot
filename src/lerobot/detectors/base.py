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

"""Detector output type shared by event-triggered / speed-adaptive replanning.

A *detector* is any callable ``frame -> bool | DetectorOutput`` that inspects a
single camera frame and decides whether an early replan should fire (and, for
speed-adaptive detectors, what replan threshold to use). Detectors are pure and
transport-agnostic: they are reused by both the async-inference client (camera
polled on its own thread) and the RTC rollout engine (frame taken from the
control-loop observation).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorOutput:
    """Structured detector output for event-triggered and speed-adaptive replanning."""

    replan_now: bool = False
    target_visible: bool | None = None
    center_px: tuple[float, float] | None = None
    speed_px_s: float | None = None
    effective_chunk_size_threshold: float | None = None
    reason: str = ""


def normalize_detector_output(output: bool | DetectorOutput) -> DetectorOutput:
    """Coerce a detector's return value to a ``DetectorOutput``.

    Boolean detectors (e.g. motion) are mapped to ``replan_now`` with a
    ``"motion"`` reason when they fire, so downstream gating code only ever
    handles the structured form.
    """
    if isinstance(output, DetectorOutput):
        return output
    return DetectorOutput(replan_now=output, reason="motion" if output else "")
