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

"""Scheduling seam between the session registry and the inference worker.

The v1 scheduler is strict round-robin over sessions with a pending
observation: every ready session gets exactly one inference per cycle,
so starvation is structurally impossible.  The seam exists so that
cross-session micro-batching can land later without redesign (blocked
today on ``predict_action_chunk`` taking a *scalar* ``inference_delay``).
"""

from __future__ import annotations

import abc

from .session import Session


class Scheduler(abc.ABC):
    """Pick which ready session(s) the worker serves next."""

    @abc.abstractmethod
    def select(self, ready: list[Session]) -> list[Session]:
        """Return the sessions to serve this cycle (subset of ``ready``)."""


class RoundRobinScheduler(Scheduler):
    """Serve one session per cycle, fairly, in client_uuid order."""

    def __init__(self) -> None:
        self._last_served: str | None = None

    def select(self, ready: list[Session]) -> list[Session]:
        if not ready:
            return []
        ring = sorted(ready, key=lambda s: s.client_uuid)
        if self._last_served is not None:
            for i, session in enumerate(ring):
                if session.client_uuid > self._last_served:
                    ring = ring[i:] + ring[:i]
                    break
            else:
                pass  # wrap: everyone is <= last served, keep sorted order
        chosen = ring[0]
        self._last_served = chosen.client_uuid
        return [chosen]
