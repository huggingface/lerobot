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

"""Per-client session state and the latest-only observation mailbox.

The server holds **no cross-request control state**: RTC prefixes and
delay hints arrive with every observation.  What a session does hold:

- Per-session processor pipeline instances.  Mandatory:
  ``RelativeActionsProcessorStep`` caches ``_last_state`` at preprocess
  and the postprocessor reads it back — a pipeline shared across clients
  would be a race.
- A one-slot mailbox: the newest observation wins; superseded requests
  are counted so drops stay visible to the client.
- Counters for the audit log and ``/metrics``.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)

from .schema import MsgHeader

if TYPE_CHECKING:
    pass


@dataclass
class MailboxItem:
    header: MsgHeader
    payload: bytes
    recv_mono: float  # server-local monotonic deposit time (for queue_wait_ms)


@dataclass
class SessionStats:
    requests: int = 0
    errors: int = 0
    superseded: int = 0  # observations overwritten before inference (lifetime)
    superseded_since_reply: int = 0
    last_inference_ms: float = 0.0
    last_queue_wait_ms: float = 0.0


class Session:
    """One connected robot client."""

    def __init__(
        self,
        session_id: str,
        client_uuid: str,
        task: str,
        robot_type: str,
        rtc_enabled: bool,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        action_publisher: Any = None,  # zenoh.Publisher (Any: zenoh optional at import)
    ) -> None:
        self.session_id = session_id
        self.client_uuid = client_uuid
        self.task = task
        self.robot_type = robot_type
        self.rtc_enabled = rtc_enabled
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.action_publisher = action_publisher

        self.episode_id = 0
        self.stats = SessionStats()
        self.alive = True
        self.last_seen_mono = time.monotonic()
        # Set when the client's liveliness token drops; GC after grace.
        self.token_dropped_mono: float | None = None

        # Processor introspection for relative-action prefix re-anchoring
        # (mirrors RTCInferenceEngine.__init__).
        self.relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        self.normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )

        self._mailbox: MailboxItem | None = None
        self._mailbox_lock = Lock()

    # ------------------------------------------------------------------
    # Mailbox (deposit-only callbacks write, the inference worker reads)
    # ------------------------------------------------------------------

    def deposit(self, header: MsgHeader, payload: bytes) -> None:
        """Latest-only deposit; counts superseded observations."""
        item = MailboxItem(header=header, payload=payload, recv_mono=time.monotonic())
        with self._mailbox_lock:
            if self._mailbox is not None:
                self.stats.superseded += 1
                self.stats.superseded_since_reply += 1
            self._mailbox = item
            self.alive = True
            self.token_dropped_mono = None
            self.last_seen_mono = item.recv_mono

    def take(self) -> MailboxItem | None:
        with self._mailbox_lock:
            item, self._mailbox = self._mailbox, None
            return item

    def take_superseded(self) -> int:
        """Atomically read-and-reset the per-reply supersession counter."""
        with self._mailbox_lock:
            count = self.stats.superseded_since_reply
            self.stats.superseded_since_reply = 0
            return count

    def has_pending(self) -> bool:
        with self._mailbox_lock:
            return self._mailbox is not None

    def clear_mailbox(self) -> None:
        with self._mailbox_lock:
            self._mailbox = None

    # ------------------------------------------------------------------
    # Episode boundary
    # ------------------------------------------------------------------

    def reset_episode(self, episode_id: int | None = None) -> None:
        """Clear per-episode state.  The shared policy is NOT touched here."""
        self.clear_mailbox()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.episode_id = episode_id if episode_id is not None else self.episode_id + 1

    def close(self) -> None:
        self.clear_mailbox()
        publisher = self.action_publisher
        self.action_publisher = None
        if publisher is not None:
            # Already-closed transport is fine on teardown.
            with contextlib.suppress(Exception):
                publisher.undeclare()


class SessionRegistry:
    """Thread-safe map of client_uuid → Session."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    def add(self, session: Session) -> Session | None:
        """Register, returning a displaced same-client session (caller closes it)."""
        with self._lock:
            old = self._sessions.get(session.client_uuid)
            self._sessions[session.client_uuid] = session
            return old

    def get(self, client_uuid: str) -> Session | None:
        with self._lock:
            return self._sessions.get(client_uuid)

    def remove(self, client_uuid: str, expected: Session | None = None) -> Session | None:
        """Remove by uuid; with ``expected``, only if it is still that exact session.

        The identity check stops a GC sweep that snapshotted an old
        session from tearing down its just-handshaked replacement.
        """
        with self._lock:
            current = self._sessions.get(client_uuid)
            if current is None or (expected is not None and current is not expected):
                return None
            return self._sessions.pop(client_uuid)

    def snapshot(self) -> list[Session]:
        with self._lock:
            return list(self._sessions.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
