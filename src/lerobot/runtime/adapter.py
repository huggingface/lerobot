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

"""Policy adapter base class for the language-conditioned runtime.

The runtime loop drives the *control algorithm* (throttling, output
rejection, the subtask -> memory cascade, diagnostics) and delegates the
*policy primitives* (act, generate text) to an adapter. :class:`BaseLanguageAdapter`
implements the algorithm once; a policy subclasses it and supplies:

* :meth:`select_action` — observation + language context -> action chunk
* :meth:`generate_text` — a text stream (``kind``) -> decoded string
* :meth:`build_messages` — the prompt for each ``kind``

A policy that needs full control can instead satisfy the
:class:`LanguageConditionedPolicyAdapter` protocol directly.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .language_runtime import RuntimeState

_SAY_RE = re.compile(r"<\s*say\s*>(.*?)<\s*/\s*say\s*>", re.IGNORECASE | re.DOTALL)


@dataclass
class GenerationConfig:
    """Text-generation knobs, fixed for the lifetime of an adapter.

    These are configuration (set once from the CLI), not per-tick runtime
    state — they live on the adapter, never in :class:`RuntimeState`.
    """

    min_new_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    chunks_per_regen: int = 1  # regenerate the language context every N action chunks
    enable_memory: bool = True  # generate a running memory note on subtask change
    enable_subtask: bool = True  # generate the low-level subtask (off => use the given text directly)


@dataclass
class LanguageDiagnostics:
    """Rejection / repeat counters surfaced in the runtime panel.

    Keyed by text ``kind`` (``subtask`` / ``memory`` / ...) so the same
    accounting works for any cascade shape.
    """

    last_raw: dict[str, str] = field(default_factory=dict)
    empty: dict[str, int] = field(default_factory=dict)
    gibberish: dict[str, int] = field(default_factory=dict)
    repeat: int = 0

    def _bump(self, table: dict[str, int], kind: str) -> int:
        table[kind] = table.get(kind, 0) + 1
        return table[kind]


class BaseLanguageAdapter(ABC):
    """Batteries-included adapter: generic high-level control, policy primitives abstract."""

    def __init__(self, policy: Any, gen: GenerationConfig | None = None) -> None:
        self.policy = policy
        self.gen = gen or GenerationConfig()
        self.diag = LanguageDiagnostics()
        self._chunks_until_regen = 0

    # --- policy primitives (subclass supplies) ---------------------------

    @abstractmethod
    def select_action(self, observation: dict[str, Any], state: RuntimeState) -> Any:
        """Produce an action chunk from the observation + current language context."""

    @abstractmethod
    def generate_text(
        self,
        kind: str,
        observation: dict[str, Any] | None,
        state: RuntimeState,
        user_text: str | None = None,
    ) -> str:
        """Generate one text stream (``kind``) and return the decoded string."""

    # --- generic control algorithm (runtime calls these) ----------------

    def update_language_state(self, observation: dict[str, Any] | None, state: RuntimeState) -> None:
        """Throttled regeneration of the language context (subtask / memory / ...)."""
        if self._chunks_until_regen > 0:
            self._chunks_until_regen -= 1
            return
        self._chunks_until_regen = max(1, self.gen.chunks_per_regen) - 1
        self._regenerate_context(observation, state)

    def handle_interjection(
        self, user_text: str, observation: dict[str, Any] | None, state: RuntimeState
    ) -> None:
        """React to a mid-run user message by regenerating the plan."""
        out = self.generate_text("interjection", observation, state, user_text=user_text)
        plan = self.plan_from_text(out)
        if plan:
            state.set_context("plan", plan, label="plan")

    def plan_from_text(self, text: str) -> str:
        """Strip ``<say>`` speech markers and reject gibberish plans."""
        plan, _speech = split_plan_and_say(text)
        return "" if looks_like_gibberish(plan) else plan

    # --- overridable cascade + shared helpers ---------------------------

    def _regenerate_context(self, observation: dict[str, Any] | None, state: RuntimeState) -> None:
        """Default hierarchy: regenerate the subtask, then memory when it changes.

        Override for a policy with a different language hierarchy.
        """
        if not self.gen.enable_subtask:
            # Direct-subtask mode: the operator supplies the subtask; don't
            # generate (and thus don't overwrite) it.
            return
        subtask = self._generate_filtered("subtask", observation, state)
        if subtask is None:
            return
        previous = state.language_context.get("subtask")
        if not state.set_context("subtask", subtask, label="subtask"):
            self.diag.repeat += 1
            return
        self.diag.repeat = 0
        if previous:
            state.extra["prior_subtask"] = previous
        if not self.gen.enable_memory:
            return
        memory = self._generate_filtered("memory", observation, state)
        if memory is not None:
            state.set_context("memory", memory, label="memory")

    def _generate_filtered(
        self, kind: str, observation: dict[str, Any] | None, state: RuntimeState
    ) -> str | None:
        """Generate one ``kind``, record diagnostics, drop empty / gibberish output."""
        text = self.generate_text(kind, observation, state)
        self.diag.last_raw[kind] = text or ""
        if not text:
            count = self.diag._bump(self.diag.empty, kind)
            if count == 1 or count % 5 == 0:
                state.log(f"  [info] {kind} gen returned empty (x{count})")
            return None
        if looks_like_gibberish(text):
            count = self.diag._bump(self.diag.gibberish, kind)
            if count == 1 or count % 30 == 0:
                state.log(f"  [info] {kind} gen rejected (gibberish x{count}): {text[:60]!r}")
            return None
        return text


def looks_like_gibberish(text: str) -> bool:
    """Heuristic filter for malformed / collapsed LM-head output."""
    if not text or not text.strip():
        return True
    stripped = text.strip()
    alpha = sum(1 for c in stripped if c.isalpha())
    if alpha < max(3, len(stripped) // 8):
        return True
    if stripped.startswith('":') and stripped.count('"') > stripped.count(" "):
        return True
    if len(set(stripped)) <= 2 and len(stripped) > 4:
        return True
    cleaned = stripped.replace("\n", " ").replace(":", " ")
    for marker in ("Assistant", "User", "Ass "):
        if marker in cleaned and len(cleaned.split()) < 4:
            return True
    tokens = [t for t in cleaned.split() if any(c.isalpha() for c in t)]
    unique_alpha = {t.lower() for t in tokens}
    if len(unique_alpha) < 3 and len(stripped) < 80:
        return True
    return len(tokens) >= 8 and len(unique_alpha) <= max(3, len(tokens) // 10)


def split_plan_and_say(text: str) -> tuple[str, str]:
    """Split ``plan <say>speech</say>`` into ``(plan, speech)``."""
    if not text:
        return "", ""
    match = _SAY_RE.search(text)
    if not match:
        return text.strip(), ""
    speech = match.group(1).strip().strip('"').strip("'")
    plan = (text[: match.start()] + text[match.end() :]).strip()
    return plan, speech
