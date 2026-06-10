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
"""``SayTool`` — text-to-speech tool wrapping Kyutai's pocket-tts.

The first concrete tool implementation. PI052 and downstream runtime
dispatchers consume this when the model emits an assistant message
with ``tool_calls=[{function: {name: "say", arguments: {text: ...}}}]``.

Why pocket-tts:

- runs on CPU (no GPU dependency); ~6× real-time on a MacBook Air M4
- ~100M parameters, ~200ms first-chunk latency
- streamable, voice-cloneable
- pip-installable, MIT-style permissive license

The pocket-tts model is loaded **lazily** the first time ``call(...)``
runs (or eagerly via ``preload()``). Loading takes a few seconds and
several hundred MB of RAM, so we don't pay the cost when the tool is
merely *registered* — only when it's *invoked*.

Optional dependency. Install with::

    pip install lerobot[tools]
    # or directly:
    pip install pocket-tts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.datasets.language import SAY_TOOL_SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class SayTool:
    """Speak a short utterance via Kyutai's pocket-tts.

    Parameters
    ----------
    schema:
        Optional schema override; defaults to the canonical
        ``SAY_TOOL_SCHEMA`` from PR 1. Custom voices or extended
        argument shapes can pass in a modified schema, but the
        implementation only reads ``arguments["text"]``.
    voice:
        One of the pocket-tts catalog voices (``alba``, ``marius``,
        ``javert``, ``jean``, ``fantine``, ``cosette``, ``eponine``,
        ``azelma``) or a path to a ``.wav`` / ``.safetensors`` voice
        file for cloning. See the pocket-tts model card for licensing.
    output_dir:
        If set, every ``call(...)`` writes a ``<timestamp>.wav`` audio
        file there in addition to returning the PCM tensor.
        ``None`` (default) skips disk writes — useful for live
        playback paths that hand the tensor directly to a sounddevice
        / WebAudio sink.
    """

    schema: dict[str, Any] = field(default_factory=lambda: dict(SAY_TOOL_SCHEMA))
    voice: str = "alba"
    output_dir: Path | None = None

    name: str = field(init=False, default="say")
    _model: Any = field(init=False, default=None, repr=False)
    _voice_state: Any = field(init=False, default=None, repr=False)
    _sample_rate: int = field(init=False, default=24000, repr=False)

    # ------------------------------------------------------------------
    # Lazy model load
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """Load the pocket-tts model + voice state into memory.

        Optional — ``call(...)`` triggers this automatically on first
        invocation. Useful when you want the multi-second load to
        happen at startup rather than on the first ``say`` the policy
        emits.
        """
        if self._model is not None and self._voice_state is not None:
            return
        try:
            from pocket_tts import TTSModel  # noqa: PLC0415  (optional dep)
        except ImportError as exc:  # pragma: no cover (env-dependent)
            raise ImportError(
                "SayTool requires pocket-tts. Install with `pip install "
                "lerobot[tools]` or `pip install pocket-tts`."
            ) from exc
        logger.info("SayTool: loading pocket-tts model + voice=%r", self.voice)
        self._model = TTSModel.load_model()
        self._voice_state = self._model.get_state_for_audio_prompt(self.voice)
        self._sample_rate = int(getattr(self._model, "sample_rate", 24000))

    # ------------------------------------------------------------------
    # Tool protocol
    # ------------------------------------------------------------------

    def call(self, arguments: dict[str, Any]) -> Any:
        """Speak ``arguments["text"]`` and return the PCM tensor.

        Optionally also writes ``<output_dir>/<timestamp>.wav`` when
        ``self.output_dir`` is set. The returned tensor is a 1-D
        ``torch.Tensor`` of float32 PCM samples at
        ``self.sample_rate`` Hz — directly playable by
        ``sounddevice.play(audio.numpy(), self.sample_rate)`` or
        encodable by ``scipy.io.wavfile.write``.
        """
        text = arguments.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"SayTool.call expects arguments={{'text': str}}, got {arguments!r}"
            )
        self.preload()

        audio = self._model.generate_audio(self._voice_state, text)

        if self.output_dir is not None:
            self._write_wav(audio, text)

        return audio

    @property
    def sample_rate(self) -> int:
        """PCM sample rate of the returned tensor (Hz)."""
        return self._sample_rate

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_wav(self, audio: Any, text: str) -> Path:
        """Write a ``.wav`` next to ``output_dir`` for offline inspection."""
        import time as _time  # noqa: PLC0415

        try:
            import scipy.io.wavfile  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SayTool.output_dir requires scipy. `pip install scipy`."
            ) from exc

        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # One file per call; suffix with a millisecond timestamp + a
        # short text snippet so a directory listing is informative.
        snippet = "".join(c if c.isalnum() else "_" for c in text[:32]).strip("_")
        ts_ms = int(_time.time() * 1000)
        path = out_dir / f"say_{ts_ms}_{snippet}.wav"

        # ``audio`` is a torch tensor; pocket-tts uses CPU, so a plain
        # ``.numpy()`` is safe.
        scipy.io.wavfile.write(path, self.sample_rate, audio.numpy())
        return path
