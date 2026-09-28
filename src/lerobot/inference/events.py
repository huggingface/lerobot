# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Bounded inference event sidecar; disk IO stays off the motor thread."""

import json
import logging
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

logger = logging.getLogger(__name__)


class EventWriter:
    """Drop overflowing telemetry instead of waiting for storage on a motor tick."""

    def __init__(self, path: Path, capacity: int = 1024):
        """Start one bounded writer for a session's JSONL sidecar."""
        self.path = path
        self.queue: Queue[dict] = Queue(maxsize=capacity)
        self.dropped = 0
        self._stop = Event()
        self._thread = Thread(target=self._run, name="InferenceEvents", daemon=True)
        self._thread.start()

    def write(self, event: dict) -> None:
        """Enqueue an owned JSON-compatible event without waiting for disk."""
        try:
            self.queue.put_nowait(event)
        except Full:
            self.dropped += 1

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                while not self._stop.is_set() or not self.queue.empty():
                    try:
                        event = self.queue.get(timeout=0.1)
                    except Empty:
                        stream.flush()
                        continue
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.write(json.dumps({"event": "writer_closed", "dropped_events": self.dropped}) + "\n")
        except Exception:
            logger.exception("Inference sidecar writer failed: %s", self.path)

    def close(self) -> None:
        """Request draining and bound teardown time if storage stalls."""
        self._stop.set()
        self._thread.join(timeout=2)
