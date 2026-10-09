# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Importable modules exercise the actual spawn worker and artifact contracts."""

import time
from pathlib import Path

import pyarrow as pa

from lerobot.data_processing.types import ItemResult, ModuleSpec, Outcome


class Echo:
    def __init__(self, log_path=None, fail_key=None, failure_marker=None, bad_schema=False):
        self.log_path = log_path
        self.fail_key = fail_key
        self.failure_marker = failure_marker
        self.bad_schema = bad_schema
        self.schema = pa.schema([("key", pa.string()), ("seed", pa.int64())])
        self.spec = ModuleSpec(
            "echo",
            "1",
            "episode",
            {"echo": self.schema},
            hf_jobs_compatible=not log_path and not failure_marker,
        )

    def log(self, text):
        if self.log_path:
            with Path(self.log_path).open("a") as stream:
                stream.write(text + "\n")

    def setup(self, context):
        self.log("setup")

    def teardown(self):
        self.log("teardown")

    def process_batch(self, items, context):
        results = []
        for item in items:
            self.log(item.key)
            if item.key == self.fail_key and not Path(self.failure_marker).exists():
                Path(self.failure_marker).touch()
                raise RuntimeError("injected interruption")
            if item.payload.get("missing"):
                results.append(ItemResult(item.item_id, Outcome.MASKED, reason="missing_camera"))
                continue
            table = pa.Table.from_pylist([{"key": item.key, "seed": item.seed}], schema=self.schema)
            if self.bad_schema:
                table = table.drop(["seed"])
            artifact = context.write_parquet(item, "echo", table)
            results.append(ItemResult(item.item_id, Outcome.COMPLETED, (artifact,)))
        return results


class RendezvousEcho(Echo):
    """Two independent spawned stages must overlap, not merely finish in order."""

    def __init__(self, signal, peer):
        super().__init__()
        self.signal, self.peer = Path(signal), Path(peer)

    def process_batch(self, items, context):
        self.signal.touch()
        deadline = time.monotonic() + 20
        while not self.peer.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("Independent stage never ran concurrently")
            time.sleep(0.01)
        return super().process_batch(items, context)


class TemporaryEcho(Echo):
    def __init__(self, marker, permanent=False):
        super().__init__()
        self.marker, self.permanent = Path(marker), permanent

    def process_batch(self, items, context):
        if self.permanent:
            raise ValueError("invalid output schema")
        if not self.marker.exists():
            self.marker.touch()
            raise ConnectionError("temporary connection failure")
        return super().process_batch(items, context)
