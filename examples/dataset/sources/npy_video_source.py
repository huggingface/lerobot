# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""A single source recipe includes acquisition, unpacking and format reading.

Run from the repository root with this config (or equivalent CLI fields):

source_factory: examples.dataset.sources.npy_video_source:ExampleSource
source:
  manifest: /shared/raw/example/source.json
  archive_uri: https://publisher.example/cleared-raw-source.tar
  archive_sha256: <verified 64-character SHA256>
output: /shared/processed/example
repo_id: lerobot/example
runtime:
  run_uri: /shared/runs/example
  workers: 16
  batch_size: 1

`lerobot-convert --config_path=convert.yaml` handles the rest. Remove the
archive fields for local inputs. Replace the reader below with MCAP/HDF5/etc.
logic; preserve original action/state units and native annotations in frames.
Upload is a separate explicitly enabled publication step, not an automatic
side effect of acquiring a source. Publisher permission/license checks precede it.
"""

from lerobot.data_processing.sources.npy_video import ArrayVideoSource


class ExampleSource(ArrayVideoSource):
    """Override acquire/open/discover/frames here for a new source format."""
