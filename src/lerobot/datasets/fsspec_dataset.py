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
"""Stream a LeRobot v3.0 dataset from an fsspec object store (S3, GCS, Volcengine TOS…).

:class:`StreamingLeRobotDataset` streams from the HF Hub or a local dir. This subclass
points it at any ``fsspec`` URL instead, with no per-provider code:

1. **metadata** — the small ``meta/`` tree is mirrored to a local dir via fsspec and handed
   to the parent as ``root``, so the stock metadata path runs unchanged (a few MB; ``data/``
   and ``videos/`` are never downloaded).
2. **low-dim data** — overrides the parent's ``_load_hf_dataset`` seam:
   ``load_dataset("parquet", data_files="<url>/data/*/*.parquet", storage_options=…,
   streaming=True)`` streams the parquet shards over fsspec. Also applies the ``episodes``
   filter (useful for held-out train/eval splits).
3. **video** — lerobot's decoder already opens each mp4 with ``fsspec.open(...)`` and lets
   torchcodec range-read it, so overriding the parent's ``_get_video_path`` seam to return
   the full ``<url>/…mp4`` fsspec URL is enough; the depth/rgb decode logic stays shared.
   ``storage_options`` are registered as protocol defaults (``fsspec.config.conf``) so that
   bare ``fsspec.open`` can authenticate.

Everything else — the shuffle buffer, sharding, delta-timestamp windows, per-item
construction — is inherited from the parent constructor untouched.

You build the fsspec URL yourself and pass credentials via ``storage_options`` (read them
from the environment; never hardcode secrets). For Volcengine TOS install ``tosfs`` to get
the ``tos://`` protocol.

Example::

    import os

    ds = FsspecLeRobotDataset(
        "tos://my-bucket/lerobot-datasets/finish_sandwich",
        storage_options={
            "key": os.environ["TOS_ACCESS_KEY"],
            "secret": os.environ["TOS_SECRET_KEY"],
            "endpoint": "https://tos-cn-beijing.volces.com",
            "region": "cn-beijing",
        },
        episodes=[0, 3, 17],  # optional held-out subset
    )
    for item in ds:  # IterableDataset — iterate, no ds[i]
        item["observation.images.front"]  # (C, H, W); also item["observation.state"], ["action"]
        break
"""

from __future__ import annotations

import os
import tempfile

import datasets
import fsspec
from datasets import load_dataset

from .streaming_dataset import StreamingLeRobotDataset


class FsspecLeRobotDataset(StreamingLeRobotDataset):
    """A :class:`StreamingLeRobotDataset` that reads a v3.0 dataset from any fsspec URL."""

    def __init__(
        self,
        url: str,
        repo_id: str | None = None,
        *,
        storage_options: dict | None = None,
        meta_cache_dir: str | None = None,
        **kwargs,
    ):
        """
        Args:
            url: dataset root on the backend, e.g. ``tos://bucket/prefix`` or ``s3://bucket/prefix``.
            repo_id: optional label only (metadata is read from the mirrored ``meta/``, never
                the Hub). Defaults to the last path segment of ``url``.
            storage_options: fsspec kwargs for the backend (TOS: ``key``/``secret``/``endpoint``/
                ``region``). Also registered as the protocol default so the video decoder's bare
                ``fsspec.open`` authenticates.
            meta_cache_dir: where to mirror ``meta/`` (default: a temp dir).
            **kwargs: forwarded to :class:`StreamingLeRobotDataset` (``episodes``,
                ``delta_timestamps``, ``image_transforms``, ``tolerance_s``, ``buffer_size``,
                ``max_num_shards``, ``seed``, ``shuffle``, ``return_uint8``, …).
        """
        self._url = url.rstrip("/")
        self._protocol, self._rpath = fsspec.core.split_protocol(self._url)
        self._rpath = (self._rpath or "").rstrip("/")
        self.storage_options = dict(storage_options or {})
        # instance-cached by fsspec, so this is the same object load_dataset/fsspec.open resolve.
        self._fs = fsspec.filesystem(self._protocol, **self.storage_options)

        # Make the credentials the default for this protocol, so the video decoder's bare
        # ``fsspec.open("<url>/…mp4")`` (which passes no storage_options) can authenticate.
        if self._protocol and self.storage_options:
            fsspec.config.conf.setdefault(self._protocol, {}).update(self.storage_options)

        # repo_id is only a label (metadata comes from the mirrored meta/); derive from the URL.
        repo_id = repo_id or (self._rpath.rsplit("/", 1)[-1] or "dataset")
        # Mirror meta/ locally and hand it to the parent as `root`: the stock metadata,
        # version-check, delta-timestamp and shuffle/shard setup then run unchanged. The
        # parent's data loading goes through the `_load_hf_dataset` seam overridden below.
        meta_root = self._mirror_meta(meta_cache_dir, repo_id)
        super().__init__(repo_id, root=meta_root, **kwargs)

    # ---- metadata mirror -------------------------------------------------
    def _mirror_meta(self, cache_dir: str | None, repo_id: str) -> str:
        local = cache_dir or tempfile.mkdtemp(prefix="fsspec_lerobot_")
        dst = os.path.join(local, repo_id.replace("/", "__"))
        meta_dst = os.path.join(dst, "meta")
        if not os.path.exists(os.path.join(meta_dst, "info.json")):
            os.makedirs(dst, exist_ok=True)
            # copy the small remote meta/ tree (info.json, stats.json, tasks.parquet,
            # episodes/chunk-*/file-*.parquet) — not data/ or videos/.
            self._fs.get(f"{self._rpath}/meta", meta_dst, recursive=True)
        if not os.path.exists(os.path.join(meta_dst, "info.json")):
            raise FileNotFoundError(
                f"no meta/info.json under {self._url}/meta — is this a LeRobot v3.0 dataset on the backend?"
            )
        return dst

    # ---- low-dim data: stream the parquet shards off the backend ---------
    def _load_hf_dataset(self) -> datasets.IterableDataset:
        ds = load_dataset(
            "parquet",
            data_files=f"{self._url}/data/*/*.parquet",
            storage_options=self.storage_options,
            split="train",
            streaming=True,
        )
        if self.episodes is not None:
            keep = {int(e) for e in self.episodes}
            # the parent ignores `episodes` when streaming; apply a lazy per-frame filter here.
            ds = ds.filter(lambda x: int(x["episode_index"]) in keep)
        return ds

    # ---- video: decoded straight off fsspec (no download) ----------------
    def _get_video_path(self, ep_idx: int, video_key: str) -> str:
        # lerobot's decoder opens this with fsspec.open(...) and lets torchcodec range-read it.
        return f"{self._url}/{self.meta.get_video_file_path(ep_idx, video_key)}"
