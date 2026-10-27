# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded demonstration reads using LeRobot's MP4 range and synthesis primitives."""

import io
import math
from collections import OrderedDict
from pathlib import Path

from lerobot.streaming.location import LocationKind, StorageLocation, hf_dataset_uri
from lerobot.streaming.manifest import EpisodeVideoManifest, VideoFileRecord
from lerobot.streaming.mp4 import fetch_mp4_index, synthesize_mp4, synthesized_mp4_size
from lerobot.streaming.range_fetch import NativeHTTPRangeFetcher
from lerobot.streaming.sidecar_utils import SidecarSpec

from .modeling_zero_wam import read_video_at_fps


def canonical_prompt_video(video: str) -> str:
    """Keep immutable Hub URLs as URLs and resolve local paths normally."""
    location = StorageLocation.parse(video)
    if location.kind is LocationKind.LOCAL:
        return str(location.local_path.resolve())
    if location.kind is not LocationKind.HF_DATASET:
        raise ValueError("Prompt videos must be local files or hf://datasets URLs")
    location = location.pinned()
    if not location.path_in_repo:
        raise ValueError("A prompt URL must identify a video file")
    return location.uri


def prompt_source_identity(video: str) -> dict:
    """Use an immutable commit for remote cache identity, or local size and modification time."""
    video = canonical_prompt_video(video)
    location = StorageLocation.parse(video)
    if location.is_remote:
        return {"video": video, "revision": location.pinned_commit}
    source = Path(video).stat()
    return {"video": video, "size": source.st_size, "mtime_ns": source.st_mtime_ns}


class PromptVideoReader:
    """Fetch one keyframe-aligned prompt interval, without localizing its source container.

    A supplied training sidecar reuses the full-source index. Otherwise at most eight parsed file
    indexes are retained. A byte limit bounds each synthesized compressed clip before payload I/O.
    Local prompts continue through the ordinary bounded decoder.
    """

    def __init__(
        self,
        videos: list[str],
        *,
        sidecar_path: Path | None = None,
        max_video_bytes: int = 512 * 1024**2,
    ):
        if max_video_bytes <= 0:
            raise ValueError("max_video_bytes must be positive")
        self.max_video_bytes = max_video_bytes
        self._fetchers: dict[str, NativeHTTPRangeFetcher] = {}
        self._indexes: OrderedDict[str, VideoFileRecord] = OrderedDict()
        self._sidecar_root: str | None = None
        self._sidecar_records: dict[str, VideoFileRecord] = {}
        remote = [canonical_prompt_video(video) for video in videos if StorageLocation.parse(video).is_remote]
        if sidecar_path is not None and remote:
            spec = SidecarSpec.from_dict(EpisodeVideoManifest.load_file_sidecar_metadata(sidecar_path))
            root = spec.data_root.rstrip("/")
            if StorageLocation.parse(root).pinned_commit is None or any(
                not video.startswith(root + "/") for video in remote
            ):
                raise ValueError("Prompt URLs and MP4 sidecar must use the same immutable source")
            if not EpisodeVideoManifest.validate_file_sidecar(sidecar_path, spec):
                raise ValueError("Invalid prompt MP4 sidecar")
            paths = [video[len(root) + 1 :] for video in remote]
            self._sidecar_records = EpisodeVideoManifest.load_file_sidecar(sidecar_path, file_paths=paths)
            if set(paths) - self._sidecar_records.keys():
                raise ValueError("MP4 sidecar is missing a prompt video")
            self._sidecar_root = root

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        for fetcher in self._fetchers.values():
            fetcher.close()
        self._fetchers.clear()
        self._indexes.clear()
        self._sidecar_records.clear()

    def read(
        self,
        video: str,
        fps: float,
        *,
        start_time_s: float = 0,
        end_time_s: float | None = None,
        max_frames: int | None = None,
    ):
        video = canonical_prompt_video(video)
        location = StorageLocation.parse(video)
        if not location.is_remote:
            return read_video_at_fps(
                video, fps, start_time_s=start_time_s, end_time_s=end_time_s, max_frames=max_frames
            )
        if (
            not math.isfinite(start_time_s)
            or start_time_s < 0
            or end_time_s is None
            or not math.isfinite(end_time_s)
            or end_time_s <= start_time_s
        ):
            raise ValueError("Remote prompts require finite start/end times with 0 <= start < end")
        if not math.isfinite(fps) or fps <= 0 or (max_frames is not None and max_frames < 1):
            raise ValueError("Prompt fps and max_frames must be positive")
        assert location.repo_id is not None and location.pinned_commit is not None
        root = self._sidecar_root or hf_dataset_uri(location.repo_id, location.pinned_commit)
        if not video.startswith(root + "/"):
            raise ValueError("Prompt URL does not match the configured MP4 sidecar")
        relative = video[len(root) + 1 :]
        if root not in self._fetchers:
            self._fetchers[root] = NativeHTTPRangeFetcher(root, max_connections=2)
        fetcher = self._fetchers[root]
        record = self._sidecar_records.get(relative) or self._indexes.get(video)
        if record is None:
            size = fetcher.info_size(relative)
            index = fetch_mp4_index(relative, fetcher.read_range, file_size=size)
            record = VideoFileRecord(relative, size, index)
            self._indexes[video] = record
            if len(self._indexes) > 8:
                self._indexes.popitem(last=False)
        elif video in self._indexes:
            self._indexes.move_to_end(video)
        sample = record.mp4.sample_slice(start_time_s, end_time_s, file_size=record.file_size)
        if synthesized_mp4_size(record.mp4, sample) > self.max_video_bytes:
            raise ValueError("Prompt interval exceeds max_video_bytes; shorten it or raise the limit")
        payload = fetcher.read_range(relative, sample.byte_offset, sample.byte_length)
        if len(payload) != sample.byte_length:
            raise OSError("Short read while streaming a prompt video")
        clip = synthesize_mp4(record.mp4, sample, payload)
        return read_video_at_fps(
            io.BytesIO(clip),
            fps,
            start_time_s=max(0.0, start_time_s - sample.source_start_pts),
            end_time_s=end_time_s - sample.source_start_pts,
            max_frames=max_frames,
        )
