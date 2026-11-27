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

"""Direct-write shard writer with worker-local scratch, O(N) deferred video concat, and optional rollover."""

import shutil
import tempfile
from collections.abc import Iterable
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path

from .dataset_metadata import LeRobotDatasetMetadata
from .dataset_writer import DatasetWriter
from .distributed import (
    DistributedEpisodeResult,
    DistributedEpisodeSpec,
    DistributedShardSpec,
    DistributedWriteSession,
    artifact_digest,
)
from .io_utils import get_file_size_in_mb
from .video_utils import get_video_info


def _open_group_writer(
    source_writer: DatasetWriter,
    dataset_root: Path,
    group_scratch: Path,
    first_spec: DistributedEpisodeSpec,
) -> tuple[LeRobotDatasetMetadata, DatasetWriter]:
    """Create an in-memory metadata view and direct-write DatasetWriter backed by local scratch."""
    group_scratch.mkdir(parents=True, exist_ok=True)
    meta = copy(source_writer._meta)
    meta.root = dataset_root
    meta.info = deepcopy(source_writer._meta.info)
    # Clear declared depth_unit on the worker's in-memory view so add_frame() infers
    # the actual input unit from frames and commit can reject unit mismatches on append.
    for depth_key in meta.depth_keys:
        feature_info = meta.features[depth_key].get("info")
        if isinstance(feature_info, dict):
            feature_info.pop("depth_unit", None)
    meta.info.total_episodes = first_spec.episode_index
    meta.info.total_frames = first_spec.dataset_from_index
    meta.episodes = None
    meta.latest_episode = None
    meta._pq_writer = None

    writer = DatasetWriter(
        meta=meta,
        root=dataset_root,
        scratch_root=group_scratch,
        rgb_encoder=source_writer._rgb_encoder,
        depth_encoder=source_writer._depth_encoder,
        encoder_threads=source_writer._encoder_threads,
        batch_encoding_size=1,
        initial_frames=first_spec.dataset_from_index,
        initial_file_indices=(first_spec.data_chunk_index, first_spec.data_file_index),
        single_file=True,
        defer_video_concat=True,
    )
    return meta, writer


def _finalize_group(
    writer: DatasetWriter,
    meta: LeRobotDatasetMetadata,
    session: DistributedWriteSession,
    group_records: list[tuple[DistributedEpisodeSpec, int, int, dict, dict, dict[str, str]]],
    written_files: dict[str, Path],
) -> list[DistributedEpisodeResult]:
    """Close Parquet writer, flush deferred videos directly to root/videos/, and collect video_info."""
    writer.finalize()
    if not group_records:
        return []
    _, chunk, file, first_video_meta, _, _ = group_records[0]
    data_rel = meta.data_path.format(chunk_index=chunk, file_index=file)
    written_files[data_rel] = session.root / data_rel
    video_info: dict[str, dict] = {}
    for key in meta.video_keys:
        video_rel = meta.video_path.format(
            video_key=key,
            chunk_index=first_video_meta[f"videos/{key}/chunk_index"],
            file_index=first_video_meta[f"videos/{key}/file_index"],
        )
        video_path = session.root / video_rel
        written_files[video_rel] = video_path
        video_info[key] = get_video_info(
            video_path,
            video_encoder=writer._depth_encoder if key in meta.depth_keys else writer._rgb_encoder,
        )
    return [
        DistributedEpisodeResult(
            episode_index=spec.episode_index,
            dataset_from_index=spec.dataset_from_index,
            dataset_to_index=spec.dataset_to_index,
            tasks=spec.tasks,
            task_indices=spec.task_indices,
            data_chunk_index=ep_chunk,
            data_file_index=ep_file,
            data_path=session.root / meta.data_path.format(chunk_index=ep_chunk, file_index=ep_file),
            video_metadata=video_metadata,
            video_info=video_info,
            stats=stats,
            depth_units=depth_units,
        )
        for spec, ep_chunk, ep_file, video_metadata, stats, depth_units in group_records
    ]


def write_distributed_shard(
    source_writer: DatasetWriter,
    session: DistributedWriteSession,
    shard: DistributedShardSpec,
    episodes: Iterable[Iterable[dict]],
    *,
    local_staging_dir: Path | None = None,
    parallel_encoding: bool = True,
) -> list[DistributedEpisodeResult]:
    """Write complete episodes directly to final Parquet/MP4 slots using worker-local scratch.

    Because unreferenced data and video files remain invisible until global metadata
    commit, Parquet and final MP4 outputs are written directly to ``session.root/data``
    and ``session.root/videos`` without shared-filesystem staging, file locks, or
    file renames. All transient frame images and intermediate per-episode video
    segments live strictly in worker-local scratch and are always cleaned up.
    """
    shard = session.resolve_shard(shard)
    fixed_boundary = session.shard_boundary_path(shard).is_file()
    try:
        reused = session.reusable_shard_results(shard)
    except (OSError, KeyError, TypeError, ValueError):
        reused = None
    if reused is not None:
        return reused
    session.cleanup_shard(shard)

    scratch_parent = Path(local_staging_dir) if local_staging_dir is not None else None
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(tempfile.mkdtemp(prefix=f"lerobot-shard-{shard.shard_index:06d}-", dir=scratch_parent))

    rollover_in_shard = session.plan.rollover_in_shard
    results: list[DistributedEpisodeResult] = []
    written_files: dict[str, Path] = {}
    episode_inputs = iter(episodes)

    active_meta: LeRobotDatasetMetadata | None = None
    active_writer: DatasetWriter | None = None
    active_records: list[tuple[DistributedEpisodeSpec, int, int, dict, dict, dict[str, str]]] = []
    group_index = 0
    result_recorded = False

    try:
        for idx, spec in enumerate(shard.episodes):
            if active_writer is None:
                group_scratch = scratch_dir / f"group-{group_index:04d}" if rollover_in_shard else scratch_dir
                group_index += 1
                active_meta, active_writer = _open_group_writer(
                    source_writer, session.root, group_scratch, spec
                )
                active_records = []
                data_rel = active_meta.data_path.format(
                    chunk_index=spec.data_chunk_index, file_index=spec.data_file_index
                )
                data_dst = session.root / data_rel
                if data_dst.exists():
                    raise FileExistsError(f"Distributed shard output already exists: {data_dst}")
                for key in active_meta.video_keys:
                    video_dst = session.root / active_meta.video_path.format(
                        video_key=key,
                        chunk_index=spec.data_chunk_index,
                        file_index=spec.data_file_index,
                    )
                    if video_dst.exists():
                        raise FileExistsError(f"Distributed shard output already exists: {video_dst}")

            try:
                frames = next(episode_inputs)
            except StopIteration as error:
                raise ValueError(f"Missing input for planned episode {spec.episode_index}.") from error
            active_writer.episode_buffer = active_writer._create_episode_buffer(spec.episode_index)
            for frame in frames:
                if active_writer.episode_buffer["size"] >= spec.length:
                    raise ValueError(f"Too many frames for planned episode {spec.episode_index}.")
                active_writer.add_frame(dict(frame))

            ep_metadata, ep_stats = active_writer.save_episode_for_shard(
                spec, parallel_encoding=parallel_encoding
            )
            chunk = ep_metadata["data/chunk_index"]
            file = ep_metadata["data/file_index"]
            data_rel = active_meta.data_path.format(chunk_index=chunk, file_index=file)
            data_path = session.root / data_rel
            written_files[data_rel] = data_path
            video_metadata = {k: v for k, v in ep_metadata.items() if k.startswith("videos/")}
            depth_units = {
                key: active_meta.features[key]["info"]["depth_unit"]
                for key in active_meta.depth_keys
                if (active_meta.features[key].get("info") or {}).get("depth_unit") is not None
            }
            active_records.append((spec, chunk, file, video_metadata, ep_stats, depth_units))

            reached_target = get_file_size_in_mb(data_path) >= active_meta.data_files_size_in_mb
            for key in active_meta.video_keys:
                reached_target |= (
                    active_writer.get_current_video_size_in_mb(key, chunk, file)
                    >= active_meta.video_files_size_in_mb
                )

            if reached_target:
                if rollover_in_shard:
                    if idx + 1 < len(shard.episodes):
                        results.extend(
                            _finalize_group(
                                active_writer,
                                active_meta,
                                session,
                                active_records,
                                written_files,
                            )
                        )
                        active_writer = None
                        active_meta = None
                        active_records = []
                elif not fixed_boundary:
                    # Preserve a sealed boundary on retry, even if regenerated file
                    # sizes differ. Later tasks may already own the remaining slots.
                    break

        # Close the last Parquet writer before sealing the immutable task boundary.
        if active_writer is not None:
            active_writer.close_writer()

        # Seal the exact completed episode prefix (no-op for rollover/single-episode shards).
        completed_count = len(results) + len(active_records)
        shard = session.record_shard_boundary(shard, completed_count)

        if active_writer is not None and active_meta is not None:
            results.extend(
                _finalize_group(
                    active_writer,
                    active_meta,
                    session,
                    active_records,
                    written_files,
                )
            )
            active_writer = None

        # Files stop changing at this point. Hash each final artifact once.
        digests: dict[str, str] = {
            relative_path: artifact_digest(file_path) for relative_path, file_path in written_files.items()
        }
        source_meta = source_writer._meta
        finalized_results = []
        for result in results:
            paths = [str(result.data_path.relative_to(session.root))]
            paths.extend(
                source_meta.video_path.format(
                    video_key=key,
                    chunk_index=result.video_metadata[f"videos/{key}/chunk_index"],
                    file_index=result.video_metadata[f"videos/{key}/file_index"],
                )
                for key in source_meta.video_keys
            )
            finalized_results.append(replace(result, artifact_digests={p: digests[p] for p in paths}))

        session.require_active()
        session.record_shard_results(shard, finalized_results)
        result_recorded = True
        return finalized_results
    except BaseException:
        if active_writer is not None:
            active_writer.close_writer()
        if not result_recorded:
            for path in written_files.values():
                path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
