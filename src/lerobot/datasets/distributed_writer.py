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

"""Single-file task packing with soft size targets and durable continuation tasks."""

import os
import shutil
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace

import pandas as pd

from lerobot.utils.utils import unflatten_dict

from .dataset_metadata import LeRobotDatasetMetadata
from .dataset_writer import DatasetWriter
from .distributed import (
    DistributedEpisodeResult,
    DistributedShardSpec,
    DistributedWriteSession,
    artifact_digest,
    commit_receipt,
)
from .io_utils import get_file_size_in_mb
from .video_utils import get_video_info


def write_distributed_shard(
    source_writer: DatasetWriter,
    session: DistributedWriteSession,
    shard: DistributedShardSpec,
    episodes: Iterable[Iterable[dict]],
) -> list[DistributedEpisodeResult]:
    """Write complete episodes until a soft size target or the task budget is reached.

    One task produces one Parquet and one MP4 per video feature. Only episodes
    actually consumed belong to its sealed boundary; pending_shards() exposes
    the untouched suffix for dispatch to any idle worker. No input is prefetched.
    """
    with session.shard_lock(shard):
        shard = session.resolve_shard(shard)
        fixed_boundary = session.shard_boundary_path(shard).is_file()
        try:
            reused = session.reusable_shard_results(shard)
        except (OSError, KeyError, TypeError, ValueError):
            # Never repair or overwrite already committed dataset files.
            if commit_receipt(session.root, session.plan).is_file():
                raise
            reused = None
        if reused is not None:
            shutil.rmtree(session.shard_staging_root(shard), ignore_errors=True)
            return reused
        session.cleanup_shard(shard)
        staging = session.shard_staging_root(shard)
        source_meta = source_writer._meta
        meta = LeRobotDatasetMetadata.create(
            source_meta.repo_id,
            fps=source_meta.fps,
            features=deepcopy(source_meta.features),
            root=staging,
            chunks_size=session.plan.chunks_size,
            data_files_size_in_mb=source_meta.data_files_size_in_mb,
            video_files_size_in_mb=source_meta.video_files_size_in_mb,
        )
        first = shard.episodes[0]
        meta.info.total_episodes = first.episode_index
        meta.info.total_frames = first.dataset_from_index
        meta.tasks = pd.DataFrame(
            {"task_index": session.plan.task_to_index.values()},
            index=pd.Index(session.plan.task_to_index, name="task"),
        ).sort_values("task_index")
        writer = DatasetWriter(
            meta=meta,
            root=staging,
            rgb_encoder=source_writer._rgb_encoder,
            depth_encoder=source_writer._depth_encoder,
            encoder_threads=source_writer._encoder_threads,
            batch_encoding_size=1,
            initial_frames=first.dataset_from_index,
            initial_file_indices=(first.data_chunk_index, first.data_file_index),
            single_file=True,
        )
        results = []
        episode_inputs = iter(episodes)
        try:
            for spec in shard.episodes:
                try:
                    frames = next(episode_inputs)
                except StopIteration as error:
                    raise ValueError(f"Missing input for planned episode {spec.episode_index}.") from error
                for frame in frames:
                    if writer.episode_buffer["size"] >= spec.length:
                        raise ValueError(f"Too many frames for planned episode {spec.episode_index}.")
                    writer.add_frame(dict(frame))
                if writer.episode_buffer["size"] != spec.length:
                    raise ValueError(f"Frame count does not match planned episode {spec.episode_index}.")
                if tuple(writer.episode_buffer["task"]) != spec.tasks:
                    raise ValueError(f"Task sequence does not match planned episode {spec.episode_index}.")
                writer.save_episode(parallel_encoding=False)
                latest = meta.latest_episode
                if latest is None:
                    raise RuntimeError("The shard writer did not produce episode metadata.")
                chunk, file = latest["data/chunk_index"][0], latest["data/file_index"][0]
                video_metadata = {k: v[0] for k, v in latest.items() if k.startswith("videos/")}
                stats = unflatten_dict({k: v[0] for k, v in latest.items() if k.startswith("stats/")})[
                    "stats"
                ]
                video_info = {}
                for key in meta.video_keys:
                    video_path = staging / meta.video_path.format(
                        video_key=key,
                        chunk_index=video_metadata[f"videos/{key}/chunk_index"],
                        file_index=video_metadata[f"videos/{key}/file_index"],
                    )
                    video_info[key] = get_video_info(
                        video_path,
                        video_encoder=writer._depth_encoder
                        if key in meta.depth_keys
                        else writer._rgb_encoder,
                    )
                results.append(
                    DistributedEpisodeResult(
                        episode_index=spec.episode_index,
                        dataset_from_index=spec.dataset_from_index,
                        dataset_to_index=spec.dataset_to_index,
                        tasks=spec.tasks,
                        task_indices=spec.task_indices,
                        data_chunk_index=chunk,
                        data_file_index=file,
                        data_path=session.root / meta.data_path.format(chunk_index=chunk, file_index=file),
                        video_metadata=video_metadata,
                        video_info=video_info,
                        stats=stats,
                        depth_units={
                            key: meta.features[key]["info"]["depth_unit"] for key in meta.depth_keys
                        },
                    )
                )
                # Preserve a sealed boundary on retry, even if regenerated file
                # sizes differ. Later tasks may already own the remaining slots.
                if not fixed_boundary:
                    data_path = staging / meta.data_path.format(chunk_index=chunk, file_index=file)
                    reached_target = get_file_size_in_mb(data_path) >= meta.data_files_size_in_mb
                    for key in meta.video_keys:
                        path = staging / meta.video_path.format(
                            video_key=key, chunk_index=chunk, file_index=file
                        )
                        reached_target |= get_file_size_in_mb(path) >= meta.video_files_size_in_mb
                    if reached_target:
                        break
        finally:
            writer.finalize()

        # Publishing this immutable boundary releases only unstarted episodes.
        # It precedes all final file moves and survives a later failed attempt.
        shard = session.record_shard_boundary(shard, len(results))
        # Files stop changing only at this point. Hash each shared artifact once.
        digests: dict[str, str] = {}
        finalized_results = []
        for result in results:
            paths = [str(result.data_path.relative_to(session.root))]
            paths.extend(
                meta.video_path.format(
                    video_key=key,
                    chunk_index=result.video_metadata[f"videos/{key}/chunk_index"],
                    file_index=result.video_metadata[f"videos/{key}/file_index"],
                )
                for key in meta.video_keys
            )
            for path in paths:
                if path not in digests:
                    digests[path] = artifact_digest(staging / path)
            finalized_results.append(replace(result, artifact_digests={p: digests[p] for p in paths}))

        session.require_active()
        for relative_path in digests:
            destination = session.root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"Distributed shard output already exists: {destination}")
            os.replace(staging / relative_path, destination)
        session.record_shard_results(shard, finalized_results)
        shutil.rmtree(staging)
        return finalized_results
