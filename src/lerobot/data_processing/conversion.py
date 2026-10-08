# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Parallel native-preserving conversion and deterministic LeRobot assembly."""

import json
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.configs import RGBEncoderConfig, rgb_encoder_defaults
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.language import (
    language_array,
    language_events_arrow_type,
    language_persistent_arrow_type,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .configs import RuntimeConfig, StageConfig
from .pipeline import run_pipeline
from .sources import load_source
from .sources.acquisition import unpack_tar
from .types import ItemResult, ModuleSpec, Outcome, Resources, canonical_json
from .worker import accepted_in_shard

GEOMETRY = pa.schema(
    [
        ("source_episode_id", pa.string()),
        ("camera", pa.string()),
        ("source_width", pa.int64()),
        ("source_height", pa.int64()),
        ("scaled_width", pa.int64()),
        ("scaled_height", pa.int64()),
        ("pad_left", pa.int64()),
        ("pad_top", pa.int64()),
    ]
)


@dataclass
class ConvertConfig:
    source_factory: str = "lerobot.data_processing.sources.npy_video:ArrayVideoSource"
    source: dict = field(default_factory=dict)
    repo_id: str = "lerobot/converted"
    output: Path = Path("converted")
    fps: int | None = None
    size: int = 512
    encoder_threads: int = 1
    encoder: RGBEncoderConfig = field(default_factory=rgb_encoder_defaults)
    runtime: RuntimeConfig = field(default_factory=lambda: RuntimeConfig(batch_size=1))

    def __post_init__(self):
        if (
            self.size < 2
            or self.size % 2
            or self.encoder_threads < 1
            or (self.fps is not None and self.fps < 1)
        ):
            raise ValueError("Conversion requires positive FPS/threads and a positive even image size")


def letterbox(image, size):
    """Resize without distortion; return the exact rounded spatial transform."""
    import cv2
    import numpy as np

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("This converter expects uint8 RGB; depth needs a separate explicit recipe")
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    scaled_width, scaled_height = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - scaled_width) // 2, (size - scaled_height) // 2
    output = np.zeros((size, size, 3), dtype=np.uint8)
    output[top : top + scaled_height, left : left + scaled_width] = cv2.resize(
        image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    return output, {
        "source_width": width,
        "source_height": height,
        "scaled_width": scaled_width,
        "scaled_height": scaled_height,
        "pad_left": left,
        "pad_top": top,
    }


class EpisodeConversion:
    def __init__(self, source_factory, source, fps, size, encoder_threads, encoder):
        self.source = load_source(source_factory, source)
        self.source.open()
        self.fps = fps or self.source.fps
        if self.fps > self.source.fps:
            raise ValueError("Upsampling requires an explicit source recipe; do not invent native actions")
        self.size, self.encoder_threads, self.encoder = size, encoder_threads, RGBEncoderConfig(**encoder)
        self.spec = ModuleSpec(
            "episode_conversion",
            "1",
            "episode",
            {"dataset": None, "geometry": GEOMETRY},
            Resources(cpus=encoder_threads),
        )

    def setup(self, context):
        import cv2

        cv2.setNumThreads(1)

    def teardown(self):
        pass

    def process_batch(self, items, context):
        results = []
        for item in items:
            language_features = {
                key: feature
                for key, feature in self.source.features.items()
                if feature["dtype"] == "language"
            }
            language_rows = {key: [] for key in language_features}
            features = {
                **{
                    key: feature
                    for key, feature in self.source.features.items()
                    if key not in language_features
                },
                "source_frame_index": {"dtype": "int64", "shape": (1,), "names": None},
            }
            if "source_frame_index" in self.source.features:
                raise ValueError("Source uses reserved conversion lineage key")
            for camera in self.source.camera_keys:
                features[camera] = {
                    "dtype": "video",
                    "shape": (self.size, self.size, 3),
                    "names": ["height", "width", "channels"],
                }
            root = context.scratch / (item.item_id + "-" + uuid.uuid4().hex)
            dataset = LeRobotDataset.create(
                "processing/episode",
                self.fps,
                features,
                root=root,
                robot_type=self.source.robot_type,
                use_videos=bool(self.source.camera_keys),
                rgb_encoder=self.encoder,
                encoder_threads=self.encoder_threads,
                video_backend="pyav",
            )
            geometries = {}
            output_count = 0
            try:
                for index, frame in enumerate(self.source.frames(item.payload)):
                    # Integer arithmetic selects the source sample at or before
                    # each output clock, retaining the original index explicitly.
                    if index != (output_count * self.source.fps) // self.fps:
                        continue
                    for camera in self.source.camera_keys:
                        frame[camera], geometry = letterbox(frame[camera], self.size)
                        if camera in geometries and geometries[camera] != geometry:
                            raise ValueError("Camera geometry changed within an episode")
                        geometries[camera] = geometry
                    import numpy as np

                    for key in language_features:
                        atoms = frame.pop(key)
                        for atom in atoms or []:
                            if atom.get("style") in {"vqa", "trace"}:
                                geometry = geometries.get(atom.get("camera"))
                                if geometry and (
                                    geometry["source_width"] != self.size
                                    or geometry["source_height"] != self.size
                                ):
                                    raise ValueError(
                                        "View-dependent native annotations need explicit geometry reprojection in the source recipe"
                                    )
                        language_rows[key].append(atoms)
                    dataset.add_frame({**frame, "source_frame_index": np.array([index], dtype=np.int64)})
                    output_count += 1
                if output_count < 2:
                    raise ValueError("Converted episode has fewer than two frames")
                dataset.save_episode(parallel_encoding=False)
            finally:
                dataset.finalize()
            if language_features:
                data_path = next((root / "data").rglob("*.parquet"))
                table = pq.read_table(data_path).replace_schema_metadata(None)
                for key, values in language_rows.items():
                    dtype = (
                        language_persistent_arrow_type()
                        if key == "language_persistent"
                        else language_events_arrow_type()
                    )
                    table = table.append_column(key, language_array(values, dtype))
                from lerobot.datasets.io_utils import write_table_one_row_group_per_episode

                write_table_one_row_group_per_episode(table, data_path)
                info_path = root / "meta/info.json"
                info = json.loads(info_path.read_text())
                info["features"].update(language_features)
                info_path.write_bytes(canonical_json(info))
            # Complete parts are immutable artifacts. Packing is a later choice,
            # not an input-ID or model-worker dependency.
            archive = context.scratch / f"{item.item_id}.tar"
            with tarfile.open(archive, "w") as stream:
                for path in sorted(root.rglob("*")):
                    if path.is_file() and not path.is_symlink():
                        stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
            artifact = context.write_asset(item, "dataset", archive)
            geometry = context.write_parquet(
                item,
                "geometry",
                pa.Table.from_pylist(
                    [
                        {"source_episode_id": item.key, "camera": camera, **value}
                        for camera, value in sorted(geometries.items())
                    ],
                    schema=GEOMETRY,
                ),
            )
            results.append(ItemResult(item.item_id, Outcome.COMPLETED, (artifact, geometry)))
            archive.unlink()
            shutil.rmtree(root)
        return results


def convert_dataset(cfg: ConvertConfig):
    import draccus

    source = load_source(cfg.source_factory, cfg.source)
    source.acquire()
    cfg.output = cfg.output.resolve()
    if cfg.output.exists():
        raise FileExistsError("Conversion output already exists; choose a new release directory")
    if not cfg.runtime.run_uri:
        cfg.runtime.run_uri = str(cfg.output.parent / (cfg.output.name + ".processing"))
    stage = StageConfig(
        "convert",
        "lerobot.data_processing.conversion:EpisodeConversion",
        {
            "source_factory": cfg.source_factory,
            "source": source.worker_config() if hasattr(source, "worker_config") else cfg.source,
            "fps": cfg.fps,
            "size": cfg.size,
            "encoder_threads": cfg.encoder_threads,
            "encoder": draccus.encode(cfg.encoder),
        },
    )
    store, completed = run_pipeline(source, [stage], cfg.runtime)
    if cfg.runtime.mode == "plan":
        return None
    plan, summary = completed["convert"]
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="conversion-release-", dir=cfg.output.parent) as temporary_dir:
        directory = Path(temporary_dir)
        roots: list[Path] = []
        lineage, geometry_rows = [], []
        for shard in range(plan.shards):
            accepted = accepted_in_shard(store, plan, shard)
            for item in plan.read_shard(store, shard):
                result = accepted[item.item_id]
                if result.outcome != Outcome.COMPLETED:
                    raise ValueError("This conversion recipe does not define dropping/masking episodes")
                archive = directory / "part.tar"
                part = directory / "parts" / item.item_id
                for artifact in result.artifacts:
                    if artifact.name == "dataset":
                        with store.open(artifact.path) as input_stream, archive.open("wb") as output_stream:
                            shutil.copyfileobj(input_stream, output_stream)
                        from .artifacts import file_checksum

                        if file_checksum(archive) != (artifact.sha256, artifact.size):
                            raise ValueError("Accepted conversion part changed during assembly")
                        unpack_tar(archive, part)
                    else:
                        with store.open(artifact.path) as stream:
                            geometry_rows.extend(pq.read_table(stream).to_pylist())
                lineage.append(
                    {"source_episode_id": item.key, "episode_index": len(roots), "item_id": item.item_id}
                )
                roots.append(part)
        release = directory / "release"
        aggregate_datasets(
            ["processing/episode"] * len(roots),
            cfg.repo_id,
            roots,
            release,
            concatenate_videos=False,
            concatenate_data=False,
        )
        pq.write_table(pa.Table.from_pylist(lineage), release / "meta/source_episode_map.parquet")
        pq.write_table(
            pa.Table.from_pylist(geometry_rows, schema=GEOMETRY), release / "meta/camera_geometry.parquet"
        )
        for relative, native_path in getattr(source, "metadata_files", {}).items():
            target = release / relative
            if (
                not target.resolve().is_relative_to(release.resolve())
                or target.exists()
                or not relative.startswith("meta/")
            ):
                raise ValueError("Source metadata would overwrite generated metadata or escape the release")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(native_path, target)
        (release / "meta/processing.json").write_bytes(
            canonical_json(
                {
                    "input": {"uri": source.dataset_ref.uri, "revision": source.dataset_ref.revision},
                    "source_recipe": cfg.source_factory,
                    "plan_id": plan.plan_id,
                    "accepted": summary.accepted_path,
                    "steps": {
                        "resize": "letterbox",
                        "size": cfg.size,
                        "fps": cfg.fps or source.fps,
                        "native_actions": "unchanged",
                        "sampling": "source frame at or before output clock",
                        "encoder": draccus.encode(cfg.encoder),
                        "faststart": True,
                    },
                }
            )
        )
        # One publication point only after metadata and ordinary loading validate.
        loaded = LeRobotDataset(cfg.repo_id, root=release, video_backend="pyav")
        if loaded.num_episodes != len(roots) or len(loaded) < 2 * len(roots):
            raise ValueError("Assembled dataset failed standard-reader validation")
        release.rename(cfg.output)
    return cfg.output
