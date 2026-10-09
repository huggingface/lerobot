# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Small, real offline-processing acceptance workloads (never submits HF Jobs).

Run with the dataset/annotations extras. Slurm mode really submits arrays, so
select a cluster-visible root, Python and partition. Model inference is real;
the exported encoder is deliberately a tiny seeded test model, not a useful
pretrained representation. Annotation uses an operator-owned real VLM endpoint.
Reports separate total wall time from stage telemetry. Fixtures are not quality
evidence for natural videos, nor are smoke timings a production scaling result.
"""

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.data_processing.artifacts import ArtifactStore, file_checksum
from lerobot.data_processing.configs import RuntimeConfig, StageConfig
from lerobot.data_processing.pipeline import run_pipeline
from lerobot.data_processing.types import DatasetRef, InputItem, fingerprint

REAL_REPO = "lerobot/svla_so100_pickplace"
REAL_REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def write_report(root, name, report):
    path = root / "reports" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    print(
        json.dumps({"report": str(path), **{k: v for k, v in report.items() if k != "atoms"}}, default=str),
        flush=True,
    )


def prepare_fixture(root):
    import av

    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    episodes = []
    for episode in range(4):
        actions = np.arange(24 * 6, dtype=np.float32).reshape(24, 6) / 100 + episode
        np.save(raw / f"{episode}-action.npy", actions)
        np.save(raw / f"{episode}-state.npy", actions + 0.01)
        videos = {}
        for camera in ("front", "wrist"):
            relative = f"{episode}-{camera}.mp4"
            videos[f"observation.images.{camera}"] = relative
            with av.open(raw / relative, "w") as container:
                stream = container.add_stream("libx264", rate=12)
                stream.width, stream.height, stream.pix_fmt = 192, 128, "yuv420p"
                for index in range(24):
                    image = np.full((128, 192, 3), 45 + episode * 20, np.uint8)
                    image[25:65, 20 + index : 60 + index, episode % 3] = 210
                    if episode == 0 and index == 0:
                        image[:] = 0
                    for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        episodes.append(
            {
                "id": str(episode),
                "task": "move the colored block",
                "arrays": {"action": f"{episode}-action.npy", "observation.state": f"{episode}-state.npy"},
                "videos": videos,
            }
        )
    manifest = raw / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "fps": 12,
                "robot_type": "acceptance_fixture",
                "features": {
                    name: {"dtype": "float32", "shape": [6], "names": None}
                    for name in ("action", "observation.state")
                },
                "episodes": episodes,
            }
        )
    )
    return manifest


def runtime(args, name, *, gpu=False):
    cfg = RuntimeConfig(
        backend=args.backend,
        workers=args.workers,
        batch_size=1,
        shard_size=1,
        max_retries=1,
        run_uri=str(args.root / "runs" / name),
    )
    cfg.slurm.partition = args.gpu_partition if gpu else args.cpu_partition
    cfg.slurm.account = args.account
    cfg.slurm.time_limit = "00:20:00"
    cfg.slurm.script_dir = args.root / "slurm"
    cfg.slurm.working_directory = Path.cwd()
    cfg.slurm.python_executable = sys.executable
    cfg.slurm.max_concurrent = args.workers
    return cfg


def convert(args, *, real=False):
    from lerobot.data_processing.conversion import ConvertConfig, convert_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    name = f"{'real' if real else 'fixture'}-{args.backend}-{args.workers}"
    source = {"manifest": str(args.root / "raw/source.json")}
    factory = "lerobot.data_processing.sources.npy_video:ArrayVideoSource"
    if real:
        from huggingface_hub import snapshot_download

        source_root = args.root / "original"
        snapshot_download(
            REAL_REPO,
            repo_type="dataset",
            revision=REAL_REVISION,
            local_dir=source_root,
            allow_patterns=["data/**", "meta/**", "videos/**", "README.md", "LICENSE*"],
        )
        source = {
            "root": str(source_root),
            "repo_id": REAL_REPO,
            "episode_indices": list(range(args.episodes)),
        }
        factory = "lerobot.data_processing.sources.lerobot:LeRobotSource"
    cfg = ConvertConfig(
        source_factory=factory, source=source, output=args.root / name, size=512, runtime=runtime(args, name)
    )
    start = time.perf_counter()
    output = convert_dataset(cfg)
    seconds = time.perf_counter() - start
    dataset = LeRobotDataset(cfg.repo_id, root=output, video_backend="pyav")
    first = dataset[0]
    cameras = [key for key in dataset.features if key.startswith("observation.images.")]
    assert all(tuple(first[camera].shape) == (3, 512, 512) for camera in cameras)
    if not real:
        np.testing.assert_array_equal(first["action"].numpy(), np.arange(6, dtype=np.float32) / 100)
        np.testing.assert_array_equal(
            first["observation.state"].numpy(), np.arange(6, dtype=np.float32) / 100 + 0.01
        )
        assert len(dataset) == 96 and dataset.num_episodes == 4
    videos = {}
    for path in sorted((output / "videos").rglob("*.mp4")):
        data = path.read_bytes()
        assert data.index(b"moov") < data.index(b"mdat")
        videos[str(path.relative_to(output))] = file_checksum(path)
    store = ArtifactStore(cfg.runtime.run_uri)
    parts = {
        row["item_id"]: [(a["name"], a["sha256"]) for a in row["artifacts"]]
        for path in store.list("stages/*/attempts/*/*/checkpoints/*.json")
        for row in store.read_json(path)["results"]
    }
    report = {
        "wall_seconds": seconds,
        "frames": len(dataset),
        "episodes": dataset.num_episodes,
        "fps": dataset.fps,
        "videos": videos,
        "parts": parts,
        "stage_metrics": [store.read_json(p) for p in store.list("metrics/*/stages/*.json")],
        "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
    }
    write_report(args.root, name, report)


class ClipSource:
    def __init__(self, root):
        self.root = root
        self.videos = sorted((root / "raw").glob("*.mp4"))
        self.dataset_ref = DatasetRef(
            "acceptance/fixture", fingerprint([(path.name, file_checksum(path)) for path in self.videos])
        )

    def discover(self, stage, store, upstream):
        for path in self.videos:
            episode, camera = path.stem.split("-")
            payload = {
                "video_path": str(path),
                "expected_frames": 24,
                "video_sha256": file_checksum(path)[0],
                "episode_index": int(episode),
                "camera": camera,
                "frame_indices": [0, 6, 12, 18, 23],
            }
            identity = {key: value for key, value in payload.items() if key != "video_path"}
            yield InputItem(
                path.stem,
                payload,
                identity_payload=identity,
                camera_seconds=2,
                physical_seconds=2 if camera == "front" else 0,
            )


def prepare_encoder(root):
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)

        def forward(self, image):
            return torch.relu(self.conv(image)).mean(dim=(2, 3))

    path = root / "encoder.pt2"
    if not path.exists():
        torch.manual_seed(1729)
        exported = torch.export.export(
            Encoder().eval(),
            (torch.zeros(5, 3, 32, 32),),
            dynamic_shapes={"image": {0: torch.export.Dim("frames", min=1, max=64)}},
        )
        torch.export.save(exported, path)
    return path


def checks(args):
    name = f"checks-{args.backend}-{args.workers}-{'cuda' if args.gpu else 'cpu'}"
    checkpoint = prepare_encoder(args.root)
    stages = [
        StageConfig(
            "quality", "lerobot.data_processing.modules.video_quality:VideoQuality", {"sample_every": 1}
        ),
        StageConfig(
            "embeddings",
            "lerobot.data_processing.modules.video_embeddings:VideoEmbeddings",
            {
                "checkpoint_uri": str(checkpoint),
                "checkpoint_sha256": file_checksum(checkpoint)[0],
                "embedding_dim": 8,
                "trusted_checkpoint": True,
                "image_size": 32,
                "device": "cuda" if args.gpu else "cpu",
                "max_batch_frames": 64,
            },
        ),
    ]
    cfg = runtime(args, name, gpu=args.gpu)
    source = ClipSource(args.root)
    start = time.perf_counter()
    # CPU stage needs the CPU partition even when the second stage requests GPU.
    cfg.slurm.partition = args.cpu_partition
    store, quality = run_pipeline(source, stages[:1], cfg)
    cfg.slurm.partition = args.gpu_partition if args.gpu else args.cpu_partition
    _, embeddings = run_pipeline(source, stages[1:], cfg)
    elapsed = time.perf_counter() - start
    before = set(store.list("stages/*/attempts/*/*/checkpoints/*.json"))
    resume_start = time.perf_counter()
    _, resumed = run_pipeline(source, stages[1:], cfg)
    resume_seconds = time.perf_counter() - resume_start
    assert set(store.list("stages/*/attempts/*/*/checkpoints/*.json")) == before
    assert resumed["embeddings"][1] == embeddings["embeddings"][1]
    rows = {}
    for stage, (_plan, summary) in {**quality, **embeddings}.items():
        assert summary.completed == 8
        rows[stage] = []
        with store.open(summary.accepted_path) as stream:
            accepted = pq.read_table(stream).to_pylist()
        for result in accepted:
            for artifact in json.loads(result["artifacts"]):
                with store.open(artifact["path"]) as stream:
                    rows[stage].extend(pq.read_table(stream).to_pylist())
    assert all(row["decoded_frames"] == 24 for row in rows["quality"])
    assert sum(row["black_frames"] for row in rows["quality"]) == 2
    assert len(rows["embeddings"]) == 40
    write_report(
        args.root,
        name,
        {
            "wall_seconds": elapsed,
            "resume_seconds": resume_seconds,
            "rows": rows,
            "summaries": {s: asdict(v[1]) for s, v in {**quality, **embeddings}.items()},
            "metrics": [store.read_json(p) for p in store.list("metrics/*/stages/*.json")],
        },
    )


def annotate(args):
    from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig
    from lerobot.scripts.lerobot_annotate import _run_annotation

    root = args.dataset
    if root is None:
        raise ValueError("--dataset must be a writable copy, not an immutable Hub snapshot")
    name = f"annotation-{args.backend}-{args.workers}"
    before = {str(p.relative_to(root)): file_checksum(p) for p in (root / "videos").rglob("*.mp4")}
    native = {
        str(p.relative_to(root)): pq.read_table(p).select(["index", "action", "observation.state"])
        for p in (root / "data").rglob("*.parquet")
    }
    cfg = AnnotationPipelineConfig(
        root=root,
        repo_id=REAL_REPO,
        only_episodes=tuple(range(args.episodes)),
        video_backend="pyav",
        runtime=runtime(args, name),
    )
    cfg.plan.n_task_rephrasings = 2
    cfg.interjections.max_interjections_per_episode = 1
    cfg.vqa.vqa_emission_hz = 0.1
    cfg.vqa.question_types = ("bbox", "count")
    cfg.vlm.model_id, cfg.vlm.model_revision = MODEL, MODEL_REVISION
    cfg.vlm.auto_serve, cfg.vlm.api_base = False, args.endpoint
    cfg.vlm.api_key_env = args.api_key_env
    cfg.vlm.max_new_tokens, cfg.vlm.temperature, cfg.vlm.client_concurrency = 1024, 0.0, 2
    start = time.perf_counter()
    summary = _run_annotation(cfg, root)
    elapsed = time.perf_counter() - start
    assert before == {str(p.relative_to(root)): file_checksum(p) for p in (root / "videos").rglob("*.mp4")}
    for relative, table in native.items():
        assert table.equals(pq.read_table(root / relative).select(table.column_names))
    atoms = []
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        for row in table.to_pylist():
            for column in ("language_events", "language_persistent"):
                for atom in row.get(column) or []:
                    atoms.append(
                        {
                            "episode_index": row["episode_index"],
                            "frame_index": row["frame_index"],
                            "column": column,
                            **atom,
                        }
                    )
    # Persistent recipes repeat the full list on every frame. Keep one example
    # frame per semantic atom rather than duplicating megabytes in the report.
    unique = list(
        {fingerprint({k: v for k, v in atom.items() if k != "frame_index"}): atom for atom in atoms}.values()
    )
    styles = {}
    for atom in atoms:
        style = atom.get("style") or "speech"
        styles[style] = styles.get(style, 0) + 1
    assert styles.get("subtask", 0) > 0
    write_report(
        args.root,
        name,
        {
            "wall_seconds": elapsed,
            "model": MODEL,
            "revision": MODEL_REVISION,
            "styles_frame_rows": styles,
            "atoms": unique,
            "written_paths": [str(p) for p in summary.written_paths],
            "validation": summary.validation_report.summary(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "convert", "real", "checks", "annotate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--backend", choices=("local", "slurm"), default="local")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu-partition", default="hopper-cpu")
    parser.add_argument("--gpu-partition", default="hopper-dev")
    parser.add_argument("--account", default="huggingface")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": sys.version,
                "cpu_affinity": len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count(),
                "backend": args.backend,
                "action": args.action,
            }
        ),
        flush=True,
    )
    if args.action == "prepare":
        prepare_fixture(args.root)
    elif args.action in {"convert", "real"}:
        convert(args, real=args.action == "real")
    elif args.action == "checks":
        checks(args)
    else:
        annotate(args)


if __name__ == "__main__":
    main()
