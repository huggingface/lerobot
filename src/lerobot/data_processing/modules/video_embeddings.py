# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Optional persistent PyTorch exported frame encoder; a bounded batch example.

Use only a trusted checkpoint with a known checksum. This extracts embeddings,
not hand actions, rewards or robot targets. Model semantics belong to its owner.
"""

from pathlib import Path

import pyarrow as pa

from ..artifacts import acquire_file, file_checksum
from ..types import ItemResult, ModuleSpec, Outcome, Resources


class VideoEmbeddings:
    def __init__(
        self,
        checkpoint_uri,
        checkpoint_sha256,
        embedding_dim,
        trusted_checkpoint=False,
        device="cpu",
        image_size=224,
        max_batch_frames=128,
    ):
        if trusted_checkpoint is not True:
            raise ValueError("Exported models require trusted_checkpoint=true; a checksum is not trust")
        if (
            embedding_dim < 1
            or image_size < 1
            or max_batch_frames < 1
            or device not in {"cpu", "cuda", "cuda:0"}
        ):
            raise ValueError("Invalid embedding dimension/device/batch limits")
        self.checkpoint_uri, self.checkpoint_sha256 = checkpoint_uri, checkpoint_sha256
        self.dimension, self.device, self.image_size, self.max_frames = (
            embedding_dim,
            device,
            image_size,
            max_batch_frames,
        )
        self.schema = pa.schema(
            [
                ("episode_index", pa.int64()),
                ("frame_index", pa.int64()),
                ("camera", pa.string()),
                ("embedding", pa.list_(pa.float32(), embedding_dim)),
            ]
        )
        self.spec = ModuleSpec(
            "video_embeddings",
            "1",
            "window",
            {"embeddings": self.schema},
            Resources(gpus=int(device.startswith("cuda"))),
        )

    def setup(self, context):
        import torch

        with context.measure("read"):
            path = acquire_file(self.checkpoint_uri, self.checkpoint_sha256, context.scratch / "model.pt2")
        # Exported models can contain executable objects. Only an explicitly
        # trusted, checksum-verified model may reach this loading boundary.
        self.model = torch.export.load(path).module().to(self.device)  # nosec B614

    def teardown(self):
        if hasattr(self, "model"):
            del self.model

    def process_batch(self, items, context):
        import av
        import cv2
        import numpy as np
        import torch

        count = sum(len(item.payload["frame_indices"]) for item in items)
        if not 0 < count <= self.max_frames:
            raise ValueError("Frame batch exceeds memory bound; reduce runtime.batch_size or clip length")
        frames, owners = [], []
        for item in items:
            payload = item.payload
            indices = payload["frame_indices"]
            if (
                not indices
                or not all(type(index) is int for index in indices)
                or indices != sorted(set(indices))
                or indices[0] < 0
            ):
                raise ValueError("Embedding frame indices must be sorted, unique and nonnegative")
            offset = payload.get("video_from_frame", 0)
            if type(offset) is not int or offset < 0:
                raise ValueError("Negative packed-video episode offset")
            video = Path(payload["video_path"])
            with context.measure("read"):
                if file_checksum(video)[0] != payload["video_sha256"]:
                    raise ValueError("Embedding source video changed after planning")
            selected = {index + offset: index for index in indices}
            with context.measure("decode"), av.open(video) as container:
                container.streams.video[0].thread_count = 1
                for index, frame in enumerate(container.decode(video=0)):
                    if index in selected:
                        frames.append(
                            cv2.resize(frame.to_ndarray(format="rgb24"), (self.image_size, self.image_size))
                        )
                        owners.append(
                            (item.item_id, payload["episode_index"], selected[index], payload["camera"])
                        )
                    if index >= max(selected):
                        break
            if sum(owner[0] == item.item_id for owner in owners) != len(indices):
                raise ValueError("Embedding clip exceeds its source video")
        with context.measure("inference"), torch.inference_mode():
            batch = (
                torch.from_numpy(np.stack(frames))
                .permute(0, 3, 1, 2)
                .to(device=self.device, dtype=torch.float32)
                / 255
            )
            values = self.model(batch).to(device="cpu", dtype=torch.float32)
            if tuple(values.shape) != (len(frames), self.dimension) or not torch.isfinite(values).all():
                raise ValueError("Model returned invalid embedding shape or non-finite values")
            values = values.numpy()
        rows = {item.item_id: [] for item in items}
        for owner, embedding in zip(owners, values, strict=True):
            identity, episode, index, camera = owner
            rows[identity].append(
                {
                    "episode_index": episode,
                    "frame_index": index,
                    "camera": camera,
                    "embedding": embedding.tolist(),
                }
            )
        return [
            ItemResult(
                item.item_id,
                Outcome.COMPLETED,
                (
                    context.write_parquet(
                        item, "embeddings", pa.Table.from_pylist(rows[item.item_id], schema=self.schema)
                    ),
                ),
            )
            for item in items
        ]
