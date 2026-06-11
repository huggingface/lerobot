# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Distributed, resumable streaming training on a large HF-hosted dataset.

This example shows how to train (or just stress the data pipeline) over a multi-TB dataset that never
touches local disk, scaling across GPUs and nodes with Accelerate. It demonstrates the large-scale
streaming features of :class:`StreamingLeRobotDataset`:

- per-rank sharding via ``split_dataset_by_node`` (each GPU streams disjoint data; ``rank``/``world_size``
  are auto-resolved from the Accelerate state, so nothing needs to be passed explicitly);
- DataLoader-worker shard splitting (no duplicate frames within a rank);
- resumable streaming via ``dataset.state_dict()`` / ``load_state_dict()`` saved into the checkpoint;
- an explicit video-decoder cache size so the working set of open decoders does not thrash.

Launch with Accelerate (single node, N GPUs):

    accelerate launch --num_processes=8 examples/scaling/train_streaming_multinode.py \
        --repo_id=pepijn223/robocasa_pretrain_human300_v4 --batch_size=64

Multinode runs use the same script under SLURM; see ``slurm/train_streaming_robocasa.sh``.

Pass ``--dummy`` to skip the model entirely and measure pure dataloading throughput.
"""

import argparse
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from lerobot.datasets import LeRobotDatasetMetadata, StreamingLeRobotDataset
from lerobot.utils.constants import ACTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", type=str, default="lerobot/droid_1.0.1")
    parser.add_argument(
        "--root", type=str, default=None, help="Local/prewarmed dataset root (else stream from Hub)."
    )
    parser.add_argument("--output_dir", type=str, default="outputs/train/streaming_multinode")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64, help="Per-process batch size.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--buffer_size", type=int, default=2000, help="Output shuffle-buffer size, in frames."
    )
    parser.add_argument("--video_decoder_cache_size", type=int, default=None)
    parser.add_argument("--n_action_steps", type=int, default=16, help="Action-chunk length (delta horizon).")
    parser.add_argument("--save_freq", type=int, default=200)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint dir to resume from.")
    parser.add_argument("--dummy", action="store_true", help="Skip the model; measure dataloading only.")
    return parser.parse_args()


def make_dataloader(
    args: argparse.Namespace, meta: LeRobotDatasetMetadata
) -> tuple[DataLoader, StreamingLeRobotDataset]:
    # Supervise an action chunk; delta_timestamps drive the SARM-style temporal window.
    delta_timestamps = {ACTION: [t / meta.fps for t in range(args.n_action_steps)]}
    # rank / world_size are resolved automatically from the Accelerate state inside the dataset.
    dataset = StreamingLeRobotDataset(
        args.repo_id,
        root=args.root,
        delta_timestamps=delta_timestamps,
        buffer_size=args.buffer_size,
        video_decoder_cache_size=args.video_decoder_cache_size,
        tolerance_s=1e-3,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    return loader, dataset


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    loader, dataset = make_dataloader(args, meta)

    if args.dummy:
        model = optimizer = None
    else:
        from lerobot.policies.act import ACTConfig, ACTPolicy
        from lerobot.utils.feature_utils import dataset_to_policy_features

        features = dataset_to_policy_features(meta.features)
        output_features = {k: ft for k, ft in features.items() if k == ACTION}
        input_features = {k: ft for k, ft in features.items() if k not in output_features}
        cfg = ACTConfig(input_features=input_features, output_features=output_features)
        model = ACTPolicy(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # Do NOT prepare the dataloader: the dataset is already rank-disjoint via
        # split_dataset_by_node, and accelerate's IterableDatasetShard would keep only every
        # world_size-th batch of it (silently training on 1/N of the data while decoding all
        # of it). Batches are moved to the device manually in the loop.
        model, optimizer = accelerator.prepare(model, optimizer)

    # Resume: restore the dataset's stream position so we don't replay already-seen data. The state holds
    # plain HF stream dicts + RNG state (not tensors), so weights_only=False is required; the file is a
    # checkpoint this script wrote itself.
    if args.resume_from is not None:
        state = torch.load(Path(args.resume_from) / "dataset_state.pt", weights_only=False)  # nosec B614
        dataset.load_state_dict(state)
        accelerator.print(f"Resumed dataset stream from {args.resume_from}")

    step = 0
    frames_seen = 0
    window_start = time.perf_counter()
    done = False
    while not done:
        for batch in loader:
            if model is not None:
                batch = {k: (v.to(accelerator.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                loss, _ = model.forward(batch)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            frames_seen += args.batch_size
            if step % args.log_freq == 0:
                elapsed = time.perf_counter() - window_start
                fps_per_proc = (args.log_freq * args.batch_size) / max(elapsed, 1e-9)
                total_fps = fps_per_proc * accelerator.num_processes
                accelerator.print(
                    f"step {step} | {fps_per_proc:.1f} frames/s/proc | {total_fps:.1f} frames/s total"
                    + ("" if model is None else f" | loss {loss.item():.3f}")
                )
                window_start = time.perf_counter()

            if step % args.save_freq == 0 and accelerator.is_main_process:
                ckpt = output_dir / f"checkpoint-{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                # Save the dataset stream position alongside the model so a restart resumes mid-stream.
                torch.save(dataset.state_dict(), ckpt / "dataset_state.pt")
                if model is not None:
                    accelerator.unwrap_model(model).save_pretrained(ckpt)

            if step >= args.steps:
                done = True
                break

    accelerator.print(f"End of training: {step} steps, ~{frames_seen} frames/proc")


if __name__ == "__main__":
    main()
