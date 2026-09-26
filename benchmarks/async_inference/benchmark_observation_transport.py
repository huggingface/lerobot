#!/usr/bin/env python
"""Measure the payload benefit and CPU cost of client-side image resizing.

This benchmark deliberately measures image tensors before gRPC serialization. It
therefore isolates the trade-off a future async-inference transport change has
to make: fewer bytes on the wire in exchange for resize work on the robot.

Example:
    uv run --extra async python benchmarks/async_inference/benchmark_observation_transport.py
"""

import argparse
import time

import torch

from lerobot.async_inference.helpers import resize_robot_observation_image


def image_bytes(image: torch.Tensor) -> int:
    return image.numel() * image.element_size()


def benchmark_resize(
    source_shape: tuple[int, int, int], target_shape: tuple[int, int, int], repeats: int
) -> tuple[torch.Tensor, float]:
    image = torch.randint(0, 256, source_shape, dtype=torch.uint8)

    # Warm-up avoids measuring one-time PyTorch dispatch work.
    resized = resize_robot_observation_image(image, target_shape)
    start = time.perf_counter()
    for _ in range(repeats):
        resized = resize_robot_observation_image(image, target_shape)
    duration_ms = (time.perf_counter() - start) * 1_000 / repeats
    return resized, duration_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--target-height", type=int, default=224)
    parser.add_argument("--target-width", type=int, default=224)
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    source_shape = (args.source_height, args.source_width, 3)
    target_shape = (3, args.target_height, args.target_width)
    resized, resize_ms = benchmark_resize(source_shape, target_shape, args.repeats)

    source_bytes = image_bytes(torch.empty(source_shape, dtype=torch.uint8)) * args.cameras
    resized_bytes = image_bytes(resized) * args.cameras
    reduction = 100 * (1 - resized_bytes / source_bytes)

    print(f"| {args.cameras} RGB cameras | Payload | Resize cost / image |")
    print("| --- | ---: | ---: |")
    print(f"| {args.source_width}x{args.source_height} capture | {source_bytes:,} B | — |")
    print(
        f"| {args.target_width}x{args.target_height} policy input | "
        f"{resized_bytes:,} B | {resize_ms:.3f} ms |"
    )
    print(f"\nPayload reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
