#!/usr/bin/env python3
"""
Script to process local dataset videos, apply object detection and masking,
and save the masked videos with the same format, frame count, and duration.
Uses PyAV for direct processing of AV1/H.264 videos.
"""
import os
import time
import argparse
import av
import numpy as np
from tqdm import tqdm
from fractions import Fraction
from lerobot.active_feedback.detect.object_detector import ObjectDetector
from lerobot.active_feedback.detect.frame_masker import FrameMasker

def process_local_dataset(input_dir, output_dir, debug=True, max_episodes=None):
    os.makedirs(output_dir, exist_ok=True)

    # Initialize detector and masker with config
    detector = ObjectDetector(
        debug=debug,
        config_path="inference_config.yaml"
    )
    frame_masker = FrameMasker(debug=debug)

    # Find camera subdirectories
    camera_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if debug:
        print(f"Found camera directories: {camera_dirs}")

    # Statistics tracking
    stats = {
        "total_videos": 0,
        "total_frames": 0,
        "processed_frames": 0,
        "detected_objects": 0,
        "processing_time": 0
    }

    # Process each camera
    for cam in camera_dirs:
        cam_in = os.path.join(input_dir, cam)
        cam_out = os.path.join(output_dir, cam)
        os.makedirs(cam_out, exist_ok=True)

        videos = sorted([f for f in os.listdir(cam_in) if f.lower().endswith('.mp4')])
        if max_episodes:
            videos = videos[:max_episodes]
        stats["total_videos"] += len(videos)

        for vid in tqdm(videos, desc=f"Camera {cam}"):
            start_time = time.time()
            in_path = os.path.join(cam_in, vid)
            out_path = os.path.join(cam_out, vid)
            process_video_with_pyav(in_path, out_path, detector, frame_masker, stats, debug)
            stats["processing_time"] += time.time() - start_time

    # Summary
    if debug:
        print("Processing complete.")
        print(f"Videos: {stats['total_videos']}, Frames: {stats['total_frames']}, Processed: {stats['processed_frames']}")
        print(f"Detected objects: {stats['detected_objects']}")
        print(f"Total time: {stats['processing_time']:.2f}s")

    return stats


def process_video_with_pyav(input_path, output_path, detector, frame_masker, stats, debug=False):
    # Open input video
    container = av.open(input_path)
    stream = container.streams.video[0]

    # Video properties
    fps = float(stream.average_rate)
    width, height = stream.width, stream.height

    # Determine frame count
    total_frames = stream.frames or (
        int(stream.duration * fps / stream.time_base.denominator)
        if stream.duration is not None else
        sum(1 for _ in container.decode(video=0))
    )
    if not stream.frames:
        container.seek(0)
    stats["total_frames"] += total_frames

    # Create output video with same fps and resolution
    output = av.open(output_path, mode='w')
    out_stream = output.add_stream('h264', rate=Fraction(int(fps * 1000), 1000))
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = 'yuv420p'
    out_stream.time_base = stream.time_base

    # Process each frame
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        result = detector.detect_and_segment(img, box_threshold=0.25)
        masks = result.get("masks", [])

        # Combine masks into boolean array
        combined = np.zeros((height, width), dtype=bool)
        for m in masks:
            mask_bool = m.astype(bool)
            combined = np.logical_or(combined, mask_bool)
        stats["detected_objects"] += len(masks)

        # Create masked image or black image
        if combined.any():
            masked = frame_masker.create_masked_image(img, combined)
        else:
            masked = np.zeros_like(img)

        # Write frame preserving timing
        out_frame = av.VideoFrame.from_ndarray(masked, format='rgb24')
        out_frame.pts = frame.pts
        out_frame.time_base = frame.time_base
        for packet in out_stream.encode(out_frame):
            output.mux(packet)

        stats["processed_frames"] += 1

    # Flush encoder
    for packet in out_stream.encode():
        output.mux(packet)

    container.close()
    output.close()

    if debug:
        print(f"{os.path.basename(input_path)}: {stats['processed_frames']}/{total_frames} frames processed.")


def main():
    parser = argparse.ArgumentParser(description="Process local dataset videos with object detection and masking, retaining original length.")
    parser.add_argument(
        "--input-dir", type=str,
        default="/home/demo/lerobot-beavr/datasets/koch_masked/videos/chunk-000",
        help="Directory containing the original dataset videos"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/home/demo/lerobot-beavr/datasets/koch_masked/videos_masked/chunk-000",
        help="Directory to save the masked videos"
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Maximum number of videos to process per camera"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable debug logging"
    )
    args = parser.parse_args()

    stats = process_local_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        debug=args.debug,
        max_episodes=args.max_episodes
    )

    print(f"Done: {stats['total_videos']} videos, {stats['processed_frames']} frames processed.")

if __name__ == "__main__":
    main()
