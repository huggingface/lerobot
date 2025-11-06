#!/usr/bin/env python

"""Simple test script to visualize DINOv2/v3 features on camera feed or image file.

Example usage:
    # With webcam (DINOv3 ViT)
    python examples/test_dinov2_visualizer.py --camera_index 0

    # With image file
    python examples/test_dinov2_visualizer.py --image_path path/to/image.jpg

    # With DINOv3 ConvNeXt model
    python examples/test_dinov2_visualizer.py --camera_index 0 --model facebook/dinov3-convnext-base-pretrain-lvd1689m

    # With DINOv2 model
    python examples/test_dinov2_visualizer.py --camera_index 0 --model facebook/dinov2-large

    # With attention visualization (ViT only)
    python examples/test_dinov2_visualizer.py --camera_index 0 --visualize_attention
"""

import argparse
import time

import cv2
import numpy as np
import rerun as rr

from lerobot.utils.vision_visualizers import make_vision_visualizer


def main():
    parser = argparse.ArgumentParser(description="Test DINOv2/v3 feature visualizer")
    parser.add_argument("--camera_index", type=int, default=None, help="Camera index (e.g., 0)")
    parser.add_argument("--image_path", type=str, default=None, help="Path to image file")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/dinov3-vit-base-pretrain-lvd1689m",
        help="DINOv2/v3 model to use (HuggingFace model name)",
    )
    parser.add_argument("--visualize_attention", action="store_true", help="Visualize attention maps (ViT only)")
    parser.add_argument("--fps", type=int, default=10, help="FPS for camera feed")

    args = parser.parse_args()

    if args.camera_index is None and args.image_path is None:
        parser.error("Either --camera_index or --image_path must be specified")

    # Initialize rerun
    rr.init("dinov2_visualizer_test", spawn=True)

    # Create visualizer
    print(f"Loading DINOv2/v3 model: {args.model}")
    print("This may take a few moments on first run (downloading weights)...")
    visualizer = make_vision_visualizer(
        visualizer_type="dinov2",
        model_name=args.model,
        visualize_attention=args.visualize_attention,
        log_to_rerun=True,
    )
    print("Model loaded and frozen for inference!")

    if args.image_path is not None:
        # Test with single image
        print(f"Processing image: {args.image_path}")
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"Could not read image: {args.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        features = visualizer(image, "test_camera")
        print(f"CLS token shape: {features['cls_token'].shape}")
        print(f"Patch tokens shape: {features['patch_tokens'].shape}")
        print(f"Spatial features shape: {features['spatial_features'].shape}")
        print("Check the Rerun viewer to see visualizations!")
        input("Press Enter to exit...")

    else:
        # Test with camera feed
        print(f"Opening camera {args.camera_index}")
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {args.camera_index}")

        print(f"Running at {args.fps} FPS. Press Ctrl+C to exit.")
        frame_time = 1.0 / args.fps

        try:
            while True:
                start = time.time()

                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Visualize features
                visualizer(frame_rgb, "test_camera")

                # Maintain FPS
                elapsed = time.time() - start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

                actual_fps = 1.0 / (time.time() - start)
                print(f"\rFPS: {actual_fps:.1f}", end="", flush=True)

        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            cap.release()
            rr.rerun_shutdown()


if __name__ == "__main__":
    main()
