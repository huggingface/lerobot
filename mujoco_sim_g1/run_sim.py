#!/usr/bin/env python3
"""Standalone MuJoCo simulator for Unitree G1"""
import argparse
import sys
from pathlib import Path

# Add sim module to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from sim.simulator_factory import SimulatorFactory, init_channel

def main():
    parser = argparse.ArgumentParser(description="Unitree G1 MuJoCo Simulator")
    parser.add_argument("--publish-images", action="store_true", 
                       help="Enable camera image publishing via ZMQ (requires offscreen rendering)")
    parser.add_argument("--camera-port", type=int, default=5555,
                       help="ZMQ port for camera publishing (default: 5555)")
    parser.add_argument("--cameras", type=str, nargs="+", default=None,
                       help="Camera names to publish (default: head_camera)")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    enable_offscreen = args.publish_images or config.get("ENABLE_OFFSCREEN", False)
    
    print("="*60)
    print("🤖 Starting Unitree G1 MuJoCo Simulator")
    print("="*60)
    print(f"📁 Scene: {config['ROBOT_SCENE']}")
    print(f"⏱️  Timestep: {config['SIMULATE_DT']}s ({int(1/config['SIMULATE_DT'])} Hz)")
    print(f"👁️  Visualization: {'ON' if config.get('ENABLE_ONSCREEN', True) else 'OFF'}")
    
    # Configure cameras if requested
    camera_configs = {}
    if enable_offscreen:
        camera_list = args.cameras or ["head_camera"]
        for cam_name in camera_list:
            camera_configs[cam_name] = {"height": 480, "width": 640}
        print(f"📷 Cameras: {', '.join(camera_list)} → ZMQ port {args.camera_port}")
    
    print("="*60)
    
    # Initialize DDS channel
    init_channel(config=config)
    
    # Create simulator
    sim = SimulatorFactory.create_simulator(
        config=config,
        env_name="default",
        onscreen=config.get("ENABLE_ONSCREEN", True),
        offscreen=enable_offscreen,
        camera_configs=camera_configs,
    )
    
    # Start simulator (blocking)
    print("\nSimulator running. Press Ctrl+C to exit.")
    if enable_offscreen and args.publish_images:
        print(f"Camera images publishing on tcp://localhost:{args.camera_port}")
    try:
        SimulatorFactory.start_simulator(
            sim,
            as_thread=False,
            enable_image_publish=args.publish_images,
            camera_port=args.camera_port,
        )
    except KeyboardInterrupt:
        print("\nShutting down simulator...")
        sim.close()

if __name__ == "__main__":
    main()

