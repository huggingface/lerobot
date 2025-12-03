#!/usr/bin/env python

"""
Deploy GR00T inference service on Modal with HTTP server.

Setup:
    1. Ensure checkpoint is in Modal volume (lerobot-outputs)
    2. Configure via environment variables
    3. Deploy: modal deploy scripts/modal_inference.py

Usage:
    modal deploy scripts/modal_inference.py
    modal run scripts/modal_inference.py::test
    curl -X POST https://your-app--serve.modal.run -H "Content-Type: application/json" -d '{"observation": {...}, "instruction": "Pick up the cube"}'

Environment Variables:
    CHECKPOINT_PATH: Path to checkpoint (default: /outputs/train/so101_gr00t_test)
    DATA_CONFIG: Data config name (default: so100_fronttop)
    EMBODIMENT_TAG: Embodiment tag (default: new_embodiment)
    DENOISING_STEPS: Denoising steps (default: 4)
"""

import os
from pathlib import Path

import modal

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

# Modal best practices: define constants
MINUTES = 60  # seconds
HOURS = 60 * MINUTES
PORT = 5555
FPS = 30

# Inference configuration - MODIFY THESE VALUES
# LeRobot saves checkpoints in: train/{run_name}/checkpoints/{step}/pretrained_model/
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", "/outputs/train/so101_gr00t_test/checkpoints/020000/pretrained_model"
)
DATA_CONFIG = os.environ.get("DATA_CONFIG", "so100_fronttop")  # Changed to front+top cameras
EMBODIMENT_TAG = os.environ.get("EMBODIMENT_TAG", "new_embodiment")
DENOISING_STEPS = int(os.environ.get("DENOISING_STEPS", "4"))

# Camera and robot configuration (for reference/local testing)
# These configurations document the expected hardware setup
camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=FPS),
    "top": OpenCVCameraConfig(index_or_path=1, width=1920, height=1080, fps=FPS),
}
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem5AA90170931", id="so101_follower_arm", cameras=camera_config
)

# Robot instance (for local development/testing - not used in Modal deployment)
robot = SO101Follower(robot_config)


app = modal.App("groot-inference")
outputs_volume = modal.Volume.from_name("lerobot-outputs", create_if_missing=False)

# Build container image with CUDA 12.4 (required by GR00T)
# Modal host has CUDA 12.9 drivers (backward compatible with 12.4)
# See: https://modal.com/docs/guide/cuda
cuda_version = "12.4.1"  # GR00T verified version (must be <= 12.9 host version)
flavor = "devel"  # devel includes nvcc compiler (required for building flash-attn)
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

inference_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .entrypoint([])  # Remove verbose logging by base image
    .apt_install(
        "git",
        "build-essential",  # gcc, g++, make for compiling flash-attn
        # OpenCV dependencies for headless environment
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        # Install PyTorch first (required by GR00T and flash-attn)
        "torch==2.5.1",
        "torchvision==0.20.1",
    )
    .pip_install(
        # Install build dependencies for flash-attn
        "packaging",
        "ninja",
        "wheel",
    )
    .run_commands(
        # Build flash-attn for A10 GPU (SM86 architecture)
        # This compiles from source using nvcc from the devel image
        "TORCH_CUDA_ARCH_LIST='8.6' pip install flash-attn==2.8.2 --no-build-isolation",
    )
    # Copy gr00t package and install with [base] extras - this installs ALL dependencies from pyproject.toml
    .add_local_dir(".", remote_path="/workspace/gr00t", copy=True)
    .run_commands(
        "pip install -e '/workspace/gr00t[base]'",  # Installs all core + [base] optional dependencies
    )
    # Install Modal-specific dependencies not in pyproject.toml
    .pip_install(
        "fastapi",
        "uvicorn",
    )
)


# Global policy instance (initialized on container startup)
_policy = None
_modality_config = None


def _get_policy():
    """
    Lazy initialization of policy on first request using LeRobot-native approach.

    This follows the standard LeRobot pattern from HuggingFace docs:
    https://huggingface.co/docs/lerobot/en/il_robots#run-inference-and-evaluate-your-policy
    """
    global _policy, _modality_config
    if _policy is None:
        print("Initializing GR00T policy with LeRobot-native approach...")

        # Load dataset to get modality config (LeRobot standard)
        dataset = LeRobotDataset(DATA_CONFIG)
        modality_config = dataset.meta.modality_config

        # Create pre/post processors using LeRobot factory pattern
        pre_processor, post_processor = make_pre_post_processors(
            modality_config=modality_config,
            policy_type="groot",
        )

        # Initialize policy with LeRobot GrootPolicy (not gr00t package)
        _policy = GrootPolicy(
            model_path=CHECKPOINT_PATH,
            modality_config=modality_config,
            modality_transform=pre_processor,
            embodiment_tag=EMBODIMENT_TAG,
            denoising_steps=DENOISING_STEPS,
        )
        _modality_config = modality_config
        print("Policy initialized successfully!")
    return _policy


@app.function(
    gpu="A10",
    image=inference_image,
    volumes={"/outputs": outputs_volume},
    timeout=HOURS,
)
def test():
    """Test policy initialization and inference with dummy data."""
    import numpy as np

    policy = _get_policy()

    # Create dummy observation matching so100_fronttop config
    dummy_obs = {
        "images": {
            "front": np.zeros((480, 640, 3), dtype=np.uint8),
            "top": np.zeros((480, 640, 3), dtype=np.uint8),
        },
        "joint_position": np.zeros(7, dtype=np.float32),
        "instruction": "Test instruction",
    }

    action = policy.get_action(dummy_obs)
    print(f"Test action shape: {action['action'].shape if 'action' in action else 'N/A'}")
    return {"status": "ok", "action_shape": str(action.get("action", {}).shape)}


@app.function(
    gpu="A10",
    image=inference_image,
    volumes={
        "/outputs": outputs_volume,
    },
    timeout=6 * HOURS,
    min_containers=1,  # Keep one container warm for fast inference
)
@modal.fastapi_endpoint(method="POST")
def serve(body: dict):
    """
    HTTP inference endpoint served directly by Modal.

    This function handles POST requests with observation data and returns actions.
    The policy is initialized once per container and reused across requests.

    Note: @modal.fastapi_endpoint serves at the root path (/), not /act

    Example client usage:
        import requests
        import numpy as np

        # Convert numpy arrays to lists for JSON serialization
        obs_serialized = {k: v.tolist() if isinstance(v, np.ndarray) else v
                          for k, v in observation_dict.items()}

        response = requests.post(
            serve.web_url,  # Calls root path directly
            json={"observation": obs_serialized, "instruction": "Pick up the cube"}
        )
        action = response.json()  # Returns lists that can be converted back to numpy
    """
    import base64

    import cv2
    import numpy as np

    policy = _get_policy()

    # Get observation from request body
    obs = body.get("observation")
    if obs is None:
        return {"error": "Missing 'observation' in request body"}, 400

    # Extract language instruction for VLA
    instruction = body.get("instruction", "")

    # Decompress images if they were compressed by the bridge
    obs_decompressed = {}
    for key, value in obs.items():
        if isinstance(value, dict) and "type" in value:
            # Handle compressed images
            if value["type"] == "jpeg_base64":
                # Decompress single frame
                jpeg_bytes = base64.b64decode(value["data"])
                image_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                obs_decompressed[key] = image_rgb
            else:
                obs_decompressed[key] = value
        elif isinstance(value, list):
            # Convert lists back to numpy arrays
            obs_decompressed[key] = np.array(value) if value else value
        else:
            obs_decompressed[key] = value

    # Add language instruction for VLA
    if instruction:
        obs_decompressed["instruction"] = instruction

    # Validate observation keys against data config
    if _modality_config is not None:
        expected_keys = set(_modality_config.keys())
        provided_keys = set(obs_decompressed.keys())
        missing = expected_keys - provided_keys
        if missing:
            return {
                "error": f"Missing required observation keys: {missing}",
                "expected": list(expected_keys),
            }, 400

    # Run inference with error handling
    try:
        action = policy.get_action(obs_decompressed)
    except Exception as e:
        import traceback

        return {"error": f"Inference failed: {str(e)}", "traceback": traceback.format_exc()}, 500

    # Convert numpy arrays to lists for JSON serialization
    # (avoiding json_numpy.patch() which breaks numpy's internal imports)
    serialized_action = {}
    for key, value in action.items():
        if isinstance(value, np.ndarray):
            serialized_action[key] = value.tolist()
        else:
            serialized_action[key] = value

    return serialized_action


@app.function(
    image=inference_image,
    volumes={
        "/outputs": outputs_volume,
    },
)
@modal.fastapi_endpoint(method="GET")
def health():
    """
    Health check endpoint (HTTP).

    Returns service status and configuration info.
    Can be accessed via browser or curl for quick status checks.

    Note: Each function has its own web_url serving at root (/)

    Example:
        curl https://your-app--health.modal.run
    """
    checkpoint_exists = Path(CHECKPOINT_PATH).exists()

    return {
        "status": "healthy" if checkpoint_exists else "degraded",
        "model": "GR00T N1.5",
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_exists": checkpoint_exists,
        "data_config": DATA_CONFIG,
        "embodiment_tag": EMBODIMENT_TAG,
        "denoising_steps": DENOISING_STEPS,
        "gpu": "A10",
        "protocol": "HTTP",
        "serve_url": serve.get_web_url(),
        "health_url": health.get_web_url(),
    }
