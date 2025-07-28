"""
Asynchronous LeRobot runner **with Gemini verification**
=======================================================

* **Exact** robot / dataset / policy loading pattern from your working snippet
  (no simulation, real hardware, uses `robot.name` for `robot_type`).
* Runs the policy at 15 FPS *while* querying Gemini concurrently.
* Stops on the first *yes* reply from Gemini or after a 60 s timeout.
* Requires `google‑genai ≥ 1.26`, `python‑dotenv`, and LeRobot dependencies.

```bash
pip install google-genai>=1.26 python-dotenv opencv-python

# .env file (put next to the script):
GEMINI_API_KEY="your‑key"
# or
GOOGLE_API_KEY="your‑key"
```
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# ── Load Gemini API key ───────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Add GEMINI_API_KEY or GOOGLE_API_KEY to your .env file.")

# ── Gemini client (async) ─────────────────────────────────────────────────────
from google import genai
from google.genai import types as genai_types

CLIENT = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash-001"  # update as newer models appear

# ── LeRobot imports — identical to your snippet ──────────────────────────────
from lerobot.robots import make_robot_from_config
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101FollowerConfig

# ── User parameters ───────────────────────────────────────────────────────────
POLICY_PATH = "aiden-li/so101-act-grabtissue"  # HF repo or local path
TASK = "Grab a tissue"
TIMEOUT_S = 60  # overall episode timeout
FPS = 15
DEVICE = "cuda"  # or "cpu"
SERIAL_PORT = "COM6"  # change for your robot
TMP_DATASET_DIR = Path("./tmp_dataset")

# ── Build and connect robot (same as snippet) ────────────────────────────────

def build_robot() -> Any:
    cfg = SO101FollowerConfig(
        port=SERIAL_PORT,
        cameras={
            "front": OpenCVCameraConfig(0, width=640, height=480, fps=FPS),
            "ego": OpenCVCameraConfig(1, width=640, height=480, fps=FPS),
            "top": OpenCVCameraConfig(2, width=640, height=480, fps=FPS),
        },
    )
    robot = make_robot_from_config(cfg)
    robot.connect()
    return robot

# ── Load policy exactly like original code ────────────────────────────────────

def load_policy(robot):
    """Load the pretrained policy **exactly** as in your working synchronous
    script so `dataset.meta` is always available for `make_policy`."""

    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    action_features = hw_to_dataset_features(robot.action_features, "action")
    features = {**obs_features, **action_features}

    # --- Create the dummy dataset (mirror original snippet) ------------------
    shutil.rmtree("tmp_dataset", ignore_errors=True)  # string path, not Path
    dataset = LeRobotDataset.create(
        repo_id="dummy/so101",
        fps=FPS,
        root="./tmp_dataset",  # EXACTLY as in the snippet
        robot_type=robot.name,
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=1,
    )

    cli_overrides = parser.get_cli_overrides("policy")
    cfg = PreTrainedConfig.from_pretrained(POLICY_PATH, cli_overrides=cli_overrides)
    cfg.pretrained_path = POLICY_PATH

    policy = make_policy(cfg, ds_meta=dataset.meta)  # dataset.meta guaranteed
    policy.reset()
    return policy, features

# ── Gemini async helper ───────────────────────────────────────────────────────

async def gemini_task_done(task: str, observation: dict[str, Any]) -> bool:
    """Return True if Gemini asserts the task is complete."""
    prompt = (
        "You are an expert robotic supervisor. Answer strictly with 'yes' or 'no'.\n\n"
        f"Task: {task}\n"
        "Current robot observation (JSON):\n"
        f"{json.dumps(observation, indent=2, default=str)}"
    )
    rsp = await CLIENT.aio.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
    )
    return rsp.text.strip().lower().startswith("y")

# ── Main asynchronous episode ────────────────────────────────────────────────

async def run_episode() -> bool:
    robot = build_robot()
    policy, features = load_policy(robot)
    device = get_safe_torch_device(DEVICE)

    start = time.perf_counter()
    g_task: Optional[asyncio.Task] = None

    try:
        while time.perf_counter() - start < TIMEOUT_S:
            # 1. Get observation without blocking the event loop
            obs = await asyncio.get_event_loop().run_in_executor(None, robot.get_observation)
            frame = build_dataset_frame(features, obs, prefix="observation")

            # 2. Policy inference (also in thread)
            act_vals = await asyncio.get_event_loop().run_in_executor(
                None,
                predict_action,
                frame,
                policy,
                device,
                policy.config.use_amp,
                TASK,
                robot.robot_type,
            )
            action = {k: act_vals[i].item() for i, k in enumerate(robot.action_features)}
            robot.send_action(action)

            # 3. Gemini query lifecycle
            if g_task is None or g_task.done():
                if g_task and g_task.done():
                    try:
                        if g_task.result():
                            return True
                    except Exception as e:  # noqa: BLE001
                        print(f"[WARN] Gemini error: {e}")
                g_task = asyncio.create_task(gemini_task_done(TASK, obs))

            await asyncio.sleep(1 / FPS)
    finally:
        robot.disconnect()

    return False

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    outcome = asyncio.run(run_episode())
    print(f"Episode result → {outcome}")
