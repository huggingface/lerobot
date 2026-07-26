#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Laptop-side sender for the SONIC whole-body walk policy, onboard deployment.

This is the counterpart to ``run_g1_onboard.py`` (which runs the SONIC decoder on the
robot). The heavy VLA (``nepyope/sonic_walk``, a pi0.5 token policy) runs here on the
laptop GPU; only the resulting 64-D latent token is shipped to the robot over ZMQ:

    laptop:  camera frame (ZMQ from robot :5555) + previous token
             -> pi0.5 -> next 64-D token
             -> PUSH JSON {motion_token.i.pos: ...} to robot :6004
    robot:   run_g1_onboard receives the token, SonicWholeBodyController decodes it
             into whole-body joint commands against local DDS at full rate.

The policy's ``observation.state`` is the token currently being executed, so we close
the loop by feeding back the *last token we sent* (the decoder holds it until a new one
arrives). This mirrors what ``lerobot-rollout`` does via the robot's token echo, but
without a controller / DDS on the laptop.

The policy is pi0.5 with chunk_size=50, so a full diffusion inference runs only about
once every 50 ticks; ``select_action`` pops one queued token per tick in between.

Run ``run_g1_onboard.py --controller SonicWholeBodyController --sonic-token-action
--cameras ...`` on the robot first, then this on the laptop:

    python -m lerobot.robots.unitree_g1.infer_sonic_g1_onboard \
        --policy-path nepyope/sonic_walk --robot-ip 192.168.123.164 \
        --task "walk back and forth"
"""

import argparse
import contextlib
import json
import logging
import signal
import time

import numpy as np
import torch

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.unitree_g1.controllers.sonic_whole_body import TOKEN_DIM, token_action_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("sonic_sender")

ACTION_PORT = 6004  # matches run_g1_onboard.py --action-port
IMAGE_KEY = "observation.images.ego_view"  # pi05 sonic_walk VISUAL input
STATE_KEY = "observation.state"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", default="nepyope/sonic_walk", help="Policy repo id or local path")
    p.add_argument("--robot-ip", default="192.168.123.164", help="Robot IP (camera + action ports)")
    p.add_argument("--action-port", type=int, default=ACTION_PORT, help="Onboard ZMQ PULL port for actions")
    p.add_argument("--camera-port", type=int, default=5555, help="Onboard ZMQ camera PUB port")
    p.add_argument("--camera-name", default="head_camera", help="Camera name served by run_g1_onboard")
    p.add_argument("--camera-width", type=int, default=640, help="Camera width")
    p.add_argument("--camera-height", type=int, default=480, help="Camera height")
    p.add_argument("--task", default="walk back and forth", help="Language prompt for the VLA")
    p.add_argument("--fps", type=float, default=30.0, help="Token send rate (matches training inference)")
    p.add_argument("--device", default="cuda", help="Torch device")
    p.add_argument("--max-ticks", type=int, default=0, help="Stop after N ticks (0 = run forever)")
    p.add_argument("--dry-run", action="store_true", help="Run inference but do not PUSH tokens to the robot")
    args = p.parse_args()

    device = torch.device(args.device)

    # --- Policy + processors (normalization stats baked into the checkpoint) ---
    logger.info("Loading policy from '%s'...", args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy = get_policy_class(policy_cfg.type).from_pretrained(args.policy_path, config=policy_cfg)
    policy = policy.to(device)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    logger.info("Policy loaded (type=%s, device=%s, chunk=%s)", policy_cfg.type, device,
                getattr(policy_cfg, "chunk_size", "?"))

    # --- Camera (ZMQ from the robot's onboard image server) ---
    cam = ZMQCamera(
        ZMQCameraConfig(
            server_address=args.robot_ip,
            port=args.camera_port,
            camera_name=args.camera_name,
            width=args.camera_width,
            height=args.camera_height,
            fps=int(args.fps),
        )
    )
    logger.info("Connecting camera %s@%s:%d ...", args.camera_name, args.robot_ip, args.camera_port)
    cam.connect()

    # --- Action PUSH socket to the onboard controller ---
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{args.robot_ip}:{args.action_port}")
    logger.info("Sending tokens to tcp://%s:%d (dry_run=%s)", args.robot_ip, args.action_port, args.dry_run)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))

    # observation.state = the token currently executing on the robot (last one we sent);
    # start at zeros, matching the decoder's zero-seeded initial state.
    prev_token = np.zeros(TOKEN_DIM, dtype=np.float32)
    period = 1.0 / args.fps
    n = 0
    t_infer_total = 0.0
    logger.info("Streaming tokens at %.0f Hz. Ctrl-C to stop.", args.fps)
    try:
        while not stop["flag"]:
            t0 = time.time()
            try:
                frame = cam.read()  # HxWxC uint8 RGB
            except Exception as e:  # noqa: BLE001
                logger.warning("Camera read failed: %s", e)
                time.sleep(period)
                continue

            raw_obs = {
                IMAGE_KEY: np.ascontiguousarray(frame),
                STATE_KEY: prev_token.copy(),
            }
            with torch.inference_mode():
                obs = prepare_observation_for_inference(raw_obs, device, args.task, "unitree_g1")
                obs = preprocessor(obs)
                action = policy.select_action(obs)
                action = postprocessor(action)
            token = action.squeeze(0).to("cpu").numpy().astype(np.float32)
            prev_token = token

            if not args.dry_run:
                msg = {token_action_key(i): float(token[i]) for i in range(TOKEN_DIM)}
                with contextlib.suppress(zmq.Again):
                    sock.send_string(json.dumps(msg), zmq.NOBLOCK)

            n += 1
            t_infer_total += time.time() - t0
            if n % 30 == 0:
                logger.info(
                    "tick %d | avg %.1f ms/tick | token[:3]=%s",
                    n, 1000.0 * t_infer_total / 30.0, np.round(token[:3], 3).tolist(),
                )
                t_infer_total = 0.0

            if args.max_ticks and n >= args.max_ticks:
                break
            time.sleep(max(0.0, period - (time.time() - t0)))
    finally:
        logger.info("Stopping sender after %d ticks.", n)
        with contextlib.suppress(Exception):
            cam.disconnect()
        with contextlib.suppress(Exception):
            sock.close(linger=0)


if __name__ == "__main__":
    main()
