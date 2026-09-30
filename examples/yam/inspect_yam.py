#!/usr/bin/env python
"""Read YAM feedback, measure grippers manually, or capture three cameras. Never enable torque."""

import argparse
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path

import draccus
import numpy as np
from PIL import Image

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots.bi_yam_follower import BiYamFollowerConfig, YamArmConfig
from lerobot.robots.bi_yam_follower.bi_yam_follower import decode_positions, make_yam_bus, verify_adapter
from lerobot.robots.bi_yam_follower.config_bi_yam_follower import MOTOR_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Rollout JSON file")
    parser.add_argument("--calibrate-grippers", action="store_true")
    parser.add_argument(
        "--snapshot", type=Path, help="Output folder for RGB images and normalized-gripper state"
    )
    args = parser.parse_args()
    profile = json.loads(args.config.read_text())
    robot_data = profile["robot"].copy()
    robot_data.pop("type", None)
    config = draccus.decode(BiYamFollowerConfig, robot_data)
    configs = {"left": config.left_arm, "right": config.right_arm}
    with ExitStack() as cleanup:
        buses = {}
        for side, arm_config in configs.items():
            verify_adapter(arm_config)
            bus = make_yam_bus(arm_config)
            bus.connect(handshake=False)
            cleanup.callback(bus.disconnect, disable_torque=False)
            buses[side] = bus
        if args.calibrate_grippers:
            measurements = {}
            print("Support both arms. Move only the grippers by hand; stop if either resists.")
            for endpoint in ("closed", "open"):
                input(f"Gently put BOTH grippers fully {endpoint}, release them, then press Enter: ")
                samples = {side: [] for side in buses}
                for _ in range(10):
                    for side, bus in buses.items():
                        samples[side].append(
                            math.radians(bus.sync_read_all_states(strict=True)["gripper"]["position"])
                        )
                    time.sleep(0.02)
                if any(np.ptp(values) > 0.03 for values in samples.values()):
                    raise ValueError("Grippers moved during measurement; configuration was not changed")
                measurements[endpoint] = {side: float(np.median(values)) for side, values in samples.items()}
            updates = {}
            for side, cfg in configs.items():
                closed, opened = measurements["closed"][side], measurements["open"][side]
                if not 0.5 < abs(opened - closed) < 10:
                    raise ValueError(f"{side} stroke is implausible; verify the physical stops")
                updates[side] = {"gripper_closed_rad": closed, "gripper_open_rad": opened}
                validated = draccus.encode(cfg)
                validated.update(updates[side])
                configs[side] = draccus.decode(YamArmConfig, validated)
            print(json.dumps(updates, indent=2))
            for side, values in updates.items():
                profile["robot"][f"{side}_arm"].update(values)
            # An atomic replacement avoids leaving a partly written rollout profile.
            temporary = args.config.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(profile, indent=2) + "\n")
            temporary.replace(args.config)
            print(f"Saved measured gripper endpoints to {args.config}")
        raw_states = {side: bus.sync_read_all_states(strict=True) for side, bus in buses.items()}
        print(
            json.dumps(
                {
                    side: {name: math.radians(states[name]["position"]) for name in MOTOR_NAMES}
                    for side, states in raw_states.items()
                },
                indent=2,
            )
        )
        if args.snapshot:
            state = np.concatenate([decode_positions(configs[side], raw_states[side]) for side in buses])
            if set(config.cameras) != {"top", "left", "right"}:
                raise ValueError("Expected top, left, and right camera configuration")
            cameras = make_cameras_from_configs(config.cameras)
            for camera in cameras.values():
                camera.connect()
                cleanup.callback(camera.disconnect)
            args.snapshot.mkdir(parents=True, exist_ok=True)
            observation = {"state": state.tolist()}
            for name, camera in cameras.items():
                path = (args.snapshot / f"{name}.png").resolve()
                Image.fromarray(camera.async_read()).save(path)
                observation[name] = str(path)
            (args.snapshot / "observation.json").write_text(json.dumps(observation, indent=2) + "\n")
            print(f"Saved RGB snapshot to {args.snapshot}; no motor commands were sent")


if __name__ == "__main__":
    main()
