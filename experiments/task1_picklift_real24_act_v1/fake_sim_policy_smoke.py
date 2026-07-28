from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim_policy_inference import CONTROL_HZ, Task1ActSimInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-environment Task1 ACT inference smoke.")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise RuntimeError("--steps must be positive.")

    policy = Task1ActSimInference()
    policy.reset_episode()
    state = np.zeros(6, dtype=np.float32)
    front = np.zeros((480, 640, 3), dtype=np.uint8)
    records = []
    for step in range(args.steps):
        result = policy.infer(state=state, front_rgb=front)
        record = {"step": step, **result.to_jsonable()}
        records.append(record)
        state = result.sent_action.copy()

    payload = {
        "status": "pass",
        "environment_created": False,
        "hardware_accessed": False,
        "control_hz_contract": CONTROL_HZ,
        "checkpoint": str(policy.checkpoint),
        "model_sha256": policy.model_hash,
        "steps": records,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
