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

"""Compare seeded native and server transitions in the same simulator image."""

import argparse
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from lerobot.env_server.client import EnvClient
from lerobot.env_server.contracts import decode_step_result
from lerobot.sim.config import SimConfig
from lerobot.sim.vector_env import VectorEnv
from lerobot.utils import init_logging


def check(config_path: Path, steps: int = 5):
    config = yaml.safe_load(config_path.read_text())
    env = VectorEnv(SimConfig.resolve(config["sim"]))
    try:
        descriptor = env.descriptor
        actions = (
            np.random.default_rng(17)
            .uniform(-0.1, 0.1, (steps, 1, descriptor.action_feature.shape[0]))
            .astype(np.float32)
        )
        env.reset(seed=[123])
        expected = [env.latest]
        for action in actions:
            env.step(action)
            expected.append(env.latest)
    finally:
        env.close()
    config["zenoh"] = {"listen_endpoints": ["tcp/127.0.0.1:7458"]}
    client = EnvClient("tcp/127.0.0.1:7458", config.get("deployment", "default"), timeout_s=1)
    with tempfile.TemporaryDirectory() as directory:
        server_config = Path(directory) / "server.yaml"
        server_config.write_text(yaml.safe_dump(config))
        with (Path(directory) / "server.log").open("w+") as log:
            server = subprocess.Popen(
                [sys.executable, "-m", "lerobot.scripts.lerobot_env_server", "--config", str(server_config)],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 120
                while True:
                    try:
                        client.describe()
                        break
                    except Exception:  # Connection refused before listen, or describe timeout.
                        client.close()
                        if server.poll() is not None or time.monotonic() >= deadline:
                            log.seek(0)
                            raise RuntimeError("Simulator startup failed:\n" + log.read()) from None
                client.timeout_s = client.setup_timeout_s = 120
                actual = [client.open(1, seeds=[123])]
                actual.extend(
                    decode_step_result(client.request("step", actions=action)["result"]) for action in actions
                )
                for reference, remote in zip(expected, actual, strict=True):
                    np.testing.assert_equal(remote.task, reference.task)
                    for key in reference.obs:
                        np.testing.assert_allclose(
                            reference.obs[key], remote.obs[key], rtol=0, atol=1e-6, err_msg=key
                        )
                    for field in ("reward", "terminated", "truncated", "is_success", "step"):
                        np.testing.assert_array_equal(
                            getattr(reference, field), getattr(remote, field), err_msg=field
                        )
                logging.info("Seeded native/server parity passed: %s, %d steps", descriptor.sim_type, steps)
            finally:
                client.close()
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    init_logging()
    check(args.config, args.steps)


if __name__ == "__main__":
    main()
