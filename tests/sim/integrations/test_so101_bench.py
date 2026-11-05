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
"""SO-101 Bench tests against a protocol stand-in for the Isaac Sim server (no Isaac Sim needed)."""

import stat
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip("zmq")
pytest.importorskip("msgpack")

from lerobot.envs.configs import SO101BenchEnvConfig  # noqa: E402
from lerobot.sim.integrations import so101_bench  # noqa: E402

FAKE_SERVER = textwrap.dedent(
    """
    import argparse
    import msgpack
    import numpy as np
    import zmq

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--init_frames", action="store_true")
    args, _ = parser.parse_known_args()
    H, W = 480, 640
    EPISODES = [
        {"trial_id": 0, "instruction": "Place each object in the plastic bin", "split": "seen", "family": "bin_1",
         "objects": ["black tape"], "appearance_profile": "canonical", "max_steps": 40},
        {"trial_id": 1, "instruction": "Move the pliers left", "split": "seen", "family": "move",
         "objects": ["blue pliers"], "appearance_profile": "varied", "max_steps": 40},
    ]

    def observation():
        images = {n: np.full((H, W, 3), 7, np.uint8).tobytes() for n in ("overhead", "wrist")}
        if args.init_frames:
            images |= {f"{k}_init": v for k, v in list(images.items())}
        return {"state": [0.0] * 6, "images": images}

    socket = zmq.Context.instance().socket(zmq.REP)
    socket.bind(f"tcp://127.0.0.1:{args.port}")
    steps = trial = 0
    while True:
        request = msgpack.unpackb(socket.recv(), raw=False)
        if request["cmd"] == "info":
            reply = {"episodes": EPISODES, "fps": 30, "cameras": {"overhead": [H, W], "wrist": [H, W]}}
        elif request["cmd"] == "reset":
            steps, trial = 0, request["trial_id"]
            reply = observation()
        elif request["cmd"] == "step":
            assert len(request["action"]) == 6
            steps += 1
            done = steps >= 5
            success = done and trial == 0
            reply = {**observation(), "reward": float(success), "terminated": done, "truncated": False,
                     "is_success": success, "end_reason": "success" if success else ""}
        else:
            socket.send(msgpack.packb({}, use_bin_type=True))
            break
        socket.send(msgpack.packb(reply, use_bin_type=True))
    """
)


@pytest.fixture
def fake_checkout(tmp_path):
    """An SO-101 Bench checkout whose sim server is the stand-in, and an isaaclab.sh that runs it."""
    server = tmp_path / so101_bench.SERVER_SCRIPT
    server.parent.mkdir(parents=True)
    server.write_text(FAKE_SERVER)
    isaaclab_sh = tmp_path / "isaaclab.sh"
    isaaclab_sh.write_text(f'#!/usr/bin/env bash\nshift\nexec "{sys.executable}" "$@"\n')
    isaaclab_sh.chmod(isaaclab_sh.stat().st_mode | stat.S_IEXEC)
    yield tmp_path, isaaclab_sh
    for client in so101_bench._CLIENTS.values():
        client.close()
        assert client.process is None
    so101_bench._CLIENTS.clear()


def _config(fake_checkout, **kwargs):
    root, isaaclab_sh = fake_checkout
    return SO101BenchEnvConfig(
        so101_bench_root=str(root), isaaclab_sh=str(isaaclab_sh), isaac_venv=None, **kwargs
    )


def test_parse_trial_ids():
    assert so101_bench.parse_trial_ids("0-3,7") == [0, 1, 2, 3, 7]
    assert so101_bench.parse_trial_ids(None) is None


def test_rejects_unknown_camera():
    with pytest.raises(ValueError, match="Unknown SO-101 Bench camera"):
        SO101BenchEnvConfig(cameras="front")


def test_rejects_parallel_tasks():
    with pytest.raises(ValueError, match="max_parallel_tasks=1"):
        SO101BenchEnvConfig(max_parallel_tasks=2)


def test_rejects_batches(fake_checkout):
    with pytest.raises(ValueError, match="batch_size=1"):
        _config(fake_checkout).create_envs(n_envs=2)


@pytest.mark.parametrize("cameras", ["overhead,wrist", "overhead,wrist,overhead_init,wrist_init"])
def test_episode_rollout(fake_checkout, cameras):
    cfg = _config(fake_checkout, cameras=cameras)
    envs = cfg.create_envs(n_envs=1)
    assert set(envs) == {"seen/bin_1", "seen/move"}
    camera_list = cameras.split(",")
    assert {key for key in cfg.features if key.startswith("observation.images.")} == {
        f"observation.images.{c}" for c in camera_list
    }
    assert cfg.features["observation.state"].shape == (6,)
    for suite, success in (("seen/bin_1", True), ("seen/move", False)):
        env = next(iter(envs[suite].values()))
        obs, _ = env.reset(seed=0)
        assert obs["observation.images.overhead"].shape == (1, 480, 640, 3)
        assert obs["observation.state"].shape == (1, 6)
        assert env.call("task_description")[0]
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(np.zeros((1, 6), dtype=np.float32))
        assert terminated[0]
        assert bool(info["is_success"][0]) == success
        env.close()
    # Every task shares one simulator.
    assert len(so101_bench._CLIENTS) == 1


def test_task_selects_suites(fake_checkout):
    envs = _config(fake_checkout, task="seen/move", episodes="0-1").create_envs(n_envs=1)
    assert set(envs) == {"seen/move"}
    assert list(envs["seen/move"]) == [1]
