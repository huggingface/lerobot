#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# TODO(aliberts): Mute logging for these tests
import subprocess
import sys
from pathlib import Path

from tests.utils import require_package


def _find_and_replace(text: str, finds_and_replaces: list[tuple[str, str]]) -> str:
    for f, r in finds_and_replaces:
        assert f in text
        text = text.replace(f, r)
    return text


def _run_script(path):
    subprocess.run([sys.executable, path], check=True)


def test_example_1():
    path = "examples/1_load_lerobot_dataset.py"
    _run_script(path)
    assert Path("outputs/examples/1_load_lerobot_dataset/episode_0.mp4").exists()


@require_package("gym_pusht")
def test_examples_3_and_2():
    """
    Train a model with example 3, check the outputs.
    Evaluate the trained model with example 2, check the outputs.
    """

    path = "examples/3_train_policy.py"

    with open(path) as file:
        file_contents = file.read()

    # Do less steps, use smaller batch, use CPU, and don't complicate things with dataloader workers.
    file_contents = _find_and_replace(
        file_contents,
        [
            ("training_steps = 5000", "training_steps = 1"),
            ("num_workers=4", "num_workers=0"),
            ('device = torch.device("cuda")', 'device = torch.device("cpu")'),
            ("batch_size=64", "batch_size=1"),
        ],
    )

    # Pass empty globals to allow dictionary comprehension https://stackoverflow.com/a/32897127/4391249.
    exec(file_contents, {})

    for file_name in ["model.safetensors", "config.json"]:
        assert Path(f"outputs/train/example_pusht_diffusion/{file_name}").exists()

    path = "examples/2_evaluate_pretrained_policy.py"

    with open(path) as file:
        file_contents = file.read()

    # Do less evals, use CPU, and use the local model.
    file_contents = _find_and_replace(
        file_contents,
        [
            ('pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))', ""),
            (
                '# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")',
                'pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")',
            ),
            ('device = torch.device("cuda")', 'device = torch.device("cpu")'),
            ("step += 1", "break"),
        ],
    )

    exec(file_contents, {})

    assert Path("outputs/eval/example_pusht_diffusion/rollout.mp4").exists()
