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
import io
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


def _read_file(path):
    with open(path) as file:
        return file.read()


def test_example_1():
    path = "examples/1_load_lerobot_dataset.py"
    _run_script(path)
    assert Path("outputs/examples/1_load_lerobot_dataset/episode_0.mp4").exists()


@require_package("gym_pusht")
def test_examples_basic2_basic3_advanced1():
    """
    Train a model with example 3, check the outputs.
    Evaluate the trained model with example 2, check the outputs.
    Calculate the validation loss with advanced example 1, check the outputs.
    """

    ### Test example 3
    file_contents = _read_file("examples/3_train_policy.py")

    # Do fewer steps, use smaller batch, use CPU, and don't complicate things with dataloader workers.
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

    ### Test example 2
    file_contents = _read_file("examples/2_evaluate_pretrained_policy.py")

    # Do fewer evals, use CPU, and use the local model.
    file_contents = _find_and_replace(
        file_contents,
        [
            (
                'pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))',
                "",
            ),
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

    ## Test example 4
    file_contents = _read_file("examples/advanced/2_calculate_validation_loss.py")

    # Run on a single example from the last episode, use CPU, and use the local model.
    file_contents = _find_and_replace(
        file_contents,
        [
            (
                'pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))',
                "",
            ),
            (
                '# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")',
                'pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")',
            ),
            ('split=f"train[{first_val_frame_index}:]"', 'split="train[30:]"'),
            ("num_workers=4", "num_workers=0"),
            ('device = torch.device("cuda")', 'device = torch.device("cpu")'),
            ("batch_size=64", "batch_size=1"),
        ],
    )

    # Capture the output of the script
    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    exec(file_contents, {})
    printed_output = output_buffer.getvalue()
    # Restore stdout to its original state
    sys.stdout = sys.__stdout__
    assert "Average loss on validation set" in printed_output
